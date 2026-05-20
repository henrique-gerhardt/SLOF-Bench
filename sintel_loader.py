import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from pathlib import Path
from typing import *
from itertools import chain
import numpy as np
import random
from PIL import Image
from flowutils import  flow_read
from augmentor import FlowAugmentor, SparseFlowAugmentor

try:
    from RAFT.core.utils.utils import InputPadder
    from utils import ReadData
    from equi_utils import rotate_PIL, rotate_eq, getRandomRotationConfig, flow_rotation, get3DFlow
except:
    from flow360.RAFT.core.utils.utils import InputPadder
    from flow360.utils import ReadData
    from flow360.equi_utils import rotate_PIL, rotate_eq, getRandomRotationConfig, flow_rotation, get3DFlow
    

class Sintel(Dataset):
    def __init__(self, 
                 root_path = Path('/data/keshav/sintel/training/'),
                 mode = 'train',
                 test_percent = 0.10,
                 items = ['frame1', 'frame2', 'fflow', 'fflow3d'],
                 transform = {'resize' : [320,640], 'rotation': False, 'seed' : 360,}) -> None:
        super().__init__()
        assert mode in ['train','val','test'], f"{mode} not in Options ['train','val','test']"
        assert set(items).issubset(set(('frame1','frame2','fflow','fflow3d'))), f"items : {items} not available, Valid options: ('frame1','frame2','fflow','fflow3d')"
        self.items = items
        self.transform = transform
        self.augmentor = FlowAugmentor(crop_size = (320,640)) if mode == 'train' else None
        
        #getting all frames
        frames = sorted((root_path/"final/").glob('*'))
        frames = sorted(chain.from_iterable([sorted(f.glob('*.png'))[:-1] for f in frames]))
        
        #getting all invalid
        invalid = sorted((root_path/"invalid/").glob('*'))
        invalid = sorted(chain.from_iterable([sorted(f.glob('*.png'))[:-1] for f in invalid]))
        
        #getting all flows
        flows = sorted(Path(root_path/"flow/").glob('*/*.flo'))
        
        self.frame_list = np.array(frames)
        self.invalid_list =  np.array(invalid)
        self.flows_list = np.array(flows)
        idx = np.arange(len(frames))
        
        random.seed(transform.get('seed'))
        np.random.seed(transform.get('seed'))
        
        self.test_idx = np.random.choice(idx, int(len(frames)*test_percent), replace = False)
        self.train_idx = idx[~np.isin(idx, self.test_idx)]
        
        if mode == 'train':
            self.idlist = self.train_idx
        else:
            self.idlist = self.test_idx
            
        random.seed()
        np.random.seed()
    
    def __len__(self)->int:
        return len(self.idlist)
    
    def process_flow(self, flow)->Any:
        #Assumption flow shape is HW2
        flow = torch.from_numpy(flow).permute(2,0,1)
        _, H, W = flow.shape
        if self.transform.get('resize'):
            flow = flow.unsqueeze(0)
            if self.augmentor is None:
                flow = flow/torch.tensor([W, H]).view(1,-1,1,1)
                flow = torch.nn.functional.interpolate(flow, self.transform.get('resize'))
                flow = flow * torch.tensor(self.transform.get('resize')[::-1]).view(1,-1,1,1)
            flow = flow[0]
        return flow.float()
    
    def process_frame(self, frame, rotcon = None)->Any:
        if self.transform.get('resize'):
            if self.augmentor is None:    
                frame = frame.resize((self.transform.get('resize')[::-1]))
            if rotcon:
                frame = rotate_PIL(frame, rots = rotcon)
        #frame = ToTensor()(frame)
        return frame
    
    def process_invalid(self, invalid, rotcon = None)->Any:
        if self.transform.get('resize'):
            if self.augmentor is None:    
                invalid = invalid.resize((self.transform.get('resize')[::-1]))
            if rotcon:
                invalid = rotate_PIL(invalid, rots = rotcon)
        #invalid = ToTensor()(invalid)
        return invalid
    
    def transform_3dflow(self, flow, rotcon = None)-> torch.Tensor:
        flow3d = get3DFlow(flow.unsqueeze(0), formats = "BCHW")
        if self.transform.get('rotation'):
            flow3d = flow_rotation(flow3d, rots = rotcon, formats = "BCHW")
            flow = rotate_eq(flow.unsqueeze(0), mode = "bilinear", map_range = True, rots = rotcon, map_min_src=flow.min(), map_max_src=flow.max())[0]
        return flow, flow3d[0]
    
    def __getitem__(self, index) -> Any:
        idx = self.idlist[index]
        
        context = {}
        rotcon = None
        
        if self.transform.get('rotation'):
            rotcon = getRandomRotationConfig(batch = 1,seed = self.transform.get('seed'))
            context['rot'] = rotcon
        
        frame_path = self.frame_list[idx]
        invalid_path = self.invalid_list[idx]
        flow_path = self.flows_list[idx]
        
        frame1 = np.array(self.process_frame(Image.open(frame_path), rotcon))
        frame2 = np.array(self.process_frame(Image.open(frame_path.parent/f"frame_{str(int(frame_path.stem.replace('frame_',''))+1).zfill(4)}.png"), rotcon))
        flow = flow_read(flow_path.as_posix())
        
        if self.augmentor:
            frame1, frame2, flow = self.augmentor(frame1, frame2, flow)
        
        flow = self.process_flow(flow)
        
        frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float()
        frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float()
        
        if 'frame1' in self.items:
            context['frame1'] = frame1
        if 'frame2' in self.items:
            context['frame2'] = frame2
        if 'fflow' in self.items:
            
            if 'fflow3d' in self.items:
                flow, flow3d = self.transform_3dflow(flow, rotcon = rotcon)
                context['fflow3d'] = flow3d
                
            context['fflow'] = flow
        
        invalid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)#self.process_invalid(Image.open(invalid_path), rotcon)
        
        context['invalid'] = invalid.float()
            
        return context

class AllSintelLoader(object):
    def __init__(self,
                 root_path = Path('/data/keshav/sintel/training/'), 
                 modes = ['train','test'], 
                 items = ['frame1','frame2', 'fflow', 'fflow3d'],
                 transform = {'resize' : (320, 640), 'rotation': False, 'seed' : 360},
                 train_batch_size = 16,
                 val_batch_size = 16,
                 test_batch_size = 16,
                 train_num_workers = 0,
                 val_num_workers = 0,
                 test_num_workers = 0,
                 train_shuffle = True,
                 test_shuffle = False,
                 val_shuffle = False,
                 drop_last = True,
                 )->None:
        super().__init__()
        self.drop_last = drop_last
        self.root_path = root_path
        self.modes = modes
        self.items = items
        self.transform = transform
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers
        self.train_shuffle = train_shuffle
        self.test_shuffle = test_shuffle
        self.val_shuffle = val_shuffle
        
        if 'train' in modes:
            self.train = Sintel(root_path, mode='train', items = items, transform=transform)
        
        
        if 'test' in modes:
            self.test = Sintel(root_path, mode='test', items = items, transform=transform)
    
    def loadtrain(self):
        assert 'train' in self.modes, "Train dataset not found"
        return DataLoader(self.train, 
                          shuffle=self.train_shuffle, 
                          num_workers = self.train_num_workers,
                          batch_size=self.train_batch_size,
                          #prefetch_factor=self.train_num_workers * 2,
                          drop_last=self.drop_last)
        
    def loadtest(self):
        assert 'test' in self.modes, "Test dataset not found"
        return DataLoader(self.test, 
                          shuffle=self.test_shuffle, 
                          num_workers = self.test_num_workers,
                          batch_size=self.test_batch_size,
                          #prefetch_factor=self.test_num_workers * 2
                          )
        
        
        
        
        
        
        
    
        