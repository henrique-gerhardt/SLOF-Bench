import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from pathlib import Path
from typing import *
from augmentor import FlowAugmentor, SparseFlowAugmentor
import numpy as np
from PIL import Image

try:
    from RAFT.core.utils.utils import InputPadder
    from utils import ReadData
    from equi_utils import rotate_PIL, rotate_eq, getRandomRotationConfig, flow_rotation, get3DFlow
except:
    from flow360.RAFT.core.utils.utils import InputPadder
    from flow360.utils import ReadData
    from flow360.equi_utils import rotate_PIL, rotate_eq, getRandomRotationConfig, flow_rotation, get3DFlow


class Flow360Loader(Dataset):
    def __init__(self, 
                 root_path = Path('/data/keshav/flow360/FLOW360'), 
                 mode = 'train', 
                 items = ['frame1','frame2', 'fflow', 'fflow3d'],
                 transform = {'resize' : None, 'rotation': False, 'seed' : 360}) -> None:
        """Create a dataset for flow360, with given transform and mode
        See following root_path format
        root_path/
            ├── train
            │   ├── 001
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ├── 002
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ...
            │   .   
            │   .   
            ├── val
            │   ├── 001
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ├── 002
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ...
            │   .   
            │   .   
            ├── test
            │   ├── 001
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ├── 002
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ...
            │   .   
            │   .  
        Args:
            root_path ([str], optional): [root_path as defined above]. Defaults to Path('/data/keshav/flow360/FLOW360').
            mode (str, optional): ['train', 'test', or 'val']. Defaults to 'train'.
            transform (bool, optional): [dictionary specifying resize and rotation parameter]. Defaults to {'resize' : None, 'rotation': False, 'seed': 360}.
                - resize(tuple(int(height),int(width))): resize the image and  2dflow,3dflow in given dimension
                - rotation(boolean): rotate image and 3dflow in random rotation
                - seed(int): seed to control consistent random rotation, defaults to 360
        """
        super(Flow360Loader, self).__init__()
        assert mode in ['train','val','test'], f"{mode} not in Options ['train','val','test']"
        assert set(items).issubset(set(('frame1','frame2','fflow','bflow','fflow3d','bflow3d'))), f"items : {items} not available, Valid options: ('frame1','frame2','fflow','bflow','fflow3d','bflow3d')"
        self.mode = mode
        self.items = items
        self.transform = transform
        self.datalist = ReadData(root_path=root_path, mode=mode, items = items, resize = self.transform.get('resize'))
        self.augmentor = FlowAugmentor(crop_size=(320,640))
    
    def __len__(self) -> int:
        return len(self.datalist)
    
    def transform_frame(self, frame, rotcon = None)->torch.Tensor:
        if self.transform.get('rotation'):
            frame = ToTensor()(rotate_PIL(frame, rots = rotcon))
        else:
            frame = ToTensor()(frame)
        
        return frame
    
    def transform_3dflow(self, flow, rotcon = None)-> torch.Tensor:
        flow3d = get3DFlow(flow.unsqueeze(0))
        if self.transform.get('rotation'):
            flow3d = flow_rotation(flow3d, rots = rotcon)
        return flow3d[0].permute(2,0,1)
    
    def __getitem__(self, index) -> Any:
        data = self.datalist[index]
        context = {}
        rotcon = None
        
        frame1 = data.frame1
        frame2 = data.frame2
        
        if self.mode == 'train':
            frame1 = np.asarray(frame1)
            frame2 = np.asarray(frame2)
            frame1, frame2, _ = self.augmentor(frame1, frame2, None, spatial = False)
            frame1 = Image.fromarray(frame1)
            frame2 = Image.fromarray(frame2)
        
        # if self.transform.get('rotation'):
        rotcon1 = getRandomRotationConfig(batch = 1)
        rotcon2 = getRandomRotationConfig(batch = 1)
        
        context['pitch'] = rotcon1[0].get('pitch')
        context['yaw'] = rotcon1[0].get('yaw')
        context['roll'] = rotcon1[0].get('roll')
        
        context['pitch_'] = rotcon2[0].get('pitch')
        context['yaw_'] = rotcon2[0].get('yaw')
        context['roll_'] = rotcon2[0].get('roll')
        
        context['rot'] = rotcon1
        context['rot_'] = rotcon2
        
        if 'frame1' in self.items:
            context['frame1'] = self.transform_frame(frame1, rotcon = rotcon)
        
        if 'frame2' in self.items:
            context['frame2'] = self.transform_frame(frame2, rotcon = rotcon)
        
        if 'fflow' in self.items:
            fflow = -data.fflow
            if 'fflow3d' in self.items:
                context['fflow3d'] = self.transform_3dflow(fflow, rotcon = rotcon)
                
            context['fflow'] = fflow.permute(2,0,1)
        
        if 'bflow' in self.items:
            bflow = data.bflow
            if 'bflow3d' in self.items:
                context['bflow3d'] = self.transform_3dflow(bflow, rotcon = rotcon)
            
            context['bflow'] = bflow.permute(2,0,1)
        
        context['valid'] = ((context['fflow'][0].abs() < 1000) & (context['fflow'][1].abs() < 1000)).float()
        
        return context
    

class AllLoader(object):
    def __init__(self, 
                 root_path = Path('/data/keshav/flow360/FLOW360'), 
                 modes = ['train','test','val'], 
                 items = ['frame1','frame2', 'fflow', 'fflow3d'],
                 transform = {'resize' : (320, 640), 'rotation': False, 'seed' : 360},
                 train_batch_size = 8,
                 val_batch_size = 8,
                 test_batch_size = 8,
                 train_num_workers = 0,
                 val_num_workers = 0,
                 test_num_workers = 0,
                 train_shuffle = True,
                 test_shuffle = False,
                 val_shuffle = False,
                 drop_last = True,
                 ) -> None:
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
            self.train = Flow360Loader(root_path, mode='train', items = items, transform=transform)
        
        if 'val' in modes:
            self.val = Flow360Loader(root_path, mode='val', items = items, transform=transform)
        
        if 'test' in modes:
            self.test = Flow360Loader(root_path, mode='test', items = items, transform=transform)
    
    def loadtrain(self):
        assert 'train' in self.modes, "Train dataset not found"
        return DataLoader(self.train, 
                          shuffle=self.train_shuffle, 
                          num_workers = self.train_num_workers,
                          batch_size=self.train_batch_size,
                          #prefetch_factor=self.train_num_workers * 2,
                          drop_last=self.drop_last)
    
    def loadval(self):
        assert 'val' in self.modes, "Val dataset not found"
        return DataLoader(self.val, 
                          shuffle=self.val_shuffle, 
                          num_workers = self.val_num_workers,
                          batch_size=self.val_batch_size,
                          #prefetch_factor=self.val_num_workers * 2
                          )
    
    def loadtest(self):
        assert 'test' in self.modes, "Test dataset not found"
        return DataLoader(self.test, 
                          shuffle=self.test_shuffle, 
                          num_workers = self.test_num_workers,
                          batch_size=self.test_batch_size,
                          #prefetch_factor=self.test_num_workers * 2
                          )