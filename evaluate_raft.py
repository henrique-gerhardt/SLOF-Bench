from RAFT import load_raft
from dataloader import AllLoader
import torch
from equi_utils import get3DFlow, flow_rotation
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from utils import maprange, stop


def getdensitymask(npy = False):
    dist = (1 - np.load("distortiondensity.npy"))
    dist = maprange(dist, minfrom = dist.min(), maxfrom = dist.max(), minto = 0.500, maxto=1.000)
    if npy:
        return dist
    dist = torch.from_numpy(dist).unsqueeze(0)
    dist = dist.cuda()
    return dist

def epe_with_range(flow, flow_, distances, predicate, USE_DENISTY_MASK = False):
    if predicate is None:
        epe = torch.sum(distances, dim = 1).sqrt()
        
        if USE_DENISTY_MASK:
            epe = epe / getdensitymask()
            
        epe = epe.view(epe.shape[0],-1).detach().cpu().numpy()
        return epe
    else:
        magf = torch.pow(flow,2).sum(1).sqrt()
        mask = predicate(magf).unsqueeze(1)
        epe = torch.sum(distances * mask, dim = 1).sqrt()
        
        if USE_DENISTY_MASK:
            epe = epe / getdensitymask()
        
        epe = epe[mask.squeeze(1)]
        epe = epe.view(-1).detach().cpu().numpy()
        return epe
        

def angularError(flow, flow_, epsilon = 1e-10, predicate = None, USE_DENISTY_MASK = False):
    #flowshape: b2hw
    ugt = flow.select(1,0)
    vgt = flow.select(1,1)
    
    
    u = flow_.select(1,0)
    v = flow_.select(1,1)
    
    ugt = ugt/(ugt**2+vgt**2 + epsilon).sqrt()
    vgt = vgt/(ugt**2+vgt**2 + epsilon).sqrt()
    
    u = u/(u**2 + v**2 + epsilon).sqrt()
    v = v/(u**2 + v**2 + epsilon).sqrt()
    
    var = ((ugt*u+v*vgt+1)/((u**2+v**2+1).sqrt()*(ugt**2 + vgt**2 + 1).sqrt()))
    
    
    var = maprange(var, minfrom=var.min(), maxfrom=var.max(), minto=-1, maxto=1)
    
    ae = torch.acos(var)
    
    if USE_DENISTY_MASK:
        ae = (ae / getdensitymask())#.clip(0, np.pi)
    
    
    if predicate:
        magf = torch.pow(flow,2).sum(1).sqrt()
        mask = predicate(magf)
        ae = ae[mask]
        return ae.view(-1).detach().cpu().numpy()
    else:
        return ae.view(ae.shape[0],-1).detach().cpu().numpy()
    
    
    
    
from pathlib import Path
@torch.no_grad()
def validate_flow360(raft, root_path = Path('/data/keshav/flow360/FLOW360_train_test'), mode = 'test', iters = 64, rotation = False, filename = None, save = True, weighted = False):
    loader = AllLoader(root_path = root_path, 
                       modes = [mode], 
                       transform = {'resize' : (320, 640), 
                                    'rotation': rotation, 
                                    'seed' : 360},
                       val_batch_size = 16,
                       test_batch_size = 16)
    validating = loader.loadval() if mode == 'val' else loader.loadtest()
    
    DEVICE = "cuda:0"
    raft.eval()
    
    validating = tqdm(validating)
    validating.set_description_str(f"[{mode.upper()}]")
    
    epe_2d_all = []
    epe_2d_lt_5 = []
    epe_2d_lt_10 = []
    epe_2d_lt_20 = []
    epe_2d_gte_20 = []
    
    angular_2d_all = []
    angular_2d_lt_5 = []
    angular_2d_lt_10 = []
    angular_2d_lt_20 = []
    angular_2d_gte_20 = []
    
    
    for val in validating:
        if stop():
            break
        frame1 = val['frame1'].cuda().to(DEVICE)
        frame2 = val['frame2'].cuda().to(DEVICE)
        if rotation:    
            rotcon = val['rot']
            roll, pitch, yaw = rotcon['roll'], rotcon['pitch'], rotcon['yaw']
            rots = [{'roll':r, 'pitch':p, 'yaw':y} for r,p,y in zip(roll, pitch, yaw)]
            
        fflow = val['fflow']
        
        if rotation:
            fflow = flow_rotation(fflow, rots = rots, formats="BCHW", map_min_src=fflow.min(), map_max_src=fflow.max())
        
        fflow = fflow.cuda().to(DEVICE)
        
        fflow3d = val['fflow3d'].cuda().to(DEVICE)
        
        
        _, fflow_ = raft(image1 = frame1, image2 = frame2, iters = iters, test_mode = True)
        
        fflow3d_ = get3DFlow(fflow_, formats = "BCHW")
        
        dist2d = torch.pow(fflow - fflow_, 2)
        
        epe_2d_all.append(epe_with_range(fflow, fflow_, dist2d, predicate = None, USE_DENISTY_MASK=weighted))
        epe_2d_lt_5.append(epe_with_range(fflow, fflow_, dist2d, predicate = lambda x:x<5.0, USE_DENISTY_MASK=weighted))
        epe_2d_lt_10.append(epe_with_range(fflow, fflow_, dist2d, predicate = lambda x:x<10.0, USE_DENISTY_MASK=weighted))
        epe_2d_lt_20.append(epe_with_range(fflow, fflow_, dist2d, predicate = lambda x:x<20.0, USE_DENISTY_MASK=weighted))
        epe_2d_gte_20.append(epe_with_range(fflow, fflow_, dist2d, predicate = lambda x:x>=20.0, USE_DENISTY_MASK=weighted))
        
        angular_2d_all.append(angularError(fflow, fflow_, USE_DENISTY_MASK=weighted))
        angular_2d_lt_5.append(angularError(fflow, fflow_, predicate = lambda x:x<5.0, USE_DENISTY_MASK=weighted))
        angular_2d_lt_10.append(angularError(fflow, fflow_, predicate = lambda x:x<10.0, USE_DENISTY_MASK=weighted))
        angular_2d_lt_20.append(angularError(fflow, fflow_, predicate = lambda x:x<20.0, USE_DENISTY_MASK=weighted))
        angular_2d_gte_20.append(angularError(fflow, fflow_, predicate = lambda x:x>=20.0, USE_DENISTY_MASK=weighted))
        
    
    np_epe_2d_all = np.concatenate(epe_2d_all)
    np_epe_2d_lt_5 = np.concatenate(epe_2d_lt_5)
    np_epe_2d_lt_10 = np.concatenate(epe_2d_lt_10)
    np_epe_2d_lt_20 = np.concatenate(epe_2d_lt_20)
    np_epe_2d_gte_20 = np.concatenate(epe_2d_gte_20)
    np_angular_2d_all = np.concatenate(angular_2d_all)
    np_angular_2d_lt_5 = np.concatenate(angular_2d_lt_5)
    np_angular_2d_lt_10 = np.concatenate(angular_2d_lt_10)
    np_angular_2d_lt_20 = np.concatenate(angular_2d_lt_20)
    np_angular_2d_gte_20 = np.concatenate(angular_2d_gte_20)
    
    M = lambda x:np.mean(x)
    
    
    results = {
        'all':[
            M(np_epe_2d_all),
            M(np_angular_2d_all),
        ],
        
        'lt5' : [
            M(np_epe_2d_lt_5),
            M(np_angular_2d_lt_5),
        ], 
        
        'lt10' : [
            M(np_epe_2d_lt_10),
            M(np_angular_2d_lt_10),
        ],
        
        'lt20' : [
            M(np_epe_2d_lt_20),
            M(np_angular_2d_lt_20),
        ], 
        
        'gte20' : [
            M(np_epe_2d_gte_20),
            M(np_angular_2d_gte_20),
        ],
        
        'metric' : [
            'epe',
            'ae',
        ],
        
        'mode' : [
            '2DRawFlow',
            '2DRawFlow',
        ]
    }
    
    data = pd.DataFrame(results)
    print(data)
    
    if save:
        if filename:    
            data.to_csv(f"quantitative_results/{filename.upper()}_{mode}_iters_{iters}_rotation_{rotation}_final.csv", index = None)
        else:
            data.to_csv(f"quantitative_results/{mode}_iters_{iters}_rotation_{rotation}_final.csv", index = None)
    return results['all'][0], data, f"quantitative_results/{filename.upper()}_{mode}_iters_{iters}_rotation_{rotation}_final.csv"

if __name__ == '__main__':
    
    import sys
    try:
        iters = int(sys.argv[1])
        rotation = sys.argv[1] == '1'
    except:
        raise Exception("Please enter number of iteration as additional argument and 1 or 0 as extra argument for rotation")
    
    raft = load_raft.load(path_root="RAFT")
    validate_flow360(raft, mode="test", iters = iters, rotation = rotation, filename = "RAFT_PRETRAIN_SINTEL")