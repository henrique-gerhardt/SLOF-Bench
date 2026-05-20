'''
Adopted from
Tobias Weis work
http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
'''
from typing import Any
import cv2
import OpenEXR
import Imath
import array
import numpy as np
import csv
import time
import datetime
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from equilib import Equi2Pers, Equi2Cube, Equi2Equi, Cube2Equi
from torchvision.transforms import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import linalg

try:    
    from equi_utils import get3DFlow
except:
    from flow360.equi_utils import get3DFlow

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def exr2depth(exr, maxvalue=1.,normalize=True):
    """ converts 1-channel exr-data to 2D numpy arrays """                                                                    
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R") ]

    # create numpy 2D-array
    img = np.zeros((sz[1],sz[0],3), np.float64)

    # normalize
    data = np.array(R)
    data[data > maxvalue] = maxvalue

    if normalize:
        data /= np.max(data)

    img = np.array(data).reshape(img.shape[0],-1)

    return img


def exr2flow(exr, h=512 ,w=1024, flowfonly = False, flowbonly = False):
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B,A) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B","A") ]

    flows = np.zeros((h,w,4), np.float64)
    flows[:,:,0] = np.array(R).reshape(flows.shape[0],-1)
    flows[:,:,1] = -np.array(G).reshape(flows.shape[0],-1)
    
    flows[:,:,2] = np.array(B).reshape(flows.shape[0],-1)
    flows[:,:,3] = -np.array(A).reshape(flows.shape[0],-1)
    
    hsvf = np.zeros((h,w,3), np.uint8)
    hsvb = np.zeros((h,w,3), np.uint8)
    
    hsvf[...,1] = 255

    magf, angf = cv2.cartToPolar(flows[...,0], flows[...,1])
    magb, angb = cv2.cartToPolar(flows[...,1], flows[...,2])
    
    hsvf[...,0] = angf*180/np.pi/2
    #hsvf[...,2] = magf
    hsvf[...,2] = cv2.normalize(magf,None,0,255,cv2.NORM_MINMAX)
    bgrf = cv2.cvtColor(hsvf,cv2.COLOR_HSV2BGR)
    
    hsvb[...,0] = angb*180/np.pi/2
    #hsvb[...,2] = magb
    hsvb[...,2] = cv2.normalize(magb,None,0,255,cv2.NORM_MINMAX)
    bgrb = cv2.cvtColor(hsvb,cv2.COLOR_HSV2BGR)
    
    flowf = flows[:,:,0:2]
    if flowfonly:
        return flowf
    flowb = flows[:,:,2:]
    
    if flowbonly:
        return flowb

    return (flowf,bgrf, magf, angf,), (flowb, bgrb, magb, angb)


def exr2normal(exr):
    """ converts 1-channel exr-data to 2D numpy arrays """                                                                    
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R","G","B") ]
    
    # create numpy 2D-array
    img = np.zeros((sz[1],sz[0],3), np.float64)
    
    R = np.array(R).reshape(*img.shape[:-1])
    G = np.array(G).reshape(*img.shape[:-1])
    B = np.array(B).reshape(*img.shape[:-1])
    return np.dstack([R,G,B])

def exr2occlusion(exr):
    """ converts 1-channel exr-data to 2D numpy arrays """                                                                    
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    R = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R") ]
    
    # create numpy 2D-array
    img = np.zeros((sz[1],sz[0],3), np.float64)
    
    R = np.array(R).reshape(*img.shape[:-1])
    return R


# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def uvzscaler(flow_uvz, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    #assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    #assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    #if clip_flow is not None:
    #    flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uvz[:,:,0]
    v = flow_uvz[:,:,1]
    z = flow_uvz[:,:,2]
    rad = np.sqrt(np.square(u) + np.square(v) + np.square(z))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    z = z / (rad_max + epsilon)
    return np.stack([u,v,z],-1)

data_root = Path("/data/keshav/flow360/DATA/")

class Data:
    def __init__(self, data_root):
        self.datalist = sorted(data_root.glob('*/**/frame/Camera_EQ_F/*'))
    
    def __len__(self):
        return len(self.datalist)
    
    def getdata(self, frame, kind = 'flow', ext = "exr", name = "Image"):
        parent = frame.parent.parent.parent/kind/frame.parent.name
        data = parent/f"{name}{frame.stem}.{ext}"
        assert data.exists(), f"{data.as_posix()} Doesn't exist"
        return data
    
    def __getitem__(self, idx):
        frame1 = self.datalist[idx]
        frame2 = frame1.parent/f"{str(int(frame1.stem)+1).zfill(4)}.png"
        
        #check if first_frame is last_frame
        if not frame2.exists():
            frame2,frame1 = frame1, frame1.parent/f"{str(int(frame1.stem)-1).zfill(4)}.png"
        
        frame1.exists(), f"{frame1.as_posix()} Frame 1 doesn't exists"
        frame2.exists(), f"{frame2.as_posix()} Frame 1 doesn't exists"
        
        fflow = self.getdata(frame2, 'corrected_fflow', ext = "npy", name = "FFlow")
        bflow = self.getdata(frame2, 'corrected_bflow', ext = "npy", name = "BFlow")
        flow = self.getdata(frame2, 'flow')
        depth1 = self.getdata(frame1, 'depth')
        depth2 = self.getdata(frame2, 'depth')
        
        normal1 = self.getdata(frame1, 'normal')
        normal2 = self.getdata(frame2, 'normal')
        
        return {'frame':(frame1, frame2),'flow':flow, 'depth':{depth1, depth2}, 'normal':(normal1, normal2),'fflow':fflow, 'bflow':bflow}
    
class FlowData:
    def __init__(self, data_root = Path("/data/keshav/flow360/DATA/")):
        self.data = Data(data_root)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        @dataclass
        class DataCollection:
            frame1,frame2 = list(map(lambda x:ToTensor()(Image.open(x)), data['frame']))
            (flowf,bgrf, magf, angf), (flowb, bgrb, magb, angb) = exr2flow(data['flow'].as_posix(), h = 512, w = 1024)
            npflowf = np.load(data['fflow'])
            npflowb = np.load(data['bflow'])
            flowpath = data['flow']
        return DataCollection
    
    
def maprange(x, minfrom=-1024, maxfrom=1024, minto=0, maxto=1, check = False):
    if check:    
        if minfrom > x.min() or maxfrom < x.max():
            x = x.clip(minfrom, maxfrom)
    return minto + ((maxto - minto)*(x - minfrom))/(maxfrom - minfrom)



# https://gist.github.com/pknowledge/4cc3c850df085fbd67838a5f22f5c761
import cv2
import numpy as np

def edge_blender(left, right, levels = 6, mode = 'horizontal'):
    def gaussianPyramid(X):
        # generate Gaussian pyramid for X
        x_ = X.copy()
        pyramid = [x_]
        for i in range(levels):
            x_ = cv2.pyrDown(x_)
            pyramid.append(x_)
        return pyramid
    
    def laplacianPyramid(X):
        # generate Laplacian Pyramid for X
        x_ = X[levels - 1]
        pyramid = [x_]
        for i in range(levels - 1, 0, -1):
            gaussian_expanded = cv2.pyrUp(X[i])
            laplacian = cv2.subtract(X[i-1], gaussian_expanded)
            pyramid.append(laplacian)
        return pyramid
    
    
    if mode == 'vertical':
        left = np.rot90(left, 1)
        right = np.rot90(right, 1)    
    
    gp_left = gaussianPyramid(left)
    gp_right = gaussianPyramid(right)
    lp_left = laplacianPyramid(gp_left)
    lp_right = laplacianPyramid(gp_right)

    # Now add left and right halves of images in each level
    lr_pyramid = []
    for left_lap, right_lap in zip(lp_left, lp_right):
        laplacian = np.hstack((left_lap, right_lap))
        lr_pyramid.append(laplacian)
    # now reconstruct
    lr_reconstruct = lr_pyramid[0]
    for i in range(1, levels):
        lr_reconstruct = cv2.pyrUp(lr_reconstruct)
        lr_reconstruct = cv2.add(lr_pyramid[i], lr_reconstruct)
    if mode == 'vertical':
        lr_reconstruct = np.rot90(lr_reconstruct, -1)
    return lr_reconstruct



def flow3d_to_RGB_image(f3d, kind = "PIL"):
    """return 3d flow visualization based on RGB color space, showing xyz displacement in sphere

    Args:
        f3d ([torch.Tensor]): BHW3
        kind (str, optional): [kind of return within options ["PIL","tensor","numpy","default"]]. Defaults to "PIL".

    Returns:
        [type]: any of ["PIL","tensor","numpy","default"]
    """
    assert kind in ["PIL","tensor","numpy","default"], f"Arg kind: {kind} not in options"
    f3d=maprange(f3d, f3d.min(), f3d.max(),0,1)
    if kind == "PIL":
        return ToPILImage()(f3d.permute(0,3,1,2)[0])
    elif kind == "tensor":
        return f3d.permute(0,3,1,2)
    elif kind == "numpy":
        return f3d[0].numpy()
    else:
        return f3d
    
def flow3d_to_RGBA_image(f3d, kind = "PIL"):
    """return 3d flow visualization based on RGBA color space, showing xy in RGB colorwheel and z in Alpha space

    Args:
        f3d ([torch.Tensor]): BHW3
        kind (str, optional): [kind of return within options ["PIL","numpy"]]. Defaults to "PIL".

    Returns:
        [type]: any of ["PIL","numpy"]
    """
    assert kind in ["PIL","numpy"], f"Arg kind: {kind} not in options"
    wheel2D = flow_to_image(f3d[0,...,:2].numpy())
    alpha = maprange(f3d[0,...,2:],f3d[0,...,2:].min().item(),f3d[0,...,2:].max().item(),0,255)
    img = np.concatenate((wheel2D, alpha),-1).astype(np.uint8)
    if kind == "PIL":
        return Image.fromarray(img)
    else:
        return img
    

def readOmniFlow(exr, h ,w, chan = ("U","V")):
    file = OpenEXR.InputFile(exr)
    # Compute the size
    dw = file.header()['dataWindow']
    
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    
    (U,V) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in chan]
        

    flows = np.zeros((h,w,2), np.float64)
    flows[:,:,0] = np.array(U).reshape(flows.shape[0],-1)
    flows[:,:,1] = np.array(V).reshape(flows.shape[0],-1)
    return flows


# CORRECT FLOW

"""
def getcube(tenflow_):
    cubes = equi2cube(tenflow_, rots = [{'roll':0,'pitch':0,'yaw':0}], w_face = 512, cube_format = "dict")[0]
    return cubes

def fixcube(cubes):
    from scipy import ndimage
    from scipy import interpolate
    thr = {'U':0.55, 'B':0.15, 'D':0.59}
    for key in ['U','B','D']:
        d_ = cubes[key].numpy()[0]
        left, right = d_[:,:252],d_[:,-252:]
        labeled, nr_objects = ndimage.label(d_> thr.get(key))
        if labeled.sum()<100:
            continue 
        scaled = torch.nn.functional.interpolate(torch.from_numpy(edge_blender(left.copy(),right.copy(), levels=3)).unsqueeze(0).unsqueeze(0), (512,512))[0][0].numpy()
        d_[np.where(labeled == 1)] = scaled[np.where(labeled == 1)]
        cubes[key][0] = torch.from_numpy(scaled)
    return cubes

all_flows = sorted(data_root.glob('*/**/flow/Camera_EQ_F/*.exr'))

def write_correct_flow(i):
    flow_path = all_flows[i]
    (flowf,bgrf, magf, angf), (flowb, bgrb, magb, angb) = utils.exr2flow(flow_path.as_posix(), h = 512, w = 1024)
    target_f_folder = flow_path.parent.parent.parent/'corrected_fflow'/flow_path.parent.name
    target_b_folder = flow_path.parent.parent.parent/'corrected_bflow'/flow_path.parent.name
    target_f = (target_f_folder/flow_path.stem.replace("Image","FFlow")).as_posix()
    target_b = (target_b_folder/flow_path.stem.replace("Image","BFlow")).as_posix()
    target_f_folder.mkdir(parents=True, exist_ok=True)
    target_b_folder.mkdir(parents=True, exist_ok=True)
    for flow,target in zip([flowf, flowb], [target_f, target_b]):
        tenflow = torch.from_numpy(flow).permute(2,0,1).unsqueeze(0)
        tenflow_ = maprange(tenflow, -40, 40, 0, 1)
        cube = getcube(tenflow_)
        cube = fixcube(cube)
        corrected_flow = maprange(cube2equi(cube, cube_format = 'dict', height = 512, width = 1024), 0, 1, -40, 40).permute(1,2,0).numpy()
        np.save(target, corrected_flow)
        
"""

from itertools import chain
class ReadData(object):
    def __init__(self, 
                 root_path = Path('/data/keshav/flow360/FLOW360'), 
                 mode = 'train',
                 items = ('frame1','frame2','fflow','bflow','fflow3d','bflow3d'),
                 resize = None) -> None:
        """See following directory structure,
        
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
            root_path   ([PosixPath]): [root path in directory structure]
            mode        ([str]): ['train', 'val', 'test']
            items       (List(str)) : [frame1, frame2, forwardflow('fflow'),backwardflow('bflow'),forward3dflow('fflow3d'),backward3dflow('bflow3d')]
        """
        super().__init__()
        assert mode in ['train', 'val', 'test'], f"Only ['train', 'val', 'test'] mode are available, mode: {mode} is invalid"
        assert set(items).issubset(set(('frame1','frame2','fflow','bflow','fflow3d','bflow3d'))), f"items : {items} not available, Valid options: ('frame1','frame2','fflow','bflow','fflow3d','bflow3d')"
        self.data_path = sorted((root_path/mode).glob('*'))
        self.frame_path = list(chain.from_iterable([sorted(x.glob('frames/*'))[:-1] for x in self.data_path]))
        self.resize = resize
        self.items = items
        
    def __len__(self) -> int:
        return len(self.frame_path)
    
    def transform_flow(self, flow)->Any:
        if self.resize:
            h, w, _ = flow.shape
            flow[:,:,0] = flow[:,:,0] / w
            flow[:,:,1] = flow[:,:,1] / h
            
            flow = torch.from_numpy(flow).permute(2,0,1).unsqueeze(0)
            flow = torch.nn.functional.interpolate(flow, self.resize)[0]
            flow[0] = flow[0] * self.resize[1]
            flow[1] = flow[1] * self.resize[0]
            flow = flow.permute(1,2,0)
            return flow
        else:
            return torch.from_numpy(flow)
    
    def transform_image(self, img)->Any:
        if img.mode == "RGBA":
            img = Image.fromarray(cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB))
        
        if self.resize:
            img = img.resize(self.resize[::-1])
        return img
    
    def __getitem__(self, key) -> Any:
        f1path = self.frame_path[key]
        f2path = f1path.parent / f"{str(int(f1path.stem)+1).zfill(4)}{f1path.suffix}"
        ffpath = Path(f1path.as_posix().replace("frames","fflows").replace(".png",".npy"))
        bfpath = Path(f1path.as_posix().replace("frames","bflows").replace(".png",".npy"))
        
        assert f1path.exists(), f"Error {f1path} doesn't exist"
        assert f2path.exists(), f"Error {f2path} doesn't exist"
        assert ffpath.exists(), f"Error {ffpath} doesn't exist"
        assert bfpath.exists(), f"Error {bfpath} doesn't exist"
        
        im1 = im2 = ff = bf = ff3d = bf3d = None
        
        if 'frame1' in self.items:    
            im1 = self.transform_image(Image.open(f1path))
        if 'frame2' in self.items:
            im2 = self.transform_image(Image.open(f2path))
        if 'fflow' in self.items:   
            ff = self.transform_flow(-np.load(ffpath))
        if 'bflow' in self.items:    
            bf = self.transform_flow(np.load(bfpath))
        if 'fflow3d' in self.items:    
            ff3d = get3DFlow(ff.unsqueeze(0))[0]
        if 'bflow3d' in self.items:    
            bf3d = get3DFlow(bf.unsqueeze(0))[0]
        
        @dataclass
        class Datum:
            frame1 = im1
            frame2 = im2
            fflow = ff
            fflow3D = ff3d
            bflow = bf
            bflow3D = bf3d
        
        return Datum
        
        

def warper(frame, flow, flow_format = "BCHW", mode = 'bilinear'):
    if flow_format!="BCHW":
        flow = flow.permute(0,3,1,2)
    
    B,C,H,W = frame.shape
    
    grid = torch.stack(torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,640))[::-1], 0)
    grid = torch.stack([grid]*frame.size(0), 0)
    scale = torch.tensor([W,H]).view(1,-1,1,1)
    
    if frame.is_cuda:
        grid = grid.float().cuda().to(frame.device)
        scale = scale.float().cuda().to(frame.device)
    flow = flow/scale
    grid = grid - flow
    return torch.nn.functional.grid_sample(input = frame, mode = mode, grid = grid.permute(0,2,3,1), align_corners = True)

def computemask(fflow, bflow, thr = 0.5):
    bflowmask = (torch.pow(bflow, 2).sum(1, keepdim = True).sqrt() > thr).float()
    flowmask_ = warper(bflowmask.float(), bflow.float(), mode = 'nearest')
    fflowmask = (torch.pow(fflow, 2).sum(1, keepdim = True).sqrt() <= thr).float()
    mask = 1 - fflowmask * flowmask_
    return mask.squeeze(1)


def stop(write = True):
    with open("externalinterrupt.txt", "r") as interrupt:
            check = interrupt.read()
        
    if check == "stop":
        print("program stopped by external interrupt")
        torch.cuda.empty_cache()
        if write:    
            with open("externalinterrupt.txt", "w") as interrupt:
                interrupt.write("continue")
        
        return True
    else:
        return False
    
    
