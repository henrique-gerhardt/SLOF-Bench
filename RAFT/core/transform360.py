import torch
import torchvision.transforms as trans
from PIL import Image
import pathlib
from tqdm import tqdm
from torch.nn.functional import grid_sample,interpolate
import torch.nn as nn
import torch.multiprocessing as mp

torch.pi = torch.acos(torch.zeros(1)).item() * 2



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def getuvCenter(u_size=4, v_size=4):
    v = torch.linspace(90, -90, v_size).cuda().to(device)
    u = torch.linspace(-180, 180, u_size).cuda().to(device)
    vdiff = torch.abs(v[1] - v[0]).long()
    udiff = torch.abs(u[1] - u[0]).long()
    uvCenter = torch.stack(torch.meshgrid([v, u]), -1).reshape(-1, 2)
    return uvCenter.cuda().to(device), udiff, vdiff

def Te2p(e_img, h_fov, v_fov, u_deg, v_deg, out_hw, in_rot_deg=torch.tensor([0.]), mode='bilinear'):
    '''
    e_img:   ndarray in shape of [H, W, *]
    h_fov,v_fov: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    b, c, h, w = e_img.shape

    h_fov, v_fov = h_fov * torch.pi / 180., v_fov * torch.pi / 180.
    in_rot = in_rot_deg * torch.pi / 180.

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * torch.pi / 180.
    v = v_deg * torch.pi / 180.
    xyz = Txyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = Txyz2uv(xyz).cuda().to(device)
    coor_xy = Tuv2coor(uv, torch.tensor([h], dtype=float).cuda().to(device), torch.tensor([w], dtype=float).cuda().to(device)).cuda().to(device)
    return coor_xy.cuda().to(device)


def TgetCors(h_fov, v_fov, u_deg, v_deg, out_hw, in_rot_deg=torch.tensor([0.]), mode='bilinear'):
    '''
    e_img_shape:   [b,c,h,w]
    h_fov,v_fov: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''

    h_fov, v_fov = h_fov * torch.pi / 180., v_fov * torch.pi / 180.
    in_rot = in_rot_deg * torch.pi / 180.

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * torch.pi / 180.
    v = v_deg * torch.pi / 180.
    xyz = Txyzpers(h_fov, v_fov, u, v, out_hw, in_rot).cuda().to(device)
    uv = Txyz2uv(xyz).cuda().to(device)
    coor_xy = Tuv2coor(uv, torch.tensor([h], dtype=float).cuda().to(device), torch.tensor([w], dtype=float).cuda().to(device))
    return coor_xy.long().cuda().to(device)

#@profile
def Tuv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    ''' 
    u, v = torch.split(uv, 1, -1)
    u, v = u.cuda().to(device), v.cuda().to(device)
    coor_x = (u / (2 * torch.pi) + 0.5) * w - 0.5
    coor_y = (-v / torch.pi + 0.5) * h - 0.5
    return torch.cat([coor_x, coor_y], -1)


def Tcoor2uv(coorxy, h, w):
    coor_x, coor_y = torch.split(coorxy.cuda().to(device), 1, -1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * torch.pi
    v = -((coor_y + 0.5) / h - 0.5) * torch.pi
    return torch.cat([u, v], -1)


def Tuv2unitxyz(uv):
    u, v = torch.split(uv, 1, -1)
    y = torch.sin(v)
    c = torch.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)

    return torch.cat([x, y, z], dim=-1)


def Txyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = torch.split(xyz, 1, -1)
    u = torch.atan2(x, z)
    c = torch.sqrt(x ** 2 + z ** 2)
    v = torch.atan2(y, c)

    return torch.cat([u, v], -1).cuda().to(device)

def Trotation_matrix(rad, ax):
    """
    rad : torch.tensor, Eg. torch.tensor([2.0])
    ax  : torch.tensor, Eg. [1,0,0] or [0,1,0] or [0,0,1]
    """
    rad = rad.cuda().to(device)
    ax = ax.cuda().to(device)
    ax = (ax / torch.pow(ax, 2).sum())
    R = torch.diag(torch.cat([torch.cos(rad)] * 3))
    R = R + torch.outer(ax, ax) * (1.0 - torch.cos(rad))
    ax = ax * torch.sin(rad)
    R = R.cuda().to(device)
    ax = ax.cuda().to(device)
    R = R.cuda().to(device) + torch.tensor([[0, -ax[2], ax[1]],
                          [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]], dtype=ax.dtype).cuda().to(device)
    return R

def Txyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    out = torch.ones((*out_hw, 3), dtype=float).cuda().to(device)
    x_max = torch.tan(torch.tensor([h_fov / 2])).item()
    y_max = torch.tan(torch.tensor([v_fov / 2])).item()
    x_rng = torch.linspace(-x_max, x_max, out_hw[1], dtype=float).cuda().to(device)
    y_rng = torch.linspace(-y_max, y_max, out_hw[0], dtype=float).cuda().to(device)
    out[..., :2] = torch.stack(torch.meshgrid(x_rng, -y_rng), -1).permute(1, 0, 2).cuda().to(device)
    Rx = Trotation_matrix(v, torch.tensor([1, 0, 0], dtype=float)).cuda().to(device)
    Ry = Trotation_matrix(u, torch.tensor([0, 1, 0], dtype=float)).cuda().to(device)
    dots = (torch.tensor([[0, 0, 1]], dtype=float).cuda().to(device) @ Rx) @ Ry
    Ri = Trotation_matrix(in_rot, dots[0])
    return (((out @ Rx) @ Ry) @ Ri)

class Wrapper360(nn.Module):
    def __init__(self, u_size = 8, v_size = 4, pads = (1,1), nnlayer = None, scale = 3):
        super(Wrapper360, self).__init__()
        self.padu, self.padv = pads
        self.scale = scale
        self.u_size, self.v_size = u_size, v_size
        self.uvCenter, self.udiff, self.vdiff = getuvCenter(u_size, v_size)
        self.nnlayer = nnlayer
        self.fovlist = []
        self.initialized = False
        
    def idx_to_cord(self,idx):
        idx = idx.cuda().to(device)
        mid = torch.tensor([self.in_shape[3] / 2., self.in_shape[2] / 2.]).reshape(1, 1, 2).cuda().to(device)
        cords = (idx - mid) / mid
        cords = cords.unsqueeze(0).float()
        return cords
    
    def idx_interpolate(self,idx):
        top = torch.cat([torch.round(idx-i) for i in [2.0,1.5,1.0]],0)
        mid = torch.cat([torch.round(idx-i) for i in [0.5,0.0,-0.5]],0)
        bot = torch.cat([torch.round(idx+i) for i in [1.0,1.5,2.0]],0)
        idx = torch.cat([top,mid,bot],1)
        idx0,idx1 = map(lambda ix:ix.long().squeeze(),(idx).chunk(2,2))
        idx0 = idx0.clamp(min = 0, max = self.in_shape[3]-1)
        idx1 = idx1.clamp(min = 0, max = self.in_shape[2]-1)
        return idx0, idx1
        
    
    def initialize(self,shape):
        self.initialized = True
        self.in_shape = torch.tensor(shape).to(device)
        self.out_shape = self.nnlayer(torch.rand(*shape).cuda()).detach().shape
        h_fov = 360/self.u_size/shape[3]
        v_fov = 90/self.v_size
        self.out_hw = (int(v_fov), int(h_fov))
        
        for i, (v, u) in tqdm(enumerate(self.uvCenter)):
            idx = Te2p(e_img = torch.rand(*shape),
                    h_fov = h_fov+self.padu,
                    v_fov = v_fov+self.padv,
                    u_deg = torch.tensor([u]),
                    v_deg = torch.tensor([v]),
                    out_hw = self.out_hw,
                    in_rot_deg = torch.tensor([180.]),
                    mode = 'nearest')
            #self.fovlist.append([*map(lambda ix:ix.long().squeeze(),(idx).chunk(2,2))])
            self.fovlist.append(idx)
            
        self.interpolate = lambda x:interpolate(x, self.out_shape[2:])
    
    def multiforward(self,x_,x,idx):
        with torch.cuda.amp.autocast():
            if self.scale==1:
                idx0,idx1 = self.fovlist[idx]
                x_[:, :, idx1, idx0] += interpolate(self.nnlayer(x[:, :, idx1, idx0]), self.out_hw)
            else:
                idx0,idx1 = self.idx_interpolate(self.fovlist[idx])
                idx0 = idx0.to(x_.device)
                idx1 = idx1.to(x_.device)
                inter = interpolate(self.nnlayer(grid_sample(input=x, grid=self.idx_to_cord(idx), align_corners=True,
                                                       mode='bilinear')), idx0.shape)
                x_[:, :, idx1, idx0] += inter
        return x_
            
        
        
    def forward(self,x):
        x_ = torch.empty(x.size(0),self.nnlayer.out_channels, x.size(2), x.size(3)).cuda().to(device)
        
        if not self.initialized:
            self.initialize(shape = [x.size(i) for i in range(4)])
        
        
        if self.scale==1:
            for idx0,idx1 in self.fovlist:
                idx0 = idx0.cuda().to(device)
                idx1 = idx1.cuda().to(device)
                x_[:, :, idx1, idx0] += interpolate(self.nnlayer(x[:, :, idx1, idx0]), self.out_hw)
        else:
            for idx in self.fovlist:             
                idx0,idx1 = self.idx_interpolate(idx)
                idx0 = idx0.cuda().to(device)
                idx1 = idx1.cuda().to(device)
                inter = interpolate(self.nnlayer(grid_sample(input=x.float(), grid=self.idx_to_cord(idx).float().cuda().to(device), align_corners=True,
                                                   mode='bilinear')), idx0.shape)
                
                x_[:, :, idx1, idx0] += inter.float()
        
        x_ = self.interpolate(x_)
        return x_
    
    
    

class Wrapper360Update(nn.Module):
    def __init__(self, u_size = 8, v_size = 4, pads = (1,1), nnlayer = None, scale = 3):
        super(Wrapper360Update, self).__init__()
        self.padu, self.padv = pads
        self.scale = scale
        self.u_size, self.v_size = u_size, v_size
        self.uvCenter, self.udiff, self.vdiff = getuvCenter(u_size, v_size)
        self.nnlayer = nnlayer
        self.fovlist = []
        self.initialized = False
        
    def idx_to_cord(self,idx):
        idx = idx.cuda().to(device)
        mid = torch.tensor([self.in_shape[3] / 2., self.in_shape[2] / 2.]).reshape(1, 1, 2).cuda().to(device)
        cords = (idx - mid) / mid
        cords = cords.unsqueeze(0).float()
        return cords
    
    def idx_interpolate(self,idx):
        top = torch.cat([torch.round(idx-i) for i in [2.0,1.5,1.0]],0)
        mid = torch.cat([torch.round(idx-i) for i in [0.5,0.0,-0.5]],0)
        bot = torch.cat([torch.round(idx+i) for i in [1.0,1.5,2.0]],0)
        idx = torch.cat([top,mid,bot],1)
        idx0,idx1 = map(lambda ix:ix.long().squeeze(),(idx).chunk(2,2))
        idx0 = idx0.clamp(min = 0, max = self.in_shape[3]-1)
        idx1 = idx1.clamp(min = 0, max = self.in_shape[2]-1)
        return idx0, idx1
        
    
    def initialize(self,shape):
        self.initialized = True
        self.in_shape = torch.tensor(shape).to(device)
        self.out_shape = self.nnlayer(torch.rand(*shape).cuda()).detach().shape
        h_fov = 360/self.u_size/shape[3]
        v_fov = 90/self.v_size
        self.out_hw = (int(v_fov), int(h_fov))
        
        for i, (v, u) in tqdm(enumerate(self.uvCenter)):
            idx = Te2p(e_img = torch.rand(*shape),
                    h_fov = h_fov+self.padu,
                    v_fov = v_fov+self.padv,
                    u_deg = torch.tensor([u]),
                    v_deg = torch.tensor([v]),
                    out_hw = self.out_hw,
                    in_rot_deg = torch.tensor([180.]),
                    mode = 'nearest')
            #self.fovlist.append([*map(lambda ix:ix.long().squeeze(),(idx).chunk(2,2))])
            self.fovlist.append(idx)
            
        self.interpolate = lambda x:interpolate(x, self.out_shape[2:])
    
    def multiforward(self,x_,x,idx):
        with torch.cuda.amp.autocast():
            if self.scale==1:
                idx0,idx1 = self.fovlist[idx]
                x_[:, :, idx1, idx0] += interpolate(self.nnlayer(x[:, :, idx1, idx0]), self.out_hw)
            else:
                idx0,idx1 = self.idx_interpolate(self.fovlist[idx])
                idx0 = idx0.to(x_.device)
                idx1 = idx1.to(x_.device)
                inter = interpolate(self.nnlayer(grid_sample(input=x, grid=self.idx_to_cord(idx), align_corners=True,
                                                       mode='bilinear')), idx0.shape)
                x_[:, :, idx1, idx0] += inter
        return x_
            
        
        
    def forward(self,net, x, corr, flow, upsample = False):
        if not self.initialized:
            self.initialize(shape = [x.size(i) for i in range(4)])
        
        x_ = torch.empty(x.size(0),self.out_shape[1], x.size(2), x.size(3)).cuda().to(device)
        
        """
        x_ = torch.empty(x.size(0),self.nnlayer.out_channels, x.size(2), x.size(3)).cuda().to(device)
        
        if not self.initialized:
            self.initialize(shape = [x.size(i) for i in range(4)])
        """
        
        if self.scale==1:
            for idx0,idx1 in self.fovlist:
                idx0 = idx0.cuda().to(device)
                idx1 = idx1.cuda().to(device)
                x_[:, :, idx1, idx0] += interpolate(self.nnlayer(net, x[:, :, idx1, idx0],corr[:, :, idx1, idx0],flow[:, :, idx1, idx0], upsample), self.out_hw)
        else:
            for idx in self.fovlist:             
                idx0,idx1 = self.idx_interpolate(idx)
                idx0 = idx0.cuda().to(device)
                idx1 = idx1.cuda().to(device)
                inter = interpolate(self.nnlayer(grid_sample(input=x.float(), grid=self.idx_to_cord(idx).float().cuda().to(device), align_corners=True,
                                                   mode='bilinear')), idx0.shape)
                
                x_[:, :, idx1, idx0] += inter.float()
        
        x_ = self.interpolate(x_)
        return x_
    

class Wrapper360Encode(nn.Module):
    def __init__(self, u_size = 8, v_size = 4, pads = (1,1), nnlayer = None, scale = 3):
        super(Wrapper360Encode, self).__init__()
        self.padu, self.padv = pads
        self.scale = scale
        self.u_size, self.v_size = u_size, v_size
        self.uvCenter, self.udiff, self.vdiff = getuvCenter(u_size, v_size)
        self.nnlayer = nnlayer
        self.fovlist = []
        self.initialized = False
        
    def idx_to_cord(self,idx):
        idx = idx.cuda().to(device)
        mid = torch.tensor([self.in_shape[3] / 2., self.in_shape[2] / 2.]).reshape(1, 1, 2).cuda().to(device)
        cords = (idx - mid) / mid
        cords = cords.unsqueeze(0).float()
        return cords
    
    def idx_interpolate(self,idx):
        top = torch.cat([torch.round(idx-i) for i in [2.0,1.5,1.0]],0)
        mid = torch.cat([torch.round(idx-i) for i in [0.5,0.0,-0.5]],0)
        bot = torch.cat([torch.round(idx+i) for i in [1.0,1.5,2.0]],0)
        idx = torch.cat([top,mid,bot],1)
        idx0,idx1 = map(lambda ix:ix.long().squeeze(),(idx).chunk(2,2))
        idx0 = idx0.clamp(min = 0, max = self.in_shape[3]-1)
        idx1 = idx1.clamp(min = 0, max = self.in_shape[2]-1)
        return idx0, idx1
        
    
    def initialize(self,shape):
        self.initialized = True
        self.in_shape = torch.tensor(shape).to(device)
        self.out_shape = self.nnlayer(torch.rand(*shape).cuda()).detach().shape
        h_fov = 360/self.u_size/shape[3]
        v_fov = 90/self.v_size
        self.out_hw = (int(v_fov), int(h_fov))
        
        for i, (v, u) in tqdm(enumerate(self.uvCenter)):
            idx = Te2p(e_img = torch.rand(*shape),
                    h_fov = h_fov+self.padu,
                    v_fov = v_fov+self.padv,
                    u_deg = torch.tensor([u]),
                    v_deg = torch.tensor([v]),
                    out_hw = self.out_hw,
                    in_rot_deg = torch.tensor([180.]),
                    mode = 'nearest')
            #self.fovlist.append([*map(lambda ix:ix.long().squeeze(),(idx).chunk(2,2))])
            self.fovlist.append(idx)
            
        self.interpolate = lambda x:interpolate(x, self.out_shape[2:])
    
    def multiforward(self,x_,x,idx):
        with torch.cuda.amp.autocast():
            if self.scale==1:
                idx0,idx1 = self.fovlist[idx]
                x_[:, :, idx1, idx0] += interpolate(self.nnlayer(x[:, :, idx1, idx0]), self.out_hw)
            else:
                idx0,idx1 = self.idx_interpolate(self.fovlist[idx])
                idx0 = idx0.to(x_.device)
                idx1 = idx1.to(x_.device)
                inter = interpolate(self.nnlayer(grid_sample(input=x, grid=self.idx_to_cord(idx), align_corners=True,
                                                       mode='bilinear')), idx0.shape)
                x_[:, :, idx1, idx0] += inter
        return x_
            
        
        
    def forward(self,X):
        x, corr = X
        
        print(f" X CORR SIZE : {x.shape}, corrsize :{corr.shape}")
        
        if not self.initialized:
            self.initialize(shape = [x.size(i) for i in range(4)])
        
        x_ = torch.empty(x.size(0),self.out_shape[1], x.size(2), x.size(3)).cuda().to(device)
        corr_ = torch.empty(x.size(0),self.out_shape[1], x.size(2), x.size(3)).cuda().to(device)
        
        """
        x_ = torch.empty(x.size(0),self.nnlayer.out_channels, x.size(2), x.size(3)).cuda().to(device)
        
        if not self.initialized:
            self.initialize(shape = [x.size(i) for i in range(4)])
        """
        
        """
        x_ = torch.empty(x.size(0),self.nnlayer.out_channels, x.size(2), x.size(3)).cuda().to(device)
        corr_ = torch.empty(x.size(0),self.nnlayer.out_channels, x.size(2), x.size(3)).cuda().to(device)
        
        if not self.initialized:
            self.initialize(shape = [x.size(i) for i in range(4)])
        """
        
        if self.scale==1:
            for idx0,idx1 in self.fovlist:
                idx0 = idx0.cuda().to(device)
                idx1 = idx1.cuda().to(device)
                x_[:, :, idx1, idx0] += interpolate(self.nnlayer(x[:, :, idx1, idx0],corr[:, :, idx1, idx0]), self.out_hw)
        else:
            for idx in self.fovlist:             
                idx0,idx1 = self.idx_interpolate(idx)
                idx0 = idx0.cuda().to(device)
                idx1 = idx1.cuda().to(device)
                x_inter, corr_inter = self.nnlayer(grid_sample(input=x.float(), grid=self.idx_to_cord(idx).float().cuda().to(device), align_corners=True,
                                                   mode='bilinear'))
                x_inter = interpolate(x_inter, idx0.shape)
                corr_inter = nterpolate(corr_inter, idx0.shape)
                
                
                x_[:, :, idx1, idx0] += x_inter.float()
                corr_[:, :, idx1, idx0] += x_inter.float()
        
        x_ = self.interpolate(x_)
        corr_ = self.interpolate(corr_)
        return x_,corr_
    

class Wrapper360GRU(nn.Module):
    def __init__(self, u_size = 8, v_size = 4, pads = (1,1), nnlayer = None, scale = 3):
        super(Wrapper360GRU, self).__init__()
        self.padu, self.padv = pads
        self.scale = scale
        self.u_size, self.v_size = u_size, v_size
        self.uvCenter, self.udiff, self.vdiff = getuvCenter(u_size, v_size)
        self.nnlayer = nnlayer
        self.fovlist = []
        self.initialized = False
        
    def idx_to_cord(self,idx):
        idx = idx.cuda().to(device)
        mid = torch.tensor([self.in_shape[3] / 2., self.in_shape[2] / 2.]).reshape(1, 1, 2).cuda().to(device)
        cords = (idx - mid) / mid
        cords = cords.unsqueeze(0).float()
        return cords
    
    def idx_interpolate(self,idx):
        top = torch.cat([torch.round(idx-i) for i in [2.0,1.5,1.0]],0)
        mid = torch.cat([torch.round(idx-i) for i in [0.5,0.0,-0.5]],0)
        bot = torch.cat([torch.round(idx+i) for i in [1.0,1.5,2.0]],0)
        idx = torch.cat([top,mid,bot],1)
        idx0,idx1 = map(lambda ix:ix.long().squeeze(),(idx).chunk(2,2))
        idx0 = idx0.clamp(min = 0, max = self.in_shape[3]-1)
        idx1 = idx1.clamp(min = 0, max = self.in_shape[2]-1)
        return idx0, idx1
        
    
    def initialize(self,shape):
        self.initialized = True
        self.in_shape = torch.tensor(shape).to(device)
        self.out_shape = self.nnlayer(torch.rand(*shape).cuda()).detach().shape
        h_fov = 360/self.u_size/shape[3]
        v_fov = 90/self.v_size
        self.out_hw = (int(v_fov), int(h_fov))
        
        for i, (v, u) in tqdm(enumerate(self.uvCenter)):
            idx = Te2p(e_img = torch.rand(*shape),
                    h_fov = h_fov+self.padu,
                    v_fov = v_fov+self.padv,
                    u_deg = torch.tensor([u]),
                    v_deg = torch.tensor([v]),
                    out_hw = self.out_hw,
                    in_rot_deg = torch.tensor([180.]),
                    mode = 'nearest')
            #self.fovlist.append([*map(lambda ix:ix.long().squeeze(),(idx).chunk(2,2))])
            self.fovlist.append(idx)
            
        self.interpolate = lambda x:interpolate(x, self.out_shape[2:])
    
    def multiforward(self,x_,x,idx):
        with torch.cuda.amp.autocast():
            if self.scale==1:
                idx0,idx1 = self.fovlist[idx]
                x_[:, :, idx1, idx0] += interpolate(self.nnlayer(x[:, :, idx1, idx0]), self.out_hw)
            else:
                idx0,idx1 = self.idx_interpolate(self.fovlist[idx])
                idx0 = idx0.to(x_.device)
                idx1 = idx1.to(x_.device)
                inter = interpolate(self.nnlayer(grid_sample(input=x, grid=self.idx_to_cord(idx), align_corners=True,
                                                       mode='bilinear')), idx0.shape)
                x_[:, :, idx1, idx0] += inter
        return x_
            
        
        
    def forward(self,h,x):
        x_ = torch.empty(x.size(0),self.nnlayer.out_channels, x.size(2), x.size(3)).cuda().to(device)
        
        if not self.initialized:
            self.initialize(shape = [x.size(i) for i in range(4)])
        
        
        if self.scale==1:
            for idx0,idx1 in self.fovlist:
                idx0 = idx0.cuda().to(device)
                idx1 = idx1.cuda().to(device)
                x_[:, :, idx1, idx0] += interpolate(self.nnlayer(h[:, :, idx1, idx0],x[:, :, idx1, idx0]), self.out_hw)
        else:
            for idx in self.fovlist:             
                idx0,idx1 = self.idx_interpolate(idx)
                idx0 = idx0.cuda().to(device)
                idx1 = idx1.cuda().to(device)
                inter = interpolate(self.nnlayer(grid_sample(input=x.float(), grid=self.idx_to_cord(idx).float().cuda().to(device), align_corners=True,
                                                   mode='bilinear')), idx0.shape)
                
                x_[:, :, idx1, idx0] += inter.float()
        
        x_ = self.interpolate(x_)
        return x_