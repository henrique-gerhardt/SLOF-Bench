#!/usr/bin/env python3

from typing import Dict, List, Optional

import torch
import random


from equilib.grid_sample import torch_grid_sample
from equilib.torch_utils import (
    create_normalized_grid,
    create_rotation_matrices,
    get_device,
    pi,
)

from torchvision.transforms import ToPILImage, ToTensor



def maprange(x, minfrom=-1024, maxfrom=1024, minto=0, maxto=1):
    if minfrom > x.min() or maxfrom < x.max():
        x = x.clip(minfrom, maxfrom)
    return minto + ((maxto - minto)*(x - minfrom))/(maxfrom - minfrom)

def getRandomRotationConfig(batch = 1, rmin = 0, rmax = 180, seed = None):
    """generate random rotation config with different yaw, pitch and roll

    Args:
        batch (int, optional): [number of configs]. Defaults to 1.
        rmin (int, optional): [minimum rand range]. Defaults to -180.
        rmax (int, optional): [maximum rand range]. Defaults to 180.
        seed ([type], optional): [random seed]. Defaults to None.

    Returns:
        [List of dict]: [Return list of dictionary of rotation configs]
    """
    random.seed(seed)
    r = lambda : random.randint(rmin,rmax)
    config = list(map(lambda x:dict(roll = r(), pitch = r(), yaw = r()), range(batch)))
    random.seed()
    return config


def matmul(
    m: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:

    M = torch.matmul(R[:, None, None, ...], m)
    M = M.squeeze(-1)

    return M


def convert_grid(
    M: torch.Tensor,
    h_equi: int,
    w_equi: int,
    method: str = "robust",
) -> torch.Tensor:

    # convert to rotation
    phi = torch.asin(M[..., 2] / torch.norm(M, dim=-1))
    theta = torch.atan2(M[..., 1], M[..., 0])

    if method == "robust":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui %= w_equi
        uj %= h_equi
    elif method == "faster":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui = torch.where(ui < 0, ui + w_equi, ui)
        ui = torch.where(ui >= w_equi, ui - w_equi, ui)
        uj = torch.where(uj < 0, uj + h_equi, uj)
        uj = torch.where(uj >= h_equi, uj - h_equi, uj)
    else:
        raise ValueError(f"ERR: {method} is not supported")

    # stack the pixel maps into a grid
    grid = torch.stack((uj, ui), dim=-3)

    return grid


def rotate_eq(
    src: torch.Tensor,
    rots: List[Dict[str, float]],
    mode: str,
    z_down: bool = False,
    height: Optional[int] = None,
    width: Optional[int] = None,
    backend: str = "native",
    inverse = False,
    map_range: bool = False,
    map_min_src: Optional[int] = -1,
    map_max_src: Optional[int] = 1,
    map_min_des: Optional[int] = 0,
    map_max_des: Optional[int] = 1 
) -> torch.Tensor:
    """Run Equi2Equi

    params:
    - src (torch.Tensor): 4 dims (b, c, h, w)
    - rot (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - z_down (bool)
    - mode (str): sampling mode for grid_sample
    - height, width (Optional[int]): height and width of the target
    - backend (str): backend of torch `grid_sample` (default: `native`)
    - map_range (bool) : map ranges if True
    - map_min_src (int): minimum_map range of source
    - map_max_src (int): maximum_map range of source
    - map_min_des (int): minimum_map range of destination
    - map_max_des (int): maximum_map range of destination

    return:
    - out (torch.Tensor)

    NOTE: acceptable dtypes for `src` are currently uint8, float32, and float64.
    Floats are prefered since numpy calculations are optimized for floats.

    NOTE: output array has the same dtype as `src`

    NOTE: you can override `equilib`'s grid_sample with over grid sampling methods
    using `override_func`. The input to this function have to match `grid_sample`.

    """

    assert (
        len(src.shape) == 4
    ), f"ERR: input `src` should be 4-dim (b, c, h, w), but got {len(src.shape)}"
    assert len(src) == len(
        rots
    ), f"ERR: length of `src` and `rot` differs: {len(src)} vs {len(rots)}"
    
    if map_range:
        src = maprange(src, map_min_src, map_max_src, map_min_des, map_max_des)

    src_dtype = src.dtype
    assert src_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input equirectangular image has dtype of {src_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    if src.device.type == "cuda":
        dtype = torch.float32 if src_dtype == torch.uint8 else src_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if src_dtype == torch.uint8 else src_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )
    if backend == "native" and src_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        src = src.type(torch.float32)

    bs, c, h_equi, w_equi = src.shape
    src_device = get_device(src)

    assert type(height) == type(
        width
    ), "ERR: `height` and `width` does not match types (maybe it was set separately?)"
    if height is None and width is None:
        height = h_equi
        width = w_equi
    else:
        assert isinstance(height, int) and isinstance(width, int)

    # initialize output tensor
    if backend == "native":
        # NOTE: don't need to initialize for `native`
        out = None
    else:
        out = torch.empty(
            (bs, c, height, width),  # type: ignore
            dtype=dtype,
            device=src_device,
        )

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if src.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    m = create_normalized_grid(
        height=height,
        width=width,
        batch=bs,
        dtype=tmp_dtype,
        device=tmp_device,
    )
    m = m.unsqueeze(-1)

    # create batched rotation matrices
    R = create_rotation_matrices(
        rots=rots,
        z_down=z_down,
        dtype=tmp_dtype,
        device=tmp_device,
    )
    
    if inverse:
        R = R.transpose(1,2)
        
    # rotate the grid
    M = matmul(m, R)

    grid = convert_grid(
        M=M,
        h_equi=h_equi,
        w_equi=w_equi,
        method="robust",
    )

    # FIXME: putting `grid` to device since `pure`'s bilinear interpolation requires it
    # FIXME: better way of forcing `grid` to be the same dtype?
    if src.dtype != grid.dtype:
        grid = grid.type(src.dtype)
    if src.device != grid.device:
        grid = grid.to(src.device)

    # grid sample
    out = torch_grid_sample(
        img=src,
        grid=grid,
        out=out,  # FIXME: is this necessary?
        mode=mode,
        backend=backend,
    )

    # NOTE: we assume that `out` keeps it's dtype

    out = (
        out.type(src_dtype)
        if src_dtype == torch.uint8
        else torch.clip(out, 0.0, 1.0)
    )
    
    if map_range:
        out = maprange(out, map_min_des, map_max_des, map_min_src, map_max_src)

    return out



def get3DFlow(flow, formats = "BHWC"):
    """Map Planar flow to spherical flow

    Args:
        flow ([torch.Tensor]): A torch tensor of size BHW2

    Returns:
        [spherical flow]: [A spherical flow of size BHW3, torch.Tensor type]
    """
    # flow is a torch tensor
    # check if the format is matched
    if formats!= "BHWC":
        flow = flow.permute(0,2,3,1)
        
    batch, height, width, channel = flow.shape
    
    if flow.is_cuda:
        picuda = pi.cuda().to(flow.device)
    else:
        picuda = pi
    
    def getgrid(grid):
        theta = grid.select(-1, 0) * 2 * picuda / width - picuda
        phi = grid.select(-1,1) * picuda / height - picuda / 2
        
        if flow.is_cuda:
            theta = theta.cuda().to(flow.device)
            phi = phi.cuda().to(flow.device)
        
        a = torch.stack((theta, phi), dim=-1)
        
        norm_A = 1
        x = norm_A * torch.cos(a[..., 1]) * torch.cos(a[..., 0])
        y = norm_A * torch.cos(a[..., 1]) * torch.sin(a[..., 0])
        z = norm_A * torch.sin(a[..., 1])
        return torch.stack((x, y, z), dim=-1)
        
    xs = torch.linspace(0, width - 1, width)
    ys = torch.linspace(0, height - 1, height)
    
    ys, xs = torch.meshgrid([ys, xs])
    
    ngrid = torch.stack((xs, ys), dim=-1).unsqueeze(0)
    
    if flow.is_cuda:
        ngrid = ngrid.cuda().to(flow.device)
    
    fgrid = ngrid - flow
    
    
    # boundary condition for pixel on x displacement
    # fgrid[...,0][fgrid[...,0] < 0] = (width - 1) + fgrid[...,0][fgrid[...,0] < 0]
    # fgrid[...,0][fgrid[...,0] > width - 1] = fgrid[...,0][fgrid[...,0] > (width - 1)] - (width - 1)
    # fgrid[...,1][fgrid[...,1] < 0] = -fgrid[...,1][fgrid[...,1] < 0]
    # fgrid[...,1][fgrid[...,1] > (height-1)] = fgrid[...,1][fgrid[...,1] > (height-1)] - (height-1)
    
    fgrid = getgrid(fgrid)
    ngrid = getgrid(ngrid)
    f3d = ngrid - fgrid
    
    if formats != "BHWC":
        f3d = f3d.permute(0,3,1,2)
    
    return f3d

def rotate_PIL(img, rots):
    img = ToTensor()(img).unsqueeze(0)
    return ToPILImage()(rotate_eq(img, rots = rots, mode = "bilinear")[0])

def flow_rotation(f3d, 
    rots: List[Dict], 
    inverse: bool = False, 
    formats: str = "BHWC", 
    map_min_src: Optional[int] = -1,
    map_max_src: Optional[int] = 1,
    map_min_des: Optional[int] = 0,
    map_max_des: Optional[int] = 1):
    """Perform flow rotation by scaling min max to 0-1 and rescaling back to original min max

    Args:
        - f3d ([torch.Tensor]): [3D flow tensor of size B3HW or BHW3]
        - rots (Dict[List]): [rotation configuration contatininb pitch, yaw and roll information]
        - inverse (bool, optional): [perform reverse operation if enalbe]. Defaults to False.
        - formats (str, optional): [specify data format BCHW or BHWC]. Defaults to "BHWC".
        - map_min_src (int): minimum_map range of source
        - map_max_src (int): maximum_map range of source
        - map_min_des (int): minimum_map range of destination
        - map_max_des (int): maximum_map range of destination.

    Returns:
        [type]: [description]
    """
    if formats != "BCHW":
        f3d = f3d.permute(0,3,1,2)
    f3d = rotate_eq(f3d, 
                    rots = rots, 
                    mode = 'bilinear', 
                    map_range=True, 
                    inverse=inverse,
                    map_min_src = map_min_src,
                    map_max_src = map_max_src,
                    map_min_des = map_min_des,
                    map_max_des = map_max_des)
    if formats != "BCHW":
        f3d = f3d.permute(0,2,3,1)
    return f3d



# from matplotlib.ticker import PercentFormatter
# import numpy as np
# plt.hist(flowf[:,:,0].flatten(), weights=np.ones(len(flowf[:,:,0].flatten())) / len(flowf[:,:,0].flatten()))

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()