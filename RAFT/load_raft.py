import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:        
    from core.update import BasicUpdateBlock, SmallUpdateBlock
    from core.extractor import BasicEncoder, SmallEncoder
    from core.corr import CorrBlock, AlternateCorrBlock
    from core.utils.utils import bilinear_sampler, coords_grid, upflow8, InputPadder
except:
    from .core.update import BasicUpdateBlock, SmallUpdateBlock
    from .core.extractor import BasicEncoder, SmallEncoder
    from .core.corr import CorrBlock, AlternateCorrBlock
    from .core.utils.utils import bilinear_sampler, coords_grid, upflow8, InputPadder

import argparse
import os
from collections import OrderedDict

torch.cuda.set_device(0)
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7" 

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self): 
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        
        self.simsiam = self.args.simsiam

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])       
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            
            # if not self.simsiam or test_mode:
                # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)
        
        if test_mode:
            return coords1 - coords0, flow_up
            
        if self.simsiam:
            return coords1 - coords0, flow_predictions
        return flow_predictions
    

def load(path_root = ".", small = False, mixed_precision = False, alternate_corr = False, simsiam = False, data_parallel = True, load = True, DEVICE_IDS = [0,1,2,3,4,5]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",default=f'{path_root}/models/raft-sintel.pth')
    parser.add_argument('--simsiam', action='store_true', help='use simsiam model', default=simsiam)
    parser.add_argument('--small', action='store_true', help='use small model', default=small)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision',default=mixed_precision)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation',default=alternate_corr)
    args,vp = parser.parse_known_args()
    model = RAFT(args)
    if data_parallel:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)
        if load:    
            model.load_state_dict(torch.load(args.model))
    else:
        model.to(f"cuda:{DEVICE_IDS[0]}")
        if load:    
            state_dict = OrderedDict([(k.replace('module.',''), v) for k, v in torch.load(args.model).items()])
            model.load_state_dict(state_dict)
    model = model.eval()
    return model
    