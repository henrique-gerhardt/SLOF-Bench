# Credits : https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved. 
"""

import torch
import torch.nn as nn
from equi_utils import flow_rotation, get3DFlow, rotate_eq
from sys import exit

class Siam360(nn.Module):
    """
    Build SiamSiam based 360 Optical flow estimator siamese network
    """
    
    def __init__(self, flow_encoder, dim = (2,320,640), finetune = False):
        super(Siam360, self).__init__()
        self.encoder = flow_encoder
        self.finetune = finetune
        if not finetune:    
        # build a projection layer
            self.projector = nn.Sequential(
                                            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, bias=False),
                                            nn.BatchNorm2d(num_features=8),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, bias=False),
                                            nn.AdaptiveAvgPool2d(output_size=(8,16)),
                                            nn.BatchNorm2d(16),
                                            nn.Flatten(),
                                            )
            
            self.predictor = nn.Sequential(nn.Linear(16*8*16, 1024, bias = False),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, 16*8*16))

    def forward(self, x1, x2, x3, x4, rots = False, rots_ = False, pitch_ = None, yaw_ = None, roll_ = None, pitch = None, roll = None, yaw = None, iters = 12):
        z1, flow_predictions1 = self.encoder(x1, x2, iters = iters)#regular flow
        if self.finetune:
            return flow_predictions1
        z2, flow_predictions2 = self.encoder(x3, x4, iters = iters)#rotational flow
        
        if rots:
            rots = [{'pitch':p, 'roll':r, 'yaw':y} for p,r,y in  zip(pitch.tolist(), roll.tolist(), yaw.tolist())]
            try:    
                z1 = rotate_eq(z1, mode = "bilinear", rots = rots, map_range=True, map_min_src=z1.min(), map_max_src = z1.max(), map_min_des=0, map_max_des=1, inverse=True)
            except Exception as e:
                torch.cuda.empty_cache()
                print(e)
                exit()
        if rots_:
            rots_ = [{'pitch':p, 'roll':r, 'yaw':y} for p,r,y in  zip(pitch_.tolist(), roll_.tolist(), yaw_.tolist())]
            try:
                z2 = rotate_eq(z2, mode = "bilinear", rots = rots_, map_range=True, map_min_src=z2.min(), map_max_src = z2.max(), map_min_des=0, map_max_des=1, inverse=True)
            except Exception as e:
                torch.cuda.empty_cache()
                print(e)
                exit()
            
        
        
        z13d = get3DFlow(z1.permute(0,2,3,1)).permute(0,3,1,2)
        z23d = get3DFlow(z2.permute(0,2,3,1)).permute(0,3,1,2)
        
        z13d = self.projector(z13d)
        
        z23d = self.projector(z23d)
        
        
        
        p1 = self.predictor(z13d)
        p2 = self.predictor(z23d)
        if rots and (not rots_):
            flow = flow_predictions2
        elif rots_ and (not rots):
            flow = flow_predictions1
        elif rots and rots_:
            flow = flow_predictions1
        else:
            print("atleast one of (rots, rots_) should be true")
            torch.cuda.empty_cache()
            exit()
        
        return p1, p2, z13d.detach(), z23d.detach(), flow

