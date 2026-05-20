from dataloader import Flow360Loader
from pathlib import Path
from torchvision.transforms import ToPILImage, ToTensor
from simsiam import Siam360
from RAFT import load_raft
import torch
import torch.nn as nn
from utils import flow_to_image
from PIL import Image
import matplotlib.pyplot as plt
from KernelTransformerNetwork.KernelTransformer import KTNLayer
FINETUNE = True
load = True
DEVICE_IDS = [0,1,2,3,4,5,6,7]
DEVICE = "cuda:0"
iters = 12

def getKTNisedRaft():
    raft = load_raft.load(path_root="RAFT", data_parallel=False, simsiam=True, load = load).train(False)
    #transforming fnet
    raft.fnet.conv1 = KTNLayer.KTNConv(raft.fnet.conv1.weight.data.cuda(), raft.fnet.conv1.bias.data.cuda(), sphereH = 320, imgW = 640, tied_weights = 20, output_shape = (160,320)).cuda()
    raft.fnet.layer1[0].conv1 = KTNLayer.KTNConv(raft.fnet.layer1[0].conv1.weight.data.cuda(), raft.fnet.layer1[0].conv1.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20).cuda()
    raft.fnet.layer1[0].conv2 = KTNLayer.KTNConv(raft.fnet.layer1[0].conv2.weight.data.cuda(), raft.fnet.layer1[0].conv2.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20).cuda()
    raft.fnet.layer1[1].conv1 = KTNLayer.KTNConv(raft.fnet.layer1[1].conv1.weight.data.cuda(), raft.fnet.layer1[1].conv1.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20).cuda()
    raft.fnet.layer1[1].conv2 = KTNLayer.KTNConv(raft.fnet.layer1[1].conv2.weight.data.cuda(), raft.fnet.layer1[1].conv2.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20).cuda()

    raft.fnet.layer2[0].conv1 = KTNLayer.KTNConv(raft.fnet.layer2[0].conv1.weight.data.cuda(), raft.fnet.layer2[0].conv1.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20, output_shape = (80,160)).cuda()
    raft.fnet.layer2[0].conv2 = KTNLayer.KTNConv(raft.fnet.layer2[0].conv2.weight.data.cuda(), raft.fnet.layer2[0].conv2.bias.data.cuda(), sphereH = 80, imgW = 160,tied_weights = 20).cuda()
    raft.fnet.layer2[1].conv1 = KTNLayer.KTNConv(raft.fnet.layer2[1].conv1.weight.data.cuda(), raft.fnet.layer2[1].conv1.bias.data.cuda(), sphereH = 80, imgW = 160,tied_weights = 20).cuda()
    raft.fnet.layer2[1].conv2 = KTNLayer.KTNConv(raft.fnet.layer2[1].conv2.weight.data.cuda(), raft.fnet.layer2[1].conv2.bias.data.cuda(), sphereH = 80, imgW = 160,tied_weights = 20).cuda()

    raft.fnet.layer3[0].conv1 = KTNLayer.KTNConv(raft.fnet.layer3[0].conv1.weight.data.cuda(), raft.fnet.layer3[0].conv1.bias.data.cuda(), sphereH = 80, imgW = 160,tied_weights = 20, output_shape = (40,80))
    raft.fnet.layer3[0].conv2 = KTNLayer.KTNConv(raft.fnet.layer3[0].conv2.weight.data.cuda(), raft.fnet.layer3[0].conv2.bias.data.cuda(), sphereH = 40, imgW = 80,tied_weights = 5)
    raft.fnet.layer3[1].conv1 = KTNLayer.KTNConv(raft.fnet.layer3[1].conv1.weight.data.cuda(), raft.fnet.layer3[1].conv1.bias.data.cuda(), sphereH = 40, imgW = 80,tied_weights = 5)
    raft.fnet.layer3[1].conv2 = KTNLayer.KTNConv(raft.fnet.layer3[1].conv2.weight.data.cuda(), raft.fnet.layer3[1].conv2.bias.data.cuda(), sphereH = 40, imgW = 80,tied_weights = 5)
    
    #transforming cnet
    raft.cnet.conv1 = KTNLayer.KTNConv(raft.cnet.conv1.weight.data.cuda(), raft.cnet.conv1.bias.data.cuda(), sphereH = 320, imgW = 640, tied_weights = 20, output_shape = (160,320)).cuda()
    raft.cnet.layer1[0].conv1 = KTNLayer.KTNConv(raft.cnet.layer1[0].conv1.weight.data.cuda(), raft.cnet.layer1[0].conv1.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20).cuda()
    raft.cnet.layer1[0].conv2 = KTNLayer.KTNConv(raft.cnet.layer1[0].conv2.weight.data.cuda(), raft.cnet.layer1[0].conv2.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20).cuda()
    raft.cnet.layer1[1].conv1 = KTNLayer.KTNConv(raft.cnet.layer1[1].conv1.weight.data.cuda(), raft.cnet.layer1[1].conv1.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20).cuda()
    raft.cnet.layer1[1].conv2 = KTNLayer.KTNConv(raft.cnet.layer1[1].conv2.weight.data.cuda(), raft.cnet.layer1[1].conv2.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20).cuda()

    raft.cnet.layer2[0].conv1 = KTNLayer.KTNConv(raft.cnet.layer2[0].conv1.weight.data.cuda(), raft.cnet.layer2[0].conv1.bias.data.cuda(), sphereH = 160, imgW = 320,tied_weights = 20, output_shape = (80,160)).cuda()
    raft.cnet.layer2[0].conv2 = KTNLayer.KTNConv(raft.cnet.layer2[0].conv2.weight.data.cuda(), raft.cnet.layer2[0].conv2.bias.data.cuda(), sphereH = 80, imgW = 160,tied_weights = 20).cuda()
    raft.cnet.layer2[1].conv1 = KTNLayer.KTNConv(raft.cnet.layer2[1].conv1.weight.data.cuda(), raft.cnet.layer2[1].conv1.bias.data.cuda(), sphereH = 80, imgW = 160,tied_weights = 20).cuda()
    raft.cnet.layer2[1].conv2 = KTNLayer.KTNConv(raft.cnet.layer2[1].conv2.weight.data.cuda(), raft.cnet.layer2[1].conv2.bias.data.cuda(), sphereH = 80, imgW = 160,tied_weights = 20).cuda()

    raft.cnet.layer3[0].conv1 = KTNLayer.KTNConv(raft.cnet.layer3[0].conv1.weight.data.cuda(), raft.cnet.layer3[0].conv1.bias.data.cuda(), sphereH = 80, imgW = 160,tied_weights = 20, output_shape = (40,80)).cuda()
    raft.cnet.layer3[0].conv2 = KTNLayer.KTNConv(raft.cnet.layer3[0].conv2.weight.data.cuda(), raft.cnet.layer3[0].conv2.bias.data.cuda(), sphereH = 40, imgW = 80,tied_weights = 5).cuda()
    raft.cnet.layer3[1].conv1 = KTNLayer.KTNConv(raft.cnet.layer3[1].conv1.weight.data.cuda(), raft.cnet.layer3[1].conv1.bias.data.cuda(), sphereH = 40, imgW = 80,tied_weights = 5).cuda()
    raft.cnet.layer3[1].conv2 = KTNLayer.KTNConv(raft.cnet.layer3[1].conv2.weight.data.cuda(), raft.cnet.layer3[1].conv2.bias.data.cuda(), sphereH = 40, imgW = 80,tied_weights = 5).cuda()
    raft = raft.cuda()
    return raft