from utils import AverageMeter
from dataloader import AllLoader
from RAFT import load_raft
import torch.nn as nn
import torch
import torch.optim as optim
from simsiam import Siam360
import os
from tqdm import tqdm
from equi_utils import rotate_eq, rotate_PIL, flow_rotation, getRandomRotationConfig
from evaluate_raft import validate_flow360
import numpy as np
from pathlib import Path
import utils
import random
from ktnfyraft import getKTNisedRaft
from sys import exit
# MAX_FLOW = 300
# NUM_EPOCH = 100
# TRAIN_BATCH = 16
# VAL_BATCH = 16
# CLIP_GRAD = True
# DEVICE_IDS = [0,1,2,3,4,5,6,7]
# TRAIN = True
# USE_DENISTY_MASK = False
# MODEL_PATH = 'cache/_smallermotion_cache_final_version001.pt'
# MODEL_NAME = f"_smallermotion_cache_final_version001{'_WEIGHTED' if USE_DENISTY_MASK else ''}"
# MODEL_SAVE = f'cache/{MODEL_NAME}.pt'
# BENCHMARK = False
# LOAD = True
# FINETUNE = False
# SWITCH_ROTATION = False
# DOUBLE_ROTATION = False


def fetch_optimizer(model, epochs, steps_per_epoch, init_lr = 0.00002, wdecay = 0.00005, epsilon = 1e-8):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=wdecay, eps=epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = init_lr, epochs = epochs, steps_per_epoch = steps_per_epoch,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class EarlyStopping():
    """
    credit: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
    

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def sequence_loss(flow_preds, flow_gt, invalid = None, gamma=0.8, max_flow=300, return_epe_only = False):
    """ Loss function defined over sequence of flow predictions """
    density = torch.from_numpy(np.load("distortiondensity.npy")).unsqueeze(0).unsqueeze(0)
    density = density.cuda()
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    if not (invalid is None):
        valid = (invalid >= 0.5) & (mag < max_flow)
    else:
        valid = ((flow_gt.select(1,0).abs() < 1000) &  (flow_gt.select(1,1).abs()<1000)).float()
        valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i].clip(-40,40) - flow_gt.clip(-40,40)).abs()# * (1.0 + density)
        flow_loss += i_weight * (valid[:,None] * i_loss).mean()
    return flow_loss

def run(args):
    init_lr = 0.05 * args.train_batch / 256
    early_stopping = EarlyStopping(patience = args.patience, min_delta = args.min_delta)
    loader = AllLoader(root_path = Path(args.data_root),modes = ['train'], items=['frame1','frame2','fflow','bflow'], train_batch_size=args.train_batch, val_batch_size=args.val_batch)
    train_loader = loader.loadtrain()
    
    if args.ktn:    
        model = Siam360(getKTNisedRaft().train(True), finetune = args.finetune)
    else:
        model = Siam360(load_raft.load(path_root="RAFT", data_parallel=False, simsiam=True, load = args.load).train(True), finetune = args.finetune)
        
    
    def train(loader, criterion, init_lr, epoch, print_freq = 10):
        optimizer, scheduler = fetch_optimizer(model, epochs = args.num_epoch, steps_per_epoch = len(loader) + args.train_batch, init_lr = init_lr)
        scaler = GradScaler(enabled = args.grad_scaler)
        
        if not args.finetune:    
            losses_sim = AverageMeter("Loss", ':.4f')
        
        losses = AverageMeter("Loss", ':.4f')
        losses_flow = AverageMeter("Loss", ':.4f')
        
        dataloader = tqdm(loader)
        
        dataloader.set_description_str(f"[TRAINING]")
        
        for i, data in enumerate(dataloader):
            if utils.stop(write = False):
                break
            optimizer.zero_grad()

            
            frame1  = data['frame1']
            frame2  = data['frame2']
            
            
            pitch = data['pitch'].cuda()
            yaw = data['yaw'].cuda()
            roll = data['roll'].cuda()
            
            pitch_ = data['pitch_'].cuda()
            yaw_ = data['yaw_'].cuda()
            roll_ = data['roll_'].cuda()
            
            
            flowgt = data['fflow']
            
            # rotconfigs
            rot_ = [{'pitch':p, 'roll':r, 'yaw':y} for p,r,y in  zip(pitch_.tolist(), roll_.tolist(), yaw_.tolist())]
            rot = [{'pitch':p, 'roll':r, 'yaw':y} for p,r,y in  zip(pitch.tolist(), roll.tolist(), yaw.tolist())]
            
            frame1_ = rotate_eq(frame1, rots = rot_, mode = "bilinear")
            frame2_ = rotate_eq(frame2, rots = rot_, mode = "bilinear")
            
            if args.double_rotation:
                frame1 = rotate_eq(frame1, rots = rot, mode = "bilinear")
                frame2 = rotate_eq(frame2, rots = rot, mode = "bilinear")
                flowgt = rotate_eq(flowgt, mode = "bilinear", rots = rot, map_range=True, map_min_src=flowgt.min(), map_max_src = flowgt.max(), map_min_des=0, map_max_des=1)
                valid  = None
            else:
                valid = data['valid']
                valid = valid.cuda()
            
            frame1 = frame1.cuda()
            frame2 = frame2.cuda()
            flowgt = flowgt.cuda()
            frame1_ = frame1_.cuda()
            frame2_ = frame2_.cuda()
            
            #regularization
            if not args.finetune:    
                l1_regularization, l2_regularization = torch.tensor(0).float().cuda(), torch.tensor(0).float().cuda()
            
            if args.finetune:
                try:    
                    flow_predictions = model(frame1, frame2, frame1_, frame2_)
                except Exception as e:
                    print(e)
                    exit()
                    
                    
            else:
                try:
                    if args.switch_rotation:
                        prob = random.random()
                        if prob>0.5:
                            p1, p2, z1, z2, flow_predictions = model(x1 = frame1_, 
                                                                x2 = frame2_, 
                                                                x3 = frame1, 
                                                                x4 = frame2, 
                                                                rots = True, 
                                                                rots_ = False, 
                                                                pitch = pitch_, 
                                                                yaw = yaw_, 
                                                                roll = roll_, 
                                                                pitch_ = None, 
                                                                roll_ = None, 
                                                                yaw_ = None,)
                        else:
                            p1, p2, z1, z2, flow_predictions = model(x1 = frame1, 
                                                                x2 = frame2, 
                                                                x3 = frame1_, 
                                                                x4 = frame2_, 
                                                                rots = False, 
                                                                rots_ = True, 
                                                                pitch = None, 
                                                                yaw = None, 
                                                                roll = None, 
                                                                pitch_ = pitch_, 
                                                                roll_ = roll_, 
                                                                yaw_ = yaw_,)
                    else:    
                        p1, p2, z1, z2, flow_predictions = model(x1 = frame1, 
                                                                x2 = frame2, 
                                                                x3 = frame1_, 
                                                                x4 = frame2_, 
                                                                rots = False, 
                                                                rots_ = True, 
                                                                pitch = None, 
                                                                yaw = None, 
                                                                roll = None, 
                                                                pitch_ = pitch_, 
                                                                roll_ = roll_, 
                                                                yaw_ = yaw_,)
                    
                except Exception as e:
                    print(e)
                    exit()
            if not args.finetune:    
                for param in model.module.parameters():
                    l1_regularization += torch.norm(param, 1)**2
                    l2_regularization += torch.norm(param, 2)**2
            
            
            loss_flow = sequence_loss(flow_predictions, flowgt, invalid = valid, max_flow = args.max_flow)
            if not args.finetune:
                loss_sim = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                loss = loss_sim + loss_flow + 1e-10*l1_regularization + 1e-5*l2_regularization #*0.1 + loss_flow_ 
            else:
                loss = loss_flow
            if not args.finetune:    
                losses_sim.update(loss_sim.item(), frame1.size(0))
            
            losses_flow.update(loss_flow, frame1.size(0))
            losses.update(loss.item(), frame1.size(0))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            if i%print_freq == 0:
                if not args.finetune:    
                    dataloader.set_postfix_str(f"Loss: {losses.avg:.4f} | FlowLoss: {losses_flow.avg:.4f} | SimLoss: {losses_sim.avg:.4f} | epoch: {epoch}")
                else:
                    dataloader.set_postfix_str(f"Loss: {losses.avg:.4f} | epoch: {epoch}")
        return losses.avg
    
    if not args.benchmark:
        if args.load:
            if Path(args.model_path).exists():
                try:
                    model.load_state_dict(torch.load(args.model_path))
                    model.eval()
                    print(f"model succesfully loaded from {args.model_path}")
                except Exception as e:
                    print(e)
                    torch.cuda.empty_cache()
                    exit()
        
        criterion = nn.CosineSimilarity(dim=1).cuda()
        model = model.cuda()
        model = nn.DataParallel(model, device_ids = args.gpus)
        epe = 1000000
        
        for epoch in range(args.num_epoch):
            try:
                model.eval()
                epe_, data, filename = validate_flow360(model.module.encoder, root_path = Path(args.data_root), mode="test", iters = 12, rotation = False, filename = args.csv_save, save = not args.train, weighted=args.use_density_mask)
                if epe_ < epe:
                    epe = epe_
                    if args.train:
                        print(f"results saved at {filename}")
                        data.to_csv(filename, index = None)
                        
                        print(f"Model saved with current epe {epe_}")
                        torch.save(model.module.state_dict(), args.model_save)
            except Exception as e:
                torch.cuda.empty_cache()
                print(e)
                exit()
        
            if (not args.train) or args.benchmark:
                break
            
            try:   
                model.train(True)
                loss_ = train(train_loader, criterion, init_lr, epoch)
                early_stopping(loss_)
                if early_stopping.early_stop:
                    break
                
            except Exception as e:
                torch.cuda.empty_cache()
                print(e)
                exit()
            
            if utils.stop():
                break