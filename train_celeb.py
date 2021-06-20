#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# - Starter using PyTorch

# # Directory settings

# ====================================================
# Directory settings
# ====================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1' # specify GPUs locally

OUTPUT_DIR = './submission'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
light = True 
dataset_path = './data/CelebAMask-HQ'

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Hair Segmentation')
    parser.add_argument('--data_type', type=str, default='korean') # korean, celeb
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--encoder_type', type=str, default='timm-efficientnet-b0')
    parser.add_argument('--decoder_type', type=str, default='Unet')
    parser.add_argument('--scheduler', type=str, default='GradualWarmupSchedulerV2')
    parser.add_argument('--encoder_lr',type=float,  default=3e-5)
    parser.add_argument('--min_lr',type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--apex', type=bool, default=False)
    args = parser.parse_args()
    return args

# # Data Loading

import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# # CFG

# In[3]:


# ====================================================
# CFG  
# ====================================================
class CFG:
    debug=False
    img_size=512
    max_len=275
    print_freq=1000
    num_workers=0
    encoder_type='se_resnext50_32x4d'
    decoder_type='Unet'
    size=512 # [512, 1024]
    freeze_epo = 0
    warmup_epo = 1
    cosine_epo = 19 #14 #19
    warmup_factor=10
    scheduler='GradualWarmupSchedulerV2' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'GradualWarmupSchedulerV2', 'get_linear_schedule_with_warmup']
    epochs=freeze_epo + warmup_epo + cosine_epo # not to exceed 9h #[1, 5, 10]
    factor=0.2 # ReduceLROnPlateau
    patience=4 # ReduceLROnPlateau
    eps=1e-6 # ReduceLROnPlateau
    T_max=4 # CosineAnnealingLR
    T_0=4 # CosineAnnealingWarmRestarts
    encoder_lr=3e-5 #[1e-4, 3e-5]
    min_lr=1e-6
    batch_size= 24 + 0 #[64, 256 + 128, 512, 1024, 512 + 256 + 128, 2048]
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=5
    dropout=0.5
    seed=42
    n_fold=1
    trn_fold=[0]
    #trn_fold=[0, 1, 2, 3, 4] # [0, 1, 2, 3, 4]
    train=True
    apex=False
    log_day='0618'
    version='v1-1'
    load_state=False
    light='light'
    data_type='celeb'


#if CFG.apex:
from torch.cuda.amp import autocast, GradScaler
if CFG.debug:
    CFG.epochs = 2
    train = train.sample(n=2, random_state=CFG.seed).reset_index(drop=True) 


# # Library

# ====================================================
# Library
# ====================================================
import sys
#sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

import os
import gc
import re
import math
import time
import random
import shutil
import pickle
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
# from transformers import get_linear_schedule_with_warmup

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import albumentations as A

import segmentation_models_pytorch as smp

import warnings 
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Utils
# ====================================================
# Utils
# ====================================================

def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)


# # CV split

import os 
data = pd.DataFrame() 
data['images'] = [dataset_path + '/images/' + c for c in sorted(os.listdir(dataset_path + '/images'))]
data['masks'] = [dataset_path + '/masks/' + c.split('.')[0] + '.png' for c in sorted(os.listdir(dataset_path + '/images'))]


from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
FOLDS = 5
kf = KFold(FOLDS)

data['fold'] = 0
for fold, (tr_idx, val_idx) in enumerate(kf.split(data)):
    data.loc[val_idx, 'fold'] = fold


# # Dataset

class TrainDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, idx):
        images = cv2.imread(self.data.loc[idx]['images'])    
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        masks = cv2.imread(self.data.loc[idx]['masks'])[:,:,0]
        masks = masks.astype(float)
        masks /= 255
        masks = np.expand_dims(masks, axis=2)

        if self.transform is not None:
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]

        return images, masks

    def __len__(self):
        return len(self.data)

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
                A.Resize(512, 512,always_apply=True),
                A.OneOf([
                    A.RandomContrast(),
                    A.RandomGamma(),
                    A.RandomBrightness(),
                    ], p=0.3),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                A.ShiftScaleRotate(p=0.2),
                A.GridDropout(p=0.1), 
                # A.Resize(512, 512,always_apply=True),
                A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(transpose_mask=True)
        ],p=1.)
    elif data == 'valid':
        return A.Compose([
            A.Resize(512, 512, always_apply=True),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(transpose_mask=True)
        ],p=1.)


# # MODEL
class Encoder(nn.Module):
    def __init__(self, encoder_name='timm-efficientnet-b3', decoder_name='Unet' , pretrained=False):
        super().__init__()
        if encoder_name in ['se_resnext50_32x4d', 'se_resnext101_32x4d']: 
            encoder_weights = 'imagenet' 
        else: 
            encoder_weights = 'noisy-student' 
        
        if decoder_name == 'Unet': 
            self.encoder = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'UnetPlusPlus':
            self.encoder = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'MAnet': 
            self.encoder = smp.MAnet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'Linknet': 
            self.encoder = smp.Linknet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'FPN':
            self.encoder = smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'PSPNet': 
            self.encoder = smp.PSPNet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'PAN': 
            self.encoder = smp.PAN(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'DeepLabV3': 
            self.encoder = smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        elif decoder_name == 'DeepLabV3Plus': 
            self.encoder = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1) # [imagenet, noisy-student]
        else:
            raise ValueError(f"decoder_type : {decoder_name} is not exist")
           
        
    #@autocast()
    def forward(self, x):
        x = self.encoder(x)
        return x


# # Scheduler

from warmup_scheduler import GradualWarmupScheduler
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


#https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
    
class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).mean()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.mean() + targets.mean() + smooth)  
        Dice_BCE = 0.9*BCE + 0.1*dice_loss
        
        return Dice_BCE.mean()


def get_dice_coeff_ori(pred, targs, eps = 1e-9):
    '''
    Calculates the dice coeff of a single or batch of predicted mask and true masks.
    
    Args:
        pred : Batch of Predicted masks (b, w, h) or single predicted mask (w, h)
        targs : Batch of true masks (b, w, h) or single true mask (w, h)
  
    Returns: Dice coeff over a batch or over a single pair.
    '''
    # sigmoid 달아야하는 지 고민 
    p = (pred.view(-1) > 0).float()
    t = (targs.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice


def get_dice_coeff(pred, targs, eps = 1e-9):
    '''
    Calculates the dice coeff of a single or batch of predicted mask and true masks.
    
    Args:
        pred : Batch of Predicted masks (b, w, h) or single predicted mask (w, h)
        targs : Batch of true masks (b, w, h) or single true mask (w, h)
  
    Returns: Dice coeff over a batch or over a single pair.
    '''
    
    
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + eps)

    
def reduce(values):
    '''    
    Returns the average of the values.
    Args:
        values : list of any value which is calulated on each core 
    '''
    return sum(values) / len(values)


def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))


# # Train 
# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, encoder, criterion, 
             optimizer, epoch,
             scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dice_coeffs = AverageMeter()
    # switch to train mode
    encoder.train()
    
    scaler = torch.cuda.amp.GradScaler()
    
    start = end = time.time()
    global_step = 0
    for step, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        targets = targets.float().to(device)
        batch_size = images.size(0)
        
        # =========================
        # zero_grad()
        # =========================
        optimizer.zero_grad()
        if CFG.apex:
            with autocast():
                y_preds = encoder(images)
                loss = criterion(y_preds, targets)
                scaler.scale(loss).backward()
        else:
            y_preds = encoder(images)
            loss = criterion(y_preds, targets)
            loss.backward()
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        #loss.backward()
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            if CFG.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            global_step += 1
            
        # record dice_coeff
        dice_coeff = get_dice_coeff(y_preds, 
                                    targets)
        dice_coeffs.update(dice_coeff, batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Dice_coeff: {dice_coeff.val:.4f}({dice_coeff.avg:.4f}) '
                  'Encoder Grad: {encoder_grad_norm:.4f}  '
                  'Encoder LR: {encoder_lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, dice_coeff=dice_coeffs,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   encoder_grad_norm=encoder_grad_norm,
                   encoder_lr=scheduler.get_lr()[0],
                   ))
    return losses.avg, dice_coeffs.avg


def valid_fn(valid_loader, encoder, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dice_coeffs = AverageMeter()
    
    # switch to evaluation mode
    encoder.eval()
    #trues = []
    #preds = []
    start = end = time.time()
    for step, (images, targets) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        targets = targets.float().to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            y_preds = encoder(images)
        
        loss = criterion(y_preds, targets)
        losses.update(loss.item(), batch_size)
        
        # record dice_coeff
        dice_coeff = get_dice_coeff(y_preds, 
                                    targets)
        dice_coeffs.update(dice_coeff, batch_size)
        
        #trues.append(labels.to('cpu').numpy())
        #preds.append(y_preds.sigmoid().to('cpu').numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Dice_coeff: {dice_coeff.val:.4f}({dice_coeff.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, dice_coeff=dice_coeffs,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    #preds = np.concatenate(preds)

    return losses.avg, dice_coeffs.avg


# ====================================================
# Train loop
# ====================================================
def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    #valid_labels = valid_folds[CFG.target_labels].values

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, 
                              batch_size=CFG.batch_size, 
                              shuffle=True, 
                              num_workers=CFG.num_workers, 
                              pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=CFG.batch_size, 
                              shuffle=False, 
                              num_workers=CFG.num_workers,
                              pin_memory=True, 
                              drop_last=False)
    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='GradualWarmupSchedulerV2':
            scheduler_cosine=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, CFG.cosine_epo)
            scheduler_warmup=GradualWarmupSchedulerV2(optimizer, multiplier=CFG.warmup_factor, total_epoch=CFG.warmup_epo, after_scheduler=scheduler_cosine)
            scheduler=scheduler_warmup        
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    encoder = Encoder(CFG.encoder_type, CFG.decoder_type, pretrained=True)
    encoder.to(device)
    
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        #print('DataParallel')
        encoder = nn.DataParallel(encoder)

    optimizer = Adam(encoder.parameters(), lr=CFG.encoder_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)


    # ====================================================
    # loop
    # ====================================================
    # criterion = nn.BCEWithLogitsLoss()
    criterion = DiceBCELoss() #['DiceBCELoss()', 'DiceLoss()']

    best_score = 0
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss, avg_tr_dice_coeff = train_fn(train_loader, encoder, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, avg_val_dice_coeff = valid_fn(valid_loader, encoder, criterion, device)

        # scoring
        #score = get_score(valid_labels, text_preds)
        score = avg_val_dice_coeff
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(score)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, GradualWarmupSchedulerV2):
            scheduler.step(epoch)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {avg_val_dice_coeff:.4f}')
        
        
        model_to_save = encoder.module if hasattr(encoder, 'module') else encoder
        
        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'encoder': model_to_save.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict(), 
                        #'text_preds': text_preds,
                       },
                        OUTPUT_DIR+f'/{CFG.encoder_type}_{CFG.decoder_type}_fold{fold}_{CFG.light}_{CFG.data_type}.pth')
            best_oof = avg_val_dice_coeff

# ====================================================
# main
# ====================================================
def main(rank=0, world_size=0):

    """
    Prepare: 1.train  2.folds
    """
    if CFG.train:
        # train
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                train_loop(data, fold)

if __name__ == '__main__':
    args = get_args()
    CFG.data_type = 'celeb'
    CFG.img_size = args.img_size
    CFG.num_workers = args.num_workers
    CFG.encoder_type = args.encoder_type
    CFG.decoder_type = args.decoder_type
    CFG.scheduler = args.scheduler
    CFG.encoder_lr = args.encoder_lr
    CFG.min_lr = args.min_lr
    CFG.batch_size = args.batch_size
    CFG.weight_decay = args.weight_decay
    CFG.apex = args.apex
    main()
