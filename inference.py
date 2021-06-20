# ====================================================
# Directory settings
# ====================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1' # specify GPUs locally

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Hair Segmentation')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--num', type=int, default=1)
    args = parser.parse_args()
    return args

args = get_args()
# # Data Loading

import os
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


# # Library
# ====================================================
# Library
# ====================================================
import sys
import albumentations as A
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

def get_transforms(*, data):
    if data == 'valid':
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


# ====================================================
# model & optimizer
# ====================================================
fold = 0
encoder = Encoder(encoder_name='timm-efficientnet-b0', decoder_name='UnetPlusPlus', pretrained=False)
model_path = f'./submission/timm-efficientnet-b0_UnetPlusPlus_fold0_light.pth'
checkpoint = torch.load(model_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder'])


def get_transforms(*, data):
    if data == 'valid':
        return A.Compose([
            A.Resize(512, 512, always_apply=True),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(transpose_mask=True)
        ],p=1.)



img_path = f'./demo/{args.num}.jpg'
def test():  
    or_images = cv2.imread(img_path)
    or_images = cv2.cvtColor(or_images, cv2.COLOR_BGR2RGB)
    images = or_images.astype(np.float32)
    images = get_transforms(data='valid')(image=images)['image'].reshape(1, 3, 512, 512)
    encoder.eval()
    masks = encoder(images)
    masks[masks > 0] = 1
    masks[masks <= 0] = 0
    masks = masks[0][0].detach().numpy().astype(np.int32)
    
    fig = plt.figure(figsize=(12, 8))
    rows, cols = 1, 2
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(or_images)
    ax1.set_title('Image')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(masks, interpolation='nearest')
    ax2.set_title('Mask')
    ax2.axis("off")
    plt.show(block=True)
    plt.savefig("./result/output.jpg")
test()
