import os
import numpy as np
from PIL import Image
from sklearn import preprocessing
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from augmend import (
    AdditiveNoise,
    Augmend,
    Elastic,
    FlipRot90,
    GaussianBlur,
    Identity,
)
from augmend.transforms import BaseTransform
from augmend.utils import _validate_rng
from conic import *

class Cellclassification(Dataset):
    def __init__(self,imgs,labels,split,target_patch_size=-1):
        self.X_img = imgs
        self.Y = labels
        self.split=split
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size,) * 2
        else:
            self.target_patch_size = None
        aug = Augmend()
        aug.add([HEStaining(amount_matrix=0.05, amount_stains=0.1)], probability=0.3)
        aug.add([FlipRot90(axis=(0, 1))], probability=0.5)
        aug.add([Elastic(grid=2, amount=3, order=1, axis=(0, 1), use_gpu=False)], probability=0.5)
        aug.add([GaussianBlur(amount=(0, 2), axis=(0, 1), use_gpu=False)], probability=0.1)
        aug.add([AdditiveNoise(0.01)], probability=0.8)
        aug.add([HueBrightnessSaturation(hue=0, brightness=0.1, saturation=(1, 1))], probability=0.9)
        self.aug=aug
        self.transforms1 = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transforms2 = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    def __getitem__(self, index):
        single_y = torch.tensor(self.Y[index]).type(torch.LongTensor)
        single_x = self.X_img[index,:,:,:]
        single_x=Image.fromarray(single_x)
        if self.target_patch_size is not None:
            single_x = single_x.resize(self.target_patch_size)
        if self.split=='train':
            single_x=self.aug(single_x)
            single_x = self.transforms1(single_x)
        else:
            single_x = self.transforms2(single_x)
        return single_x,single_y

    def __len__(self):
        return len(self.X_img)
