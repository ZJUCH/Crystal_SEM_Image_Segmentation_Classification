# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import os
import io
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CrystalSEMImageDataset(Dataset):
    def __init__(self, tags_dict, root_dir, multiple_crystals = None, transform = None):
        self.tags = tags_dict
        self.root_dir = root_dir
        self.multiple_crystals = multiple_crystals
        self.transform = transform
    
    def __len__(self):
        return len(self.tags)
    
    def __getitem__(self, idx):
        names = os.listdir(self.root_dir)
        img_name = os.path.join(self.root_dir, names[idx])
        print(img_name.replace('\\', '/'))
        original = mpimg.imread(img_name.replace('\\', '/'))
        # print(original)
        # plt.figure("original")
        # plt.imshow(original) # 显示图片
        # plt.axis('off') # 不显示坐标轴
        # plt.show()
        
        if self.multiple_crystals:
            ptLeftTop = (self.multiple_crystals['left'], self.multiple_crystals['top'])
            ptRightBottom = (self.multiple_crystals['right'], self.multiple_crystals['bottom'])
            point_color = (0, 255, 0) # BGR
            thickness = 1 
            lineType = 4
            image = cv2.rectangle(original, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
            sample = {'original': original, 'segmented': image, 'borders': self.multiple_crystals,'tag': self.tags[names[idx]]}
        else:
            sample = {'image': original, 'tag': self.tags[names[idx]]}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    