# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import os
import io

from torch import nn

NUM_CLASSES = 2

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1_i = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2_i = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3)
        
        self.conv1_o = nn.Conv2d(in_channels=NUM_CLASSES, out_channels=8, kernel_size=3)
        self.conv2_o = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3)
        
        self.conv1_c = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3)
        self.conv2_c = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3)
        
        self.dense1 = nn.Linear(in_features=28*28, out_features=2)
        
        self.maxpool = nn.MaxPool2d(3)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, img, seg):
        
        activation = F.relu
        
        x1 = img
        x2 = seg
        
        x1 = self.conv1_i(x1)
        x1 = activation(x1)
        x1 = self.maxpool(x1)
        
        x1 = self.conv2_i(x1)
        x1 = activation(x1)
        x1 = self.maxpool(x1)
        
        x2 = self.conv1_o(x2)
        x2 = activation(x2)
        x2 = self.maxpool(x2)
        
        x2 = self.conv2_o(x2)
        x2 = activation(x2)
        x2 = self.maxpool(x2)
        
        concat = torch.cat([x1, x2], dim=1)
        
        c = self.conv1_c(concat)
        c = activation(c)
        
        c = self.conv2_c(c)
        c = activation(c)

        c = c.squeeze()
        c = c.view(c.shape[0], -1)
        
        c = self.dense1(c)
        c = activation(c)
        c = self.softmax(c)

        return c
