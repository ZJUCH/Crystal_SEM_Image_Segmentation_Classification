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

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        self.dconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=NUM_CLASSES, kernel_size=3)
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        
        activation = F.relu
        
        x = self.conv1(x)
        x = activation(x)
        
        x = self.conv2(x)
        x = activation(x)
        
        x = self.dconv1(x)
        x = activation(x)
        
        x = self.dconv2(x)
        x = activation(x)
        x = self.softmax(x)
        
        return x
