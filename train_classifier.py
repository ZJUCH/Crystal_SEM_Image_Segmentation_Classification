# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import os
import io

from dataset import CrystalSEMImageDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#model = torchvision.models.

