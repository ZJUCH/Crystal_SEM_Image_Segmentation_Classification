# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import os
import io
import pretrainedmodels

from dataset import CrystalSEMImageDataset

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    my_densenet121 = torchvision.models.densenet121(pretrained = True)
    my_densenet161 = torchvision.models.densenet161(pretrained = True)
    my_densenet169 = torchvision.models.densenet169(pretrained = True)
    my_densenet201 = torchvision.models.densenet201(pretrained = True)

    my_inception = torchvision.models.inception_v3(pretrained = True)
    
    my_resnet18 = torchvision.models.resnet18(pretrained = True)
    my_resnet34 = torchvision.models.resnet34(pretrained = True)
    my_resnet50 = torchvision.models.resnet50(pretrained = True)
    my_resnet101 = torchvision.models.resnet101(pretrained = True)
    my_resnet152 = torchvision.models.resnet152(pretrained = True)

    my_vgg11 = torchvision.models.vgg11(pretrained = True)
    my_vgg13 = torchvision.models.vgg13(pretrained = True)
    my_vgg16 = torchvision.models.vgg16(pretrained = True)
    my_vgg19 = torchvision.models.vgg19(pretrained = True)
    

if __name__ == "__main__":
    main()


