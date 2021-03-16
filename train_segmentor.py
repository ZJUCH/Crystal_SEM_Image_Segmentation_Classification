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
import cv2

from generator import Generator
from discriminator import Discriminator
from dataset import CrystalSEMImageDataset

BATCHS_SIZE = 4

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    image_path = "SEM\高晶面图"
    #filename = os.path.join(image_path, 'tags.json')
    filename = 'tags.json'
    with open(filename, 'r') as file_obj:
        tags = json.load(file_obj)
    # print(tags)
    # #filename = os.path.join(image_path, 'borders.json')
    # filename = 'borders.json'
    # with open(filename, 'r') as file_obj:
    #     borders = json.load(file_obj)
    # print(borders)
    
    crystal_dataset = CrystalSEMImageDataset(tags, image_path)#, borders)
    
    for i in range(len(crystal_dataset)):
        sample = crystal_dataset[i]
        print(i, sample['image'].shape, sample['tag'])
    print("----------------------------")
    return
    dataloader = DataLoader(crystal_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    loss_function = nn.BCELoss()
    loss_Gs = []
    loss_Ds = []
    
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0001)
    optimizerG = optim.Adam(generator.parameters(), lr=0.0001)
    
    for i, data in enumerate(dataloader):
        x = data['original']
        real = data['segmented']
    
        discriminator.zero_grad()
        
        # all real batch
        label = torch.full((BATCHS_SIZE, 2), real_label).to(device)
        output = discriminator(x, real)
        
        loss_real = loss_function(output, label)
        loss_real.backward()
        D_x = output.mean().item()
        
        # all fake batch
        label.fill_(fake_label)
        fake = generator(x)
        output = discriminator(x, fake.detach())
        
        loss_fake = loss_function(output, label)
        loss_fake.backward()
        D_Gx1 = output.mean().item()
        
        loss_D = loss_real + loss_fake
        optimizerD.step()
        
        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(x, fake)
        
        loss_G = loss_function(output, label)
        loss_G.backward()
        D_Gx2 = output.mean().item()
        optimizerG.step()
    
        loss_Gs.append(loss_G.item())
        loss_Ds.append(loss_D.item())
        
        # print('G: {} D_real: {} D_fake: {}'.format(D_Gx2, D_x, D_Gx1))
        print('D: {}, G: {}'.format(loss_D.item(), loss_G.item()))
        
        if i == 1000:
            break

if __name__ == "__main__":
    main()