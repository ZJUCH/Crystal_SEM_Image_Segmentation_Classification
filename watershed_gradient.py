# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import json
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology,feature, color, data, filters

# 基于梯度的分水岭分割
class WaterShed_Gradient():
    def __init__(self, input):
        self.original = input
    
    def segment(self):
        self.image = cv2.cvtColor(self.original.copy(), cv2.COLOR_RGB2GRAY)
        self.denoised = filters.rank.median(self.image, morphology.disk(2)) #过滤噪声
        self.markers = filters.rank.gradient(self.denoised, morphology.disk(5)) < 10
        self.markers = ndi.label(self.markers)[0]
        self.gradient = filters.rank.gradient(self.denoised, morphology.disk(2)) #计算梯度
        self.labels = morphology.watershed(self.gradient, self.markers, mask = self.image) #基于梯度的分水岭算法

    def plot(self):
        plt.ion()
        plt.subplot(221), plt.imshow(self.image, cmap = 'gray'),
        plt.title('Original'), plt.axis('off')
        plt.subplot(222), plt.imshow(self.gradient, cmap = 'gray'),
        plt.title('Gradient'), plt.axis('off')
        plt.subplot(223), plt.imshow(self.markers, cmap = 'gray'),
        plt.title('Markers'), plt.axis('off')
        plt.subplot(224), plt.imshow(self.labels, cmap = 'gray'),
        plt.title('Result'), plt.axis('off')
        plt.tight_layout()
        input("Next")