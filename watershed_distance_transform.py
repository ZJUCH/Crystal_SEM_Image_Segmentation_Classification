# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import json
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology,feature, color, data, filters

# 基于距离变换的分水岭图像分割
class WaterShed_Distance_Transform():
    def __init__(self, input):
        self.original = input

    def segment(self):
        self.image = self.original.copy()
        self.distance = ndi.distance_transform_edt(self.image) #距离变换
        self.local_maxi = feature.peak_local_max(self.distance, indices=False, footprint=np.ones((3, 3, 3)), labels = self.image)   #寻找峰值
        self.markers = ndi.label(self.local_maxi)[0]           #初始标记点
        self.labels = morphology.watershed(-self.distance, self.markers, mask = self.image) #基于距离变换的分水岭算法

    def plot(self):
        plt.ion()
        plt.subplot(221), plt.imshow(self.image, cmap = 'gray'),
        plt.title('Original'), plt.axis('off')
        plt.subplot(222), plt.imshow(-self.distance, cmap = 'jet'),
        plt.title('Distance'), plt.axis('off')
        plt.subplot(223), plt.imshow(self.markers, cmap = 'gray'),
        plt.title('Markers'), plt.axis('off')
        plt.subplot(224), plt.imshow(self.labels, cmap = 'gray'),
        plt.title('Result'), plt.axis('off')
        plt.tight_layout()
        input("Next")