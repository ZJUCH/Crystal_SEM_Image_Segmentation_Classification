# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import json
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology,feature, color, data, filters

# 基于标记的分水岭分割
class WaterShed_Mark():
    def __init__(self, input):
        self.original = input

    def segment(self):
        self.img = self.original.copy()

        self.ret, self.thresh = cv2.threshold(self.img, 70, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        self.opening = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

        self.opening = cv2.cvtColor(self.opening, cv2.COLOR_RGB2GRAY)

        self.sure_bg = cv2.dilate(self.opening, kernel, iterations = 3)
        
        self.dist_transform = cv2.distanceTransform(self.opening, 1, 5)
        self.ret, self.sure_fg = cv2.threshold(self.dist_transform, 0.5*self.dist_transform.max(), 255, 0)

        self.sure_fg = np.uint8(self.sure_fg)
        self.unknown = cv2.subtract(self.sure_bg, self.sure_fg)
       
        self.ret, self.markers1 = cv2.connectedComponents(self.sure_fg)

        self.markers = self.markers1 + 1

        self.markers[self.unknown == 255] = 0

        self.markers3 = cv2.watershed(self.img, self.markers)
        self.img[self.markers3 == -1] = [0, 0, 255]

    def plot(self):
        plt.ion()
        # plt.subplot(241), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)),
        plt.subplot(241), plt.imshow(self.original),
        plt.title('Original'), plt.axis('off')
        plt.subplot(242), plt.imshow(self.thresh, cmap='gray'),
        plt.title('Threshold'), plt.axis('off')
        plt.subplot(243), plt.imshow(self.sure_bg, cmap='gray'),
        plt.title('Dilate'), plt.axis('off')
        plt.subplot(244), plt.imshow(self.dist_transform, cmap='gray'),
        plt.title('Dist Transform'), plt.axis('off')
        plt.subplot(245), plt.imshow(self.sure_fg, cmap='gray'),
        plt.title('Threshold'), plt.axis('off')
        plt.subplot(246), plt.imshow(self.unknown, cmap='gray'),
        plt.title('Unknow'), plt.axis('off')
        plt.subplot(247), plt.imshow(np.abs(self.markers), cmap='jet'),
        plt.title('Markers'), plt.axis('off')
        plt.subplot(248), plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)),
        plt.title('Result'), plt.axis('off')
        plt.tight_layout()
        plt.pause(0.01)
        input("Next")