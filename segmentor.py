# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import json
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology,feature, color, data, filters

from watershed_mark import WaterShed_Mark
from watershed_distance_transform import WaterShed_Distance_Transform
from watershed_gradient import WaterShed_Gradient
        
def main():
    image_path = "SEM\高晶面图"
    dirs = os.listdir(image_path)
    # print(dirs)
    
    filename = 'tags.json'
    with open(filename, 'r') as file_obj:
        tags = json.load(file_obj)
    
    for name in dirs:
        original = mpimg.imread(os.path.join(image_path, name).replace('\\', '/'))
        # img = original.copy()
        print(name)
        # gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        print(original)
        
        # 基于标记的分水岭分割
        watershed_mark = WaterShed_Mark(original)
        watershed_mark.segment()
        watershed_mark.plot()

        # # 基于距离变换的分水岭图像分割
        # watershed_dist = WaterShed_Distance_Transform(original)
        # watershed_dist.segment()
        # watershed_dist.plot()

        # # 基于梯度的分水岭分割
        # watershed_gradient = WaterShed_Gradient(original)
        # watershed_gradient.segment()
        # watershed_gradient.plot()

if __name__ == "__main__":
    main()
