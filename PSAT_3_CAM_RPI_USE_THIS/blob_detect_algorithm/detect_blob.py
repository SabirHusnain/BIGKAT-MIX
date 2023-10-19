# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:14:14 2016

@author: ps09og
"""

import cv2
import numpy as np
import matplotlib.pylab as plt

def weighted_greysums(img):
    """Find the center of gravity for an image"""
    
    x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    weightedx = x * img
    weightedy = y * img
    
    x_center = np.sum(weightedx) / np.sum(img)
    y_center = np.sum(weightedy) / np.sum(img)
    
    return x_center, y_center
    
    
    
img = cv2.imread("blob.png")
#
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#t_val = 150 #Threshold value 
#t, thresh = cv2.threshold(gray, t_val, 255, cv2.THRESH_BINARY)
#   
#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None)

print(weighted_greysums(gray))
plt.imshow(gray, cmap = 'gray')
circle1 = plt.Circle(weighted_greysums(gray), 1 , color='b', fill = True, linewidth = 2)
plt.gca().add_artist(circle1)  


#[x, y] = meshgrid(1:size(img, 2), 1:size(img, 1));
#weightedx = x .* img;
#weightedy = y .* img;
#xcentre = sum(weightedx(:)) / sum(img(:));
#ycentre = sum(weightedy(:)) / sum(img(:));

