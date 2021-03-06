# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:20:01 2018

@author: Proprietario
http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#sphx-glr-auto-examples-segmentation-plot-segmentations-py
"""

#from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import cv2

class segmScikit():
    
    def __init__(self, imgpath):
        self.img = cv2.imread(imgpath)
        
    
    def makeSegmentation(self):
        
        segments_fz = felzenszwalb(self.img, scale=1000, sigma=0.5, min_size=250)
        #segments_slic = slic(self.img, n_segments=4, compactness=10, sigma=1)
        segments_quick = quickshift(self.img, kernel_size=3, max_dist=100, ratio=0.5)
        gradient = sobel(rgb2gray(self.img))
        segments_watershed = watershed(gradient, markers=10, compactness=0.001)
        
        print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
        #print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
        print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))
        
        fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        
        ax[0, 0].imshow(mark_boundaries(self.img, segments_fz))
        ax[0, 0].set_title("Felzenszwalbs's method")
        #ax[0, 1].imshow(mark_boundaries(self.img, segments_slic))
        ax[0, 1].set_title('SLIC')
        ax[1, 0].imshow(mark_boundaries(self.img, segments_quick))
        ax[1, 0].set_title('Quickshift')
        ax[1, 1].imshow(mark_boundaries(self.img, segments_watershed))
        ax[1, 1].set_title('Compact watershed')
        
        for a in ax.ravel():
            a.set_axis_off()
        
        plt.tight_layout()
        plt.show()

path= "D:/aaa/a.jpg"       
a = segmScikit(path)
a.makeSegmentation()
