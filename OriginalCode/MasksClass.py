# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:34:22 2018

@author: Proprietario
"""

import cv2
import numpy as np

#making an ellipse

class MasksClass ():
    
    def __init__(self, image):
        self.image = image
        
    def createCakeSlicesMask (self, startAngle, endAngle):
        
        ellipse = np.zeros((self.image.shape[0],self.image.shape[1]), np.uint8)
        midrows = int (self.image.shape[0] / 2)
        midcols = int (self.image.shape[1] / 2)
        # create the slice at the given angle range
        cv2.ellipse(ellipse, (midcols, midrows), (midcols *2 , midrows *2 ), 180, startAngle, endAngle, 255, -1)
        
        return ellipse
        '''
        cv2.imshow("Ellipse Mask", ellipse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    def createMaskFromThreshold (self, Min_threshold = 1, Max_threshold = 255):
        
        # convert original into grayscale
        graythresold = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
         # threshold
        ret,thresh = cv2.threshold(graythresold, Min_threshold, Max_threshold, cv2.THRESH_BINARY)
        #return threshold image BINARY for masking
        return thresh