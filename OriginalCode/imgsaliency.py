# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:40:16 2018

@author: Proprietario
https://github.com/yhenon/pyimgsaliency
"""

import cv2
import pyimgsaliency as psal

class ImageSaliency():
    
    def __init__(self, imgpath):
        self.img = cv2.imread(imgpath)
        self.filename = imgpath
        
    
    def runSaliency(self):

        # get the saliency maps using the 3 implemented methods
        #rbd = psal.get_saliency_rbd(self.filename).astype('uint8')
        
        ft = psal.get_saliency_ft(self.filename).astype('uint8')
        
        mbd = psal.get_saliency_mbd(self.filename).astype('uint8')
        
        # often, it is desirable to have a binary saliency map
        binary_sal = psal.binarise_saliency_map(mbd,method='adaptive')
        
        img = cv2.imread(self.filename)
        
        cv2.imshow('img',img)
        #cv2.imshow('rbd',rbd)
        cv2.imshow('ft',ft)
        cv2.imshow('mbd',mbd)
        
        #openCV cannot display numpy type 0, so convert to uint8 and scale
        cv2.imshow('binary',255 * binary_sal.astype('uint8'))
        
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()