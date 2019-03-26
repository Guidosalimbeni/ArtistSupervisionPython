# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:13:19 2018

@author: Proprietario
https://github.com/mayoyamasaki/saliency-map
"""

#import sys
from src.saliency_map import SaliencyMap
#from src.utils import OpencvIo
import cv2
# =============================================================================
# 
# if __name__ == "__main__":
#     oi = OpencvIo()
#     src = oi.imread(sys.argv[1])
#     sm = SaliencyMap(src)
#     oi.imshow_array([sm.map])
# =============================================================================

class Image_Saliency_2():
    
    def __init__(self, imgpath):
        self.img = cv2.imread(imgpath)
        self.filename = imgpath
        
    
    def runSaliency2(self):

        sm = SaliencyMap(self.filename)
        
        cv2.imshow('sal22', sm)
        cv2.imshow('orig', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()