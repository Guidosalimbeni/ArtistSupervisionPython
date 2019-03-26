# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:57:16 2018

@author: Proprietario
"""
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from FeatureDetection import FeatureDetection
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

class ShapeDetection(FeatureDetection):
    
    def __init__(self, imagepath, imagepathTemplate, imagepathTemplateCusp):
        
        FeatureDetection.__init__(self, imagepath, imagepathTemplate, imagepathTemplateCusp)
        
    
    def grabcut(self):
        
        img = self.image.copy()
        mask = np.zeros(img.shape[:2],np.uint8)
        
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        rect = (2,2,self.image.shape[1]-20,self.image.shape[0]-20)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        
        cv2.imshow('rabcut', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        
        #plt.imshow(img),plt.colorbar(),plt.show()
    
    def hog_detection_segm(self):
        
        im = self.image.copy()

        gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 

        image = gr
        
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True)
        
        # Rescale histogram for better display
        #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
        
# =============================================================================
#         print (hog_image.shape)
#         cv2.imshow('ori', self.image)
#         cv2.imshow('hog', hog_image)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
        
        ax[0].axis('off')
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_adjustable('box-forced')
        
        
        ax[1].axis('off')
        ax[1].imshow(hog_image, cmap=plt.cm.gray)
        ax[1].set_title('Histogram of Oriented Gradients')
        ax[1].set_adjustable('box-forced')
        
        plt.show()
        
        
    def drawPolyScores (self, contours = None):
        
        if contours :
            
            for cnt in contours:
                
                cv2.polylines(self.image, cnt, 1, (255,0,255))
                
        test = np.array([[10,10], [25,25], [15,45], [52,120]])
        cv2.polylines(self.image, np.int32([test]), 1, (255,0,255))
            
        cv2.imshow('polylines', self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
            
        
        
    def shapePolygonsNum(self):
        
        # TODO elaborate on edges or thresh and type of apporx and cnts external or not
        approxCnts = self.cntApproxDetection()
        
        print('number of edges in approx cnts' , len(approxCnts))
        
        
        #TODO to determine if they are triangle or not and within the triangle shape
    
    
    # TODO if the center of the cnt is in the compositional boundaries or
    # TODO how far they are https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
    
    def superPixelSegmentation(self, num_segments = 10):
        # https://www.pyimagesearch.com/2014/12/29/accessing-individual-superpixel-segmentations-python/
        
        #load the image and convert it to a floating point data type
        copy = self.image.copy()
        image = img_as_float(copy)
        
        black = np.zeros(self.image.shape[:2], dtype = "uint8")
        
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        segments = slic(image, n_segments = num_segments, sigma = 5)
        
        segImg = mark_boundaries(black, segments)
        	# show the output of SLIC

        cv2.imshow('segm', segImg)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        return segments
    
    def accessingSuperpixel(self):
        
        supImg = self.image.copy()
        # TODO apply color differents color and plot the resulting figure
        # to see how the perceived images are seen once segmented in 4, 5 etc. or just 2
        segments = self.superPixelSegmentation()
        
        
        for (i, segVal) in enumerate(np.unique(segments)):
            # construct a mask for the segment
            #print ( "[x] inspecting segment %d" % (i))
            mask = np.zeros(self.image.shape[:2], dtype = "uint8")
            mask[segments == segVal] = 255
            
            
            # show the masked region
            '''
            cv2.imshow("Mask", mask)
            cv2.imshow("Applied", cv2.bitwise_and(supImg, supImg, mask = mask))
            cv2.waitKey()
            cv2.destroyAllWindows()
            '''
            
    def thresholdTest(self):
        
        copyThresh = self.image.copy()
        gray = cv2.cvtColor(copyThresh, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresh', thresh)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
            

    