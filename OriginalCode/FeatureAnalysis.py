# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:13:08 2018

Class for implementing the scores for the composition UI and also the display image
with all the scores

@author: Guido Salimbeni
"""
import cv2
import numpy as np
from skimage import measure
from skimage.measure import compare_ssim as ssim


class FeatureAnalysis():
    
    def __init__(self, imagepath):
        self.image = cv2.imread(imagepath)
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        
    
    def displayScoresImage (self):
        
        # collect all the scores that are not part of the display label images
        
        ssimAsymmetry = self.calculateAsimmetry()
        fractalScoreFromTarget = self.fractalDimMinkowskiBoxCount()
        ratioForeVsBackground = self.areaForegroundVsbackground()
        
        # return the display image for the UI
        rows, cols, depth = self.image.shape
        blackboard = np.zeros(self.image.shape, dtype="uint8")
        # make solid color for the background
        blackboard[:] = (218,218,218)
        
        
        scoringListtoDisplay= []
        scoringListtoDisplay.append(ssimAsymmetry)
        scoringListtoDisplay.append(fractalScoreFromTarget)
        scoringListtoDisplay.append(ratioForeVsBackground)
        
        scalar = .9
        padding = 10
        switch = True
        colors = [(120,60,120), (60,120,120)]
        for score in scoringListtoDisplay:
            if switch:
                blackboard[padding:padding+10, 10:int(score*scalar*rows)] = colors[0]
                padding += padding
            else:
                blackboard[padding:padding+10, 10:int(score*scalar*rows)] = colors[1]
                padding += padding
            switch = not switch
        #scores = np.array([[10, int(- 10 + rows -(percentageWarm * rows * .6))], [20, int(- 10 + rows - (percentageCold * rows * .6))]])
        #cv2.polylines(blackboard, np.int32([scores]), 1, (255,0,255))
        cv2.putText(blackboard, "Asym , fract, ratioFore", (100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        # send to UI json writing (as a return)
        
        return blackboard, ssimAsymmetry, fractalScoreFromTarget, ratioForeVsBackground
    
    def areaForegroundVsbackground (self):
        
        imageGray = self.gray.copy()
        meanGray = np.mean(imageGray)
        # blur to increase extraction
        imageBlurred = cv2.GaussianBlur(imageGray, (3,3), 0)

        ret, threshImageBinary = cv2.threshold(imageBlurred, meanGray, 255, cv2.THRESH_BINARY)

        # count the number of pixel greater than 1 so in the area foreground
        totalPixelInForeground = (threshImageBinary > 1).sum()
        
        ratioForeVsBackground = totalPixelInForeground / threshImageBinary.size

        return ratioForeVsBackground
    
    def calculateAsimmetry(self, image = None):
        # calculate the the hist correlation of flipped image
        # if image is not provided uses self.image
        if image is None:
            image = self.image
        
        imageCopy = image.copy()
        imageFlipLfRt = cv2.flip(imageCopy, 1)
        imageFlipLf = imageFlipLfRt.copy()
        #imageFlipLfHalfRows = int(imageFlipLf.shape[0] /2)
        imageFlipLfHalfCols = int(imageFlipLf.shape [1] / 2)
        imageHalfCols = int(imageCopy.shape [1] / 2)
        
        imageFlipLf [:,0 : imageFlipLfHalfCols ] = 0
        imageCopy [:, 0 : imageHalfCols] = 0
        
        imageFlipLf = imageFlipLf [0 : image.shape[0], imageFlipLfHalfCols : image.shape[1]]
        imageCopy = imageCopy [0 : image.shape[0], imageHalfCols : image.shape[1]]

        similaritySSIM = self.compare_images (imageFlipLf, imageCopy)

        ssimAsymmetry = 1 - similaritySSIM
        
        return ssimAsymmetry
    
    def compare_images(self, imageA, imageB):
    
        s = ssim(imageA,imageB, win_size=None, gradient=False, data_range=None, 
                 multichannel = True, gaussian_weights=False, full=False, dynamic_range = None)
        
        return s
        
        
    def calculateWarmOrCold(self):

        imageHSV = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        #calculate the percentage, assuming warm from 150 to 0 and 0 to 30
        # and cold from 31 to 149
        warm = 0
        cold = 0
        rows, cols, depth = imageHSV.shape
        for row in range (0, rows):
            for col in range( 0, cols):
                pixel = imageHSV[row, col, 0]
                if pixel < 30:
                    warm += 1 
                elif pixel > 150:
                    warm += 1
                else:
                    cold += 1
        percentageWarm = warm / self.image[:,:,0].size
        percentageCold = cold / self.image[:,:,0].size
        
        return percentageWarm, percentageCold
    
    def fractalDimMinkowskiBoxCount(self, target = 1.5):
        
        Z = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # calculate the threshold on the mean
        mean, standardDeviation = cv2.meanStdDev(Z)
        threshold = mean[0][0]
        
        # Only for 2d image
        assert(len(Z.shape) == 2)
    
        # From https://github.com/rougier/numpy-100 (#87)
        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                   np.arange(0, Z.shape[1], k), axis=1)
    
            # We count non-empty (0) and non-full boxes (k*k)
            return len(np.where((S > 0) & (S < k*k))[0])
    
        # Transform Z into a binary array
        Z = (Z < threshold)
    
        # Minimal dimension of image
        p = min(Z.shape)
    
        # Greatest power of 2 less than or equal to p
        n = 2**np.floor(np.log(p)/np.log(2))
    
        # Extract the exponent
        n = int(np.log(n)/np.log(2))
    
        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2**np.arange(n, 1, -1)
    
        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))
    
        # Fit the successive log(sizes) with log (counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        # print ("this is the fractal {}".format(-coeffs[0]))
        fractal = -coeffs[0]
        
        distFromTarget = abs(target - fractal) 
        # avoiding nan results
        if str(distFromTarget) == 'nan':
            distFromTarget = 1
        
        fractalScoreFromTarget = 1 - distFromTarget
            
        return fractalScoreFromTarget
    

    
    
        
        

