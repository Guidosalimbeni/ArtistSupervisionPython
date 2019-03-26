# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:35:31 2018

@author: Guido Salimbeni originally developed by Lorenzo

designed to work on mutation and black background

run from command line:
python fractEvo_v2.py -p D:\aaa -c D:\aaa\test.csv

This code take a picture as an input and write a Json file as an output 
the csv file contains the name of the file, the score total score and the individual scores of the 6 features
"""
# import the library for run the script from command
import argparse
# import the library
import cv2
import numpy as np
from scipy.spatial import distance as dist # if complex to install the library I can write another distance algorithm
import os, os.path
import csv
from skimage.measure import compare_ssim as ssim

# construct the argument parse and parse the arguments
# =============================================================================
# ap = argparse.ArgumentParser()
# ap.add_argument("-p","--path", required = True, help = "path to the images to load")
# ap.add_argument("-c","--csvOutputPath", required = True, help = "path to the csv file to write")
# args = vars(ap.parse_args())
# =============================================================================

# the class that contains the 8 features extractions

class FractEvo ():
    
    def __init__ (self, imagepath):
        self.image = cv2.imread(imagepath)
        self.imagepath = imagepath
        self.totalPixels = self.image.size
        self.gray = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
        
    def analyze_area_Lorenzo(self, thresMinLorenzo = 1 , thresMaxLorenzo = 256, 
                             maxArea = 60000,
                             minArea = 5000,
                             penalty = 4,
                             minMeanAreaLitlleCnt = 10,
                             minLenCnts = 3,
                             FractEvoConstraint = True,
                             fractionFractEVO = 2/3):
        
        #copyImage = self.gray.copy()
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        ret,threshImage = cv2.threshold(blurred, thresMinLorenzo, thresMaxLorenzo, cv2.THRESH_BINARY)
        # count number of not black pixels
        totalAreaLorenzo = (threshImage > 0).sum()

        # determine the external silhoutte
        ing2,contours, hierarchy = cv2.findContours(threshImage, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_NONE)
        
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)
            #cv2.drawContours(self.image, [cnt], -1, (255, 0, 255), 1)
        
        # penalise the shape with many little contours
        meanAreaLittleCnt = (sum(areas) - areas[-1]) / len(contours)
        #print (meanAreaLittleCnt, len(contours) )
        # Lorenzo means probably another set of images
        # since there is no relation to 2/3 of the whole 
        # in his code yes there is. is in the factor
        


        totalArea = self.image.shape[0] * self.image.shape[1]
        if FractEvoConstraint == False:
        
            if totalAreaLorenzo < minArea or totalAreaLorenzo > maxArea :
                return (totalAreaLorenzo/totalArea) / penalty
            elif (meanAreaLittleCnt < minMeanAreaLitlleCnt and len(contours) > minLenCnts):
                return (totalAreaLorenzo/totalArea) / penalty
            else:
                return totalAreaLorenzo/totalArea
        if FractEvoConstraint == True:
            
            if totalAreaLorenzo < minArea or totalAreaLorenzo > maxArea :
                
# =============================================================================
#                 cv2.putText(threshImage, "Area {}".format(0), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('area', threshImage)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
                return 0
            else:
                mid = (maxArea - minArea ) * (fractionFractEVO)
                mid += minArea
                r = (totalAreaLorenzo - mid) / mid
                
# =============================================================================
#                 cv2.putText(threshImage, "Area {}".format(1 - abs(r)), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('area', threshImage)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
                return 1 - abs(r)

    def calculateAsymmetryIndex (self, image = None, extraWeight = False, 
                                 MinAreaSilh = 5000, 
                                 MaxAreaSilh = 45000,
                                 penalty = 5):
        if image is None:
            image = self.image
        
        imageCopy = image.copy()
        imageFlipLfRt = cv2.flip(imageCopy, 1)
        imageFlipLf = imageFlipLfRt.copy()
        imageFlipLfHalfCols = int(imageFlipLf.shape [1] / 2)
        imageHalfCols = int(imageCopy.shape [1] / 2)
        
        imageFlipLf [:,0 : imageFlipLfHalfCols ] = 0
        imageCopy [:, 0 : imageHalfCols] = 0
        
        imageFlipLf = imageFlipLf [0 : image.shape[0], imageFlipLfHalfCols : image.shape[1]]
        imageCopy = imageCopy [0 : image.shape[0], imageHalfCols : image.shape[1]]
        
        similaritySSIM = ssim(imageFlipLf,imageCopy, win_size=None, gradient=False, data_range=None, 
                 multichannel = True, gaussian_weights=False, full=False, dynamic_range = None)
        ssimAsymmetry = 1 - similaritySSIM
        
        # for weighting the result of asymmetry to the area of the shape
        ImageCopy = self.image.copy()
        gray = cv2.cvtColor(ImageCopy, cv2.COLOR_BGR2GRAY)
        #a get the outer silhouette and max one shape only
        ret,thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # area foreground
        area = (thresh > 0).sum()
        # aread right and area left of the silhoutte (half orizzontally)
        row = thresh.shape[0]
        col = thresh.shape[1]
        areaDX = (thresh[0:row, 0: int(col/2)]>0).sum()
        areaSX = (thresh[0:row, int(col/2): col]>0).sum()
        
        absDiff = abs(areaDX-areaSX)
        absDiffRatio = absDiff / areaDX
        # calculate what weight to give to the score 
        # penalty big or too little silhoutte
        # add extra weight for area diffeence in 
        # horizz pixel count along the half of the figure
        if area < MinAreaSilh or area > MaxAreaSilh:
            
# =============================================================================
#             cv2.putText(self.image, "Asym {}".format(ssimAsymmetry / penalty), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.imshow('Asym', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            
            return ssimAsymmetry / penalty
        else:
            if extraWeight:
                return ssimAsymmetry +  absDiffRatio
            else:
                
# =============================================================================
#                 cv2.putText(self.image, "Asym {}".format(ssimAsymmetry), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('Asym', self.image)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
#                 
# =============================================================================
                return ssimAsymmetry
    
    def analyze_blobs_Lorenzo(self, thresMinLorenzo = 100, maxBlobs = 350, previousMethod = False,
                              fractionFractEVO = 0.5, minBlobs = 0, minArea = 4):
        #convert to gray
        grayBlob = self.gray.copy()
        # threshold
        ret,thresh = cv2.threshold(grayBlob, thresMinLorenzo, 255, cv2.THRESH_BINARY)
        
        # calculate the total area silhoutte for final score
        graySilh = self.gray.copy()
        ret,threshSilh = cv2.threshold(graySilh, 1, 255, cv2.THRESH_BINARY)
        totalAreaSilhoutte = (threshSilh>0).sum()
        
        # find contours
        ing2, contours, hierarchy = cv2.findContours(thresh,
                                                     cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        
        
        sortedContours = sorted(contours, key = cv2.contourArea, reverse = True)
        selectedContours = [cnt for cnt in sortedContours if cv2.contourArea(cnt) > minArea]
        
        # Approximates a polygonal curve(s) with the specified precision
        contours_poly = []
        radius_list = []
        totalAreaBlobs = 0
        for contour in selectedContours[:]:
            contour_poly = cv2.approxPolyDP(contour, 3, True)
            contours_poly.append(contour_poly)
            (x,y),radius = cv2.minEnclosingCircle(contour_poly)
            radius_list.append(radius)
            # draw contours and circle
            radius = (int(radius))
            
# =============================================================================
#             cv2.drawContours(self.image, [contour], -1, (255, 0, 255), 1)
#             cv2.circle(self.image,(int(x),int(y)),radius,(0, 255, 255), 1)
# =============================================================================
            
            # calculate areas of circles
            areablob = 3.14 * radius **2
            totalAreaBlobs += areablob

        
        n_blobs_Lorenzo = len(radius_list)
        # Lorenzo wrote min 0 max 10 and scale factor 0.5
        # but this values not matches with the set of image that
        # now the values of n_blobs are from 0 to 350 circa
        
        if previousMethod:
            if n_blobs_Lorenzo > maxBlobs:
                
# =============================================================================
#                 cv2.putText(self.image, "Blobs {}".format(0), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('BlobsLots', self.image)
#                 cv2.imshow('thresh', thresh)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
                return 0 
            else:
                mid = (maxBlobs - minBlobs ) * (fractionFractEVO)
                mid += minBlobs
                r = (n_blobs_Lorenzo - mid) / mid
                
# =============================================================================
#                 cv2.putText(self.image, "Blobs {}".format(1 - abs(r)), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('Blobs', self.image)
#                 cv2.imshow('thresh', thresh)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
                return  1 - abs(r)
        else:
            
            r = totalAreaBlobs / totalAreaSilhoutte
            
# =============================================================================
#             cv2.putText(self.image, "Blobs {}".format(r), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.imshow('Blobs', self.image)
#             cv2.imshow('thresh', thresh)
#             cv2.imshow('threshSilh', threshSilh)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return r
        
        
    def findCentreOfMass(self,
                         #distMINWeight = 0.20, 
                         #distMAXWeight = 224,
                         distMin = 20,
                         distMax = 224,
                         fractionFractEVO = 0.5, FractEvoConstraint = False):
        gray = self.gray
        #a get the outer silhouette and max one shape only
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
        ret,thresh = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
        ing2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
        # since we look only at the foreground
        # we take only the bigger contour
        
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        OneBigContour = sorted_contours [:1] # hardcoded for having one only result
        # loop left open to implementation for the FUTURE TODO superclass that has
        # also more contours or background not black
        # loop over the contours - replaced with OneBigContour that is one only contour
        
        for c in OneBigContour:
            
        	# compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
# =============================================================================
#             # area for weight
#             area = cv2.contourArea(c)
# =============================================================================
            # draw circle and write text for the center of cnt
# =============================================================================
#             cv2.circle(self.image, (cX, cY), 7, (255, 255, 255), -1)
#             cv2.putText(self.image, "Center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
#             
# =============================================================================
            
            
        # how far is the centroid to the image centre
        ImageCentreX = self.image.shape[1] / 2
        ImageCentreY = self.image.shape[0] / 2
        
        distanceCentreOfMasstoCentreofImage = dist.euclidean((ImageCentreX, ImageCentreY), (cX, cY))
        
        # higher number means more interesting shape since the shape will be in 
        # the frame however but its centre of mass not in the centre of the image frame
        # more interesting since less trivial
# =============================================================================
#         ratio = area/self.totalPixels
#         weightedDistance = distanceCentreOfMasstoCentreofImage * ratio
# =============================================================================
        
        #if weightedDistance < distMINWeight:
        if distanceCentreOfMasstoCentreofImage < distMin:
            
# =============================================================================
#             cv2.putText(self.image, "CenterMass {}".format(0), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.imshow('CoMass', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return 0
        elif distanceCentreOfMasstoCentreofImage > distMax:
            
# =============================================================================
#             cv2.putText(self.image, "CenterMass {}".format(0), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.imshow('CoMass', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return 0
        else:
            if FractEvoConstraint == True:
                
                
                mid = (abs(distMax - distMin )) * (fractionFractEVO)
                mid += distMin
                r = abs((distanceCentreOfMasstoCentreofImage - mid)) / mid
                
# =============================================================================
#                 cv2.putText(self.image, "CenterMass {}".format(1 - abs(r)), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('CoMass', self.image)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
                return 1 - abs(r)
            else:
                return 1
            
    
    
    
    def analyse_nfeatures_FAST(self, threshLorenzoDetector = 20, maxNum = 1000,
                               minNum = 100,
                               FractEvoConstraint = True, 
                               fractionFractEVO = 0.7):
        # not used NOT USED
        
        grayFast = self.gray
        # create fast detector object
        fast = cv2.FastFeatureDetector_create(threshLorenzoDetector)
        # Obtain Key Points, by default non max suppression is On
        # to turn off set fast.setBool('nonmaxSuppression', False)
        
        keypoints = fast.detect(grayFast, None)
        
        numberOfFeaturesLor = len(keypoints)
        if FractEvoConstraint == False:
            if numberOfFeaturesLor > maxNum:
                return 0
            else:
                return  (1 - (maxNum - numberOfFeaturesLor)/maxNum)*0.7
        elif FractEvoConstraint == True:
                mid = (abs(maxNum - minNum )) * (fractionFractEVO)
                mid += minNum
                r = abs((numberOfFeaturesLor - mid)) / mid
                
# =============================================================================
#                 copyimg = self.image.copy()
#                 copyimg = cv2.drawKeypoints(self.image, keypoints, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#                 cv2.putText(copyimg, "fast {}".format(1 - abs(r)), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('Fast', copyimg)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
                return 1 - abs(r)
        else:
            
# =============================================================================
#             cv2.drawKeypoints(self.image, keypoints,self.image, cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
#             cv2.putText(self.image, "fast {}".format(0), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.imshow('Fast', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return 0
        

    def analyze_holes_Lorenzo (self,  thresMinLorenzo = 10, holeMinArea = 20,
                               maxNumOfHoles = 40,
                               minNumofHoles = 1,
                               FractEvoConstraint = True,
                               fractionFractEVO = 0.5 ):
        
        grayHoles = self.gray
        
        # only for display
        copyfordisplay = self.image.copy()
        
        ret,thresh = cv2.threshold(grayHoles, thresMinLorenzo, 255, cv2.THRESH_BINARY)
        # find contours
        threshCopy = thresh.copy()
        ing2, contours, hierarchies = cv2.findContours(threshCopy,
                                                     cv2.RETR_CCOMP,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        # get the total area of contours
        # loop over the contorurs and count if their area is bigger than
        # of a fixed value holeMinArea
        all_contous_areas = self.utils_get_contour_areas(contours)
  
        counted = []
        noncounted = []
        for  cnt ,area, hierarchy in zip (contours,all_contous_areas, hierarchies[0]):
        
            if hierarchy[2] == -1 and area > holeMinArea:
                counted.append(area)

                cv2.drawContours(copyfordisplay, [cnt], -1, (0, 255, 0), 1)
                
                
            else:
                noncounted.append(area)

        numberOfHolesLorenzo = (len(counted))
        # if there are too many holes it means that the mesh is too noise
        # an alternative would be to add gaussian blur to the initial gray image
        if FractEvoConstraint == False:
            if numberOfHolesLorenzo > maxNumOfHoles:
                return 0
            else:
                return numberOfHolesLorenzo
            
        if FractEvoConstraint == True:
            if numberOfHolesLorenzo < minNumofHoles or numberOfHolesLorenzo > maxNumOfHoles :
                
# =============================================================================
#                 cv2.putText(copyfordisplay, "Holes {}".format(0), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('Holes', copyfordisplay)
#                 cv2.imshow('thresh', thresh)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
                return 0
            else:
                mid = (maxNumOfHoles - minNumofHoles ) * (fractionFractEVO)
                mid += minNumofHoles
                mid = int(mid)
                r = (numberOfHolesLorenzo - mid) / mid
                
# =============================================================================
#                 cv2.putText(copyfordisplay, "Holes {}".format(1 - abs(r)), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('Holes', copyfordisplay)
#                 cv2.imshow('thresh', thresh)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
                return 1 - abs(r)
            
    def clearSilhoutte (self, extraConstraint = False,
                        minDist = 40 , maxDist = 80):
        
        graySilhoutte = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #ImageCopy = self.image.copy()
        #a get the outer silhouette and max one shape only
        Imgblurred = cv2.GaussianBlur(graySilhoutte, (5, 5), 0)
        ret,thresh = cv2.threshold(Imgblurred, 1, 255, cv2.THRESH_BINARY)
        ing2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # draw a bounding rectangle 
        x = 0 ; y = 0 ; w = 0 ; h = 0   
        if len(contours)> 1:
            # no single clear silhoutte
            sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
            
            x,y,w,h = cv2.boundingRect(sorted_contours[0])
            
            #cv2.rectangle(ImageCopy,(x,y),(x+w,y+h),(0,255,0),2)

        else:
            x,y,w,h = cv2.boundingRect(contours[0])
            
            #cv2.rectangle(ImageCopy,(x,y),(x+w,y+h),(0,255,0),2)
            
        
        # check the distances of the corners and the frames
        distLeftHighBoundingBox2LeftHighFrameCorner = dist.euclidean((0,0), (x, y))
        #distRightHighBoundingBox2RightHighFrameCorner = dist.euclidean ((0, self.image.shape[1]), (x, y + w))
        #distLeftLowBoundingBox2LeftLowFrameCorner = dist.euclidean((self.image.shape[0],0), (x + h, y))
        distRightLowBoundingBox2RightLowFrameCorner = dist.euclidean ((self.image.shape[0], self.image.shape[1]), (x + w, y + h))
        # calcualter the ratio of distances
        #DistCentre2Corner = dist.euclidean((self.image.shape[0]/2, self.image.shape[1]/2), (0,0))
        A = distLeftHighBoundingBox2LeftHighFrameCorner 
        #B = distRightHighBoundingBox2RightHighFrameCorner
        #C = distLeftLowBoundingBox2LeftLowFrameCorner 
        D = distRightLowBoundingBox2RightLowFrameCorner 
        
        ratioSilhoutte = (A  + D ) /2
        # the rect touches the border it gives zero score        
        if x == 0 or y == 0 or (x+w) == self.image.shape[1] or (y+h) == self.image.shape[0]:
            
# =============================================================================
#             cv2.putText(self.image, "clear Silh {}".format(0), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.imshow('silh', self.image)
#             cv2.imshow('thresh', thresh)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
                
            
            return 0
        # alternative method to give more info to the method BUT not used
        if extraConstraint == True:
            if ratioSilhoutte > maxDist:
                
                return  0
            elif ratioSilhoutte < minDist:
                
                return 0
            else:
                mid = (maxDist + minDist) / 2
                diff = abs(ratioSilhoutte - mid) / mid
                
                return 1 - diff
        else:
            
# =============================================================================
#             cv2.putText(self.image, "clear Silh {}".format(1), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.imshow('silh', self.image)
#             cv2.imshow('thresh', thresh)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return 1
    
    def perimeterCntVsHoles(self,thresMinLorenzo = 10, holeMinArea = 10, 
                            perimeterThresh = 750, numberOfHolesThresh = 5):
        # If Length of Silhouette is long, then reward high no of holes Else 
        # If Length of Silhouette is short, then reward low no of holes.
        
        Gray = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
        # threshold
        ret,ThresPerim = cv2.threshold(Gray, 1, 255, cv2.THRESH_BINARY)
        threshCopy = ThresPerim.copy()
        ing2, contours, hierarchy = cv2.findContours(threshCopy,
                                                     cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        OneBigContour = sorted_contours [:1]
        
        numberOfHoles,copyfordisplay = self._get_holes_Lorenzo ( thresMinLorenzo = thresMinLorenzo, holeMinArea = holeMinArea)
        
        for cnt in OneBigContour:
            
            perimeter = cv2.arcLength(cnt,True)
            #print ('perimeter' , perimeter)
        
        #print ('number of holes', numberOfHoles)
        
        if perimeter > perimeterThresh:
            if numberOfHoles > numberOfHolesThresh:
                scorePerVsHoles = 1
                
# =============================================================================
#                 cv2.putText(copyfordisplay, "perVsHoles {}".format(1), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('per', copyfordisplay)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
            else:
                scorePerVsHoles = 0
# =============================================================================
#                 
#                 cv2.putText(copyfordisplay, "perVsHoles {}".format(0), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('per', copyfordisplay)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
        if perimeter < perimeterThresh:
            if numberOfHoles < numberOfHolesThresh:
                scorePerVsHoles = 1
                
# =============================================================================
#                 cv2.putText(copyfordisplay, "perVsHoles {}".format(1), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('per', copyfordisplay)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
            else:
                scorePerVsHoles = 0
                
# =============================================================================
#                 cv2.putText(copyfordisplay, "perVsHoles {}".format(0), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#                 cv2.imshow('per', copyfordisplay)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
# =============================================================================
                
        return scorePerVsHoles
    
    def _get_holes_Lorenzo (self,  thresMinLorenzo = 10, holeMinArea = 10):
        
        copyfordisplay = self.image.copy()
        grayHoles = self.gray
        ret,thresh = cv2.threshold(grayHoles, thresMinLorenzo, 255, cv2.THRESH_BINARY)
        # find contours
        threshCopy = thresh.copy()
        ing2, contours, hierarchies = cv2.findContours(threshCopy,
                                                     cv2.RETR_CCOMP,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        # get the total area of contours
        # loop over the contorurs and count if their area is bigger than
        # of a fixed value holeMinArea
        all_contous_areas = self.utils_get_contour_areas(contours)
  
        counted = []
        noncounted = [] # not used
        for  cnt ,area, hierarchy in zip (contours,all_contous_areas, hierarchies[0]):
        
            if hierarchy[2] == -1 and area > holeMinArea:
                counted.append(area)
                
                cv2.drawContours(copyfordisplay, [cnt], -1, (0, 255, 0), 1)
                
            else:
                noncounted.append(area)
                
                
        numberOfHoles = len(counted)
        
        return numberOfHoles, copyfordisplay
    
    def largeAreaReward(self, threshold = 1, targetArea = 45000):
        
        grayArea = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
         # threshold
        ret,thresh = cv2.threshold(grayArea, threshold, 255, cv2.THRESH_BINARY)
        # count the total area of main silhoutte
        totalAreaSilhoutte= (thresh > 0).sum()
        
                # extract the score
        distFromTargetvalue = abs(targetArea - totalAreaSilhoutte)
        distFromTarget = distFromTargetvalue / targetArea
        
        if totalAreaSilhoutte > targetArea:
            
# =============================================================================
#             cv2.putText(self.image, "areaScore {}".format(1), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.putText(self.image, "areat {}".format( totalAreaSilhoutte), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             
#             cv2.imshow('per', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return 1
        else:
# =============================================================================
#             cv2.putText(self.image, "areaScore {}".format(1 - distFromTarget), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.putText(self.image, "area {}".format( totalAreaSilhoutte), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             
#             cv2.imshow('per', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return  1 - distFromTarget

    
    def longLenghtSilhoutteReward(self, targetLenght = 2000):
        
        Gray = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
        # threshold
        ret,ThresPerim = cv2.threshold(Gray, 1, 255, cv2.THRESH_BINARY)
        threshCopy = ThresPerim.copy()
        ing2, contours, hierarchy = cv2.findContours(threshCopy,
                                                     cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        # calculate perimeter bigger contour
        for cnt in sorted_contours [:1]:
            perimeterSilhoutte = cv2.arcLength(cnt,True)
        # extract the score
        distFromTargetvalue = abs(targetLenght - perimeterSilhoutte)
        distFromTarget = distFromTargetvalue / targetLenght
        
        if perimeterSilhoutte > targetLenght:
            
# =============================================================================
#             cv2.putText(self.image, "lenghtScore {}".format(1), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.putText(self.image, "lenght {}".format( perimeterSilhoutte), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             
#             cv2.imshow('per', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return 1
        else:
# =============================================================================
#             cv2.putText(self.image, "lenghtScore {}".format(1 - distFromTarget), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.putText(self.image, "lenght {}".format( perimeterSilhoutte), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             
#             cv2.imshow('per', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return  1 - distFromTarget

    
    def jaggedlySilhoutteReward(self, goodDiffConvexTarget = 600):
                        
        # gives the ratio of the distance in leght between the perimeters
        # of the silhoutte contours and the convex perimenters
        
        # find bigger external contour
        grayConv = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #ImageCopy = self.image.copy()
        # get the outer silhouette and max one shape only
        Imgblurred = cv2.GaussianBlur(grayConv, (5, 5), 0)
        ret,thresh = cv2.threshold(Imgblurred, 1, 255, cv2.THRESH_BINARY)
        ing2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                 
        # single clear silhoutte
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
            
        # calculate its perimeter
        perimeterExternalCnt = cv2.arcLength(contours[0], True)

    
        # calculate convexHull
        hull = cv2.convexHull(contours[0])
        
        #calculate perimenter convexHull
        perimeterHull = cv2.arcLength(hull, True)
        
        # draw contours
# =============================================================================
#         cv2.drawContours(self.image, [hull], -1, (0,255,0), 1)
# =============================================================================
        

        # determine the ratio
        diff = abs(perimeterExternalCnt - perimeterHull)

        # extract the score
        distFromTargetvalue = abs(goodDiffConvexTarget - diff)
        distFromTarget = distFromTargetvalue / goodDiffConvexTarget
        
        if diff > goodDiffConvexTarget:
            
# =============================================================================
#             cv2.putText(self.image, "jaggedly {}".format(1), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.putText(self.image, "diff {}".format( diff), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             
#             cv2.imshow('convex', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return 1
        else:
# =============================================================================
#             cv2.putText(self.image, "Jaggedly {}".format(1 - distFromTarget), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             cv2.putText(self.image, "diff {}".format( diff), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             
#             cv2.imshow('convex', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return  1 - distFromTarget
        
    
    def colorfulnessReward(self, weightedByTheArea = True):
        # measuring colorfulness in an image 
        # paper ref https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf
        # split the image into its respective RGB components
        (B, G, R) = cv2.split(self.image.astype("float"))
     
        # compute rg = R - G
        rg = np.absolute(R - G)
     
        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
     
    	# compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
     
    	# combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
     
    	# derive the "colorfulness" metric
        if weightedByTheArea:
            grayArea = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
            # threshold
            ret,thresh = cv2.threshold(grayArea, 1, 255, cv2.THRESH_BINARY)
            # count the total area of main silhoutte
            totalAreaSilhoutte= (thresh > 0).sum()
            
            # ratio to weight the colorfulness
            ratio = totalAreaSilhoutte / (self.image.shape[0] * self.image.shape[1])
            
            colorfulnessScore = ((stdRoot + (0.3 * meanRoot)) /100) * ratio
            
# =============================================================================
#             cv2.putText(self.image, "colorfulnessScore {}".format(colorfulnessScore), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             
#             cv2.imshow('convex', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            return colorfulnessScore
        
        else:
            
            colorfulnessScore = (stdRoot + (0.3 * meanRoot)) /100
            
# =============================================================================
#             cv2.putText(self.image, "colorfulnessScore {}".format(colorfulnessScore), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#             
#             cv2.imshow('convex', self.image)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            
            
            return colorfulnessScore
        

    
    def contrastReward(self):
        
        #calculate the standard deviation 
        imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # mask the black part
        ret,mask = cv2.threshold(imageGray, 1, 255, cv2.THRESH_BINARY)
        
        mean, standardDeviation = cv2.meanStdDev(imageGray, mask = mask)
        

        
        MaxStD = self.utils_Max_std(imageGray)
        
        contrast = standardDeviation[0][0] / MaxStD[0][0]
        
# =============================================================================
# 
#         
#         cv2.putText(self.image, "contrast {}".format(contrast), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#         
#         cv2.imshow('contrast', self.image)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
 
        return contrast
    
    
    
    def fractalDimMinkowskiBoxCount(self, target = 1.5):
        
        Z = self.gray
        
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
        
        fractal = -coeffs[0]
        
        distFromTargetvalue = abs(target - fractal)
        distFromTarget = distFromTargetvalue / target
        
        # avoiding nan results
        if str(distFromTarget) == 'nan':
            distFromTarget = 0
            
# =============================================================================
#         cv2.putText(self.image, "Fractal {}".format(1 - distFromTarget), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (250, 250, 250), 1)
#         cv2.imshow('fract', self.image)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
            
        return 1 - distFromTarget
    

    
    def utils_Max_std (self, xmin = 0, xmax = 255):
        
        maxStandDevNum = np.sqrt(self.totalPixels * ((xmax - xmin)/2)**2) / np.sqrt(self.totalPixels - 1)
        
        return (maxStandDevNum)
    
    def utils_get_contour_areas (self, contours):
        all_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            all_areas.append(area)
        return all_areas

# this is the code to load the image and run the class to extract the score

# debugging path
path = "D:\\aaa"
#maskpath = "D:\\aaa\\mask"


# working path
#path = "D:\\google drive\\organic\\Machine Learning\\MLCapture256x256"

#path = args["path"]
#☼ original path
#path = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionMaya\\images"
#imageOutputPath = 'D:\\google drive\\A PhD Project at Godlsmiths\\Artist supervision code\\images'
csvOutputPath = 'fractevoScores08may2018_test.csv'
#csvOutputPath = args["csvOutputPath"]

# load the images
imagepaths = []
imageNames = []
valid_images = [".jpg", ".png", ".tga", ".gif"]
for f in os.listdir(path):

    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    if ext.lower() not in valid_images:
        continue
    imagepaths.append(os.path.join(path,f))
    imageNames.append(name)
# store the total number of images
totalNumOfImages = len(imagepaths)  

# list of the features to write
areaLorenzoList= []
nFeaturesList = []
nBlobsList = []
nHolesList = []
asymmetryScoreList = []
clearSilhoutteList = []
centreOfMassList = []
fractalScoreList = []
perimeterVsHolesScoreList = []
longLenghtSilhoutteRewardList = []
jaggedlySilhoutteRewardList = []
largeAreaRewardList = []
colorfulnessRewardList = []
contrastRewardList = []

totalScoreList = []

# make the calculation 
for i in range(totalNumOfImages):
    imagepath = imagepaths[i]
    imageName = imageNames[i]
    #
    
    
    frac = FractEvo(imagepath)
    # 1
    areaLorenzo = frac.analyze_area_Lorenzo( thresMinLorenzo = 1 , thresMaxLorenzo = 256, 
                             maxArea = 60000,
                             minArea = 5000,
                             penalty = 4,
                             minMeanAreaLitlleCnt = 10,
                             minLenCnts = 3,
                             FractEvoConstraint = True,
                             fractionFractEVO = 3/4)
# =============================================================================
#     nFeatures = frac.analyse_nfeatures_FAST(threshLorenzoDetector = 100, maxNum = 1000,
#                                minNum = 100,
#                                FractEvoConstraint = True, 
#                                fractionFractEVO = 0.7)
# =============================================================================
    # 2
    nBlobs = frac.analyze_blobs_Lorenzo(thresMinLorenzo = 100, maxBlobs = 350,previousMethod = False,
                              fractionFractEVO = 0.5, minBlobs = 0, minArea = 20)
    # 3
    nHoles = frac.analyze_holes_Lorenzo(thresMinLorenzo = 10, holeMinArea = 4,
                               maxNumOfHoles = 40,
                               minNumofHoles = 1,
                               FractEvoConstraint = True,
                               fractionFractEVO = 0.5 )
    # 4
    asymmetryScore = frac.calculateAsymmetryIndex(image = None, extraWeight = False, 
                                 MinAreaSilh = 5000, 
                                 MaxAreaSilh = 45000,
                                 penalty = 5)
    # 5
    clearSilhoutte = frac.clearSilhoutte(extraConstraint =  False, minDist =  20, maxDist = 80)
    # 6
    centreOfMass = frac.findCentreOfMass(distMin = 0, distMax = 175,
                         fractionFractEVO = 0.5, FractEvoConstraint = True)
    # 6
    fractalScore = frac.fractalDimMinkowskiBoxCount(target = 1.5)
    # 7
    perimeterVsHolesScore = frac.perimeterCntVsHoles(thresMinLorenzo = 10, holeMinArea = 4, 
                            perimeterThresh = 800, numberOfHolesThresh = 5)
    # 8
    longLenghtSilhoutteReward = frac.longLenghtSilhoutteReward(targetLenght = 2000)
    # 9
    largeAreaReward = frac.largeAreaReward(threshold = 1, targetArea = 30000)
    # 10
    jaggedlySilhoutteReward = frac.jaggedlySilhoutteReward(goodDiffConvexTarget=500)
    # 11
    colorfulnessReward = frac.colorfulnessReward(weightedByTheArea = False)
    # 12
    contrastReward = frac.contrastReward()
    
    areaLorenzoList.append(areaLorenzo)
    #nFeaturesList.append(nFeatures)
    nBlobsList.append(nBlobs)
    nHolesList.append(nHoles)
    asymmetryScoreList.append(asymmetryScore)
    clearSilhoutteList.append(clearSilhoutte)
    centreOfMassList.append(centreOfMass)
    perimeterVsHolesScoreList.append(perimeterVsHolesScore)
    fractalScoreList.append(fractalScore)
    longLenghtSilhoutteRewardList.append(longLenghtSilhoutteReward)
    largeAreaRewardList.append(largeAreaReward)
    jaggedlySilhoutteRewardList.append(jaggedlySilhoutteReward)
    colorfulnessRewardList.append(colorfulnessReward)
    contrastRewardList.append(contrastReward)
    
    A = 1; B = 1 ; C = 1 ; D = 1 ; E = 1 ; F = 1 ; G = 1 
    totalScore = (A * areaLorenzo 
    #+ B * nFeatures 
    + C * nBlobs + D * asymmetryScore + E * clearSilhoutte 
    + F * centreOfMass + G * fractalScore + perimeterVsHolesScore
    + longLenghtSilhoutteReward
    + largeAreaReward
    + jaggedlySilhoutteReward
    + colorfulnessReward
    + contrastReward)
    
    totalScoreList.append(totalScore)
    
   
    del(frac)
# selected features csv file output 

#os.chmod(csvOutputPath, 0777)
file_to_output = open(csvOutputPath, 'w', newline='')
csv_writer = csv.writer(file_to_output, delimiter = ',')
csv_writer.writerow (['file', 
                      'areaLorenzo',
                      #'nFeatures',
                      'nBlobs',
                      'nHoles',
                      'asymmetryScore',
                      'clearSilhoutte',
                      'centreOfMass',
                      'fractalScore',
                      'longLenghtSilhoutteReward',
                      'largeAreaReward',
                      'jaggedlySilhoutteReward',
                      'colorfulnessReward',
                      'contrastReward',
                      'totalScore'
                      ])

for i in range(totalNumOfImages):
    csv_writer.writerow ([imageNames[i],
                          areaLorenzoList[i],
                          #nFeaturesList[i],
                          nBlobsList[i],
                          nHolesList[i],
                          asymmetryScoreList[i],
                          clearSilhoutteList[i],
                          centreOfMassList[i],
                          fractalScoreList[i],
                          longLenghtSilhoutteRewardList[i],
                          largeAreaRewardList[i],
                          jaggedlySilhoutteRewardList[i],
                          colorfulnessRewardList[i],
                          contrastRewardList[i],
                          totalScoreList[i]
                         ])
file_to_output.close()

print ('finished scoring')