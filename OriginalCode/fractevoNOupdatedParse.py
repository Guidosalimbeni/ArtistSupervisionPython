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
ap = argparse.ArgumentParser()
ap.add_argument("-p","--path", required = True, help = "path to the images to load")
ap.add_argument("-c","--csvOutputPath", required = True, help = "path to the csv file to write")
args = vars(ap.parse_args())

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
                             fractionFractEVO = 3/4):
        
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
        # in his code

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
                return 0
            else:
                mid = (maxArea - minArea ) * (fractionFractEVO)
                mid += minArea
                r = (totalAreaLorenzo - mid) / mid
                return 1 - abs(r)

    def calculateAsymmetryIndex (self, image = None, extraWeight = False, 
                                 MinAreaSilh = 10000, 
                                 MaxAreaSilh = 21000,
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
            return ssimAsymmetry / penalty
        else:
            if extraWeight:
                return ssimAsymmetry +  absDiffRatio
            else:
                return ssimAsymmetry
    
    def analyze_blobs_Lorenzo(self, thresMinLorenzo = 100, maxBlobs = 350,
                              fractionFractEVO = 0.5, minBlobs = 0):
        #convert to gray
        grayBlob = self.gray.copy()
        # threshold
        ret,thresh = cv2.threshold(grayBlob, thresMinLorenzo, 255, cv2.THRESH_BINARY)
        # find contours
        ing2, contours, hierarchy = cv2.findContours(thresh,
                                                     cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key = cv2.contourArea)
        # Approximates a polygonal curve(s) with the specified precision
        contours_poly = []
        radius_list = []
        for contour in contours[:-2]:
            contour_poly = cv2.approxPolyDP(contour, 3, True)
            contours_poly.append(contour_poly)
            (x,y),radius = cv2.minEnclosingCircle(contour_poly)
            radius_list.append(radius)
            #cv2.drawContours(self.image, [contour], -1, (255, 0, 255), 1)
            radius = (int(radius))
            #cv2.circle(self.image,(int(x),int(y)),radius,(0, 255, 255), 1)

        
        n_blobs_Lorenzo = len(radius_list)
        # Lorenzo wrote min 0 max 10 and scale factor 0.5
        # but this values not matches with the set of image that
        # now the values of n_blobs are from 150 to 450 circa
        
        if n_blobs_Lorenzo > maxBlobs:
            return 0 
        else:
            mid = (maxBlobs - minBlobs ) * (fractionFractEVO)
            mid += minBlobs
            r = (n_blobs_Lorenzo - mid) / mid
            return  1 - abs(r)
        
        
    def findCentreOfMass(self,distMINWeight = 0.20, distMAXWeight = 1,
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
            # area for weight
            area = cv2.contourArea(c)
            
            
        # how far is the centroid to the image centre
        ImageCentreX = self.image.shape[1] / 2
        ImageCentreY = self.image.shape[0] / 2
        
        distanceCentreOfMasstoCentreofImage = dist.euclidean((ImageCentreX, ImageCentreY), (cX, cY))
        
        # higher number means more interesting shape since the shape will be in 
        # the frame however but its centre of mass not in the centre of the image frame
        # more interesting since less trivial
        ratio = area/self.totalPixels
        weightedDistance = distanceCentreOfMasstoCentreofImage * ratio
        
        if weightedDistance < distMINWeight:
            return 0
        elif weightedDistance > distMAXWeight:
            return 0
        else:
            if FractEvoConstraint == True:
                return weightedDistance * fractionFractEVO
            else:
                return weightedDistance
            
    def analyse_nfeatures_FAST(self, threshLorenzoDetector = 100, maxNum = 1000,
                               minNum = 100,
                               FractEvoConstraint = True, 
                               fractionFractEVO = 0.7):
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
                mid = (maxNum - minNum ) * (fractionFractEVO)
                mid += minNum
                r = (numberOfFeaturesLor - mid) / mid
                return 1 - abs(r)
        else:
            return 0
        

    def analyze_holes_Lorenzo (self,  thresMinLorenzo = 20, holeMinArea = 10,
                               maxNumOfHoles = 20,
                               minNumofHoles = 1,
                               FractEvoConstraint = True,
                               fractionFractEVO = 0.5 ):
        
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
        noncounted = []
        for  cnt ,area, hierarchy in zip (contours,all_contous_areas, hierarchies[0]):
        
            if hierarchy[2] == -1 and area > holeMinArea:
                counted.append(area)
                #cv2.drawContours(self.image, [cnt], -1, (0, 255, 0), 1)
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
                
                return 0
            else:
                mid = (maxNumOfHoles - minNumofHoles ) * (fractionFractEVO)
                mid += minNumofHoles
                mid = int(mid)
                r = (numberOfHolesLorenzo - mid) / mid
                
                return 1 - abs(r)
            
    def clearSilhoutte (self, extraConstraint = False,
                        minDist = 40 , maxDist = 80):
        
        graySilhoutte = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ImageCopy = self.image.copy()
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
            else:
                scorePerVsHoles = 0
                
        if perimeter < perimeterThresh:
            if numberOfHoles < numberOfHolesThresh:
                scorePerVsHoles = 1
            else:
                scorePerVsHoles = 0
                
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
        
        distFromTarget = abs(target - fractal) 
        # avoiding nan results
        if str(distFromTarget) == 'nan':
            distFromTarget = 1
        
        return 1 - distFromTarget
    
    def utils_get_contour_areas (self, contours):
        all_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            all_areas.append(area)
        return all_areas

# this is the code to load the image and run the class to extract the score

# debugging path
#path = "D:\\aaa"
path = args["path"]
#☼ original path
#path = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionMaya\\images"
#imageOutputPath = 'D:\\google drive\\A PhD Project at Godlsmiths\\Artist supervision code\\images'
#csvOutputPath = 'FeaturesScoresWtest.csv'
csvOutputPath = args["csvOutputPath"]

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
totalScoreList = []
# make the calculation 
for i in range(totalNumOfImages):
    imagepath = imagepaths[i]
    imageName = imageNames[i]
    
    frac = FractEvo(imagepath)
    areaLorenzo = frac.analyze_area_Lorenzo( thresMinLorenzo = 1 , thresMaxLorenzo = 256, 
                             maxArea = 60000,
                             minArea = 5000,
                             penalty = 4,
                             minMeanAreaLitlleCnt = 10,
                             minLenCnts = 3,
                             FractEvoConstraint = True,
                             fractionFractEVO = 3/4)
# =============================================================================
#     nFeatures = frac.analyse_nfeatures_FAST(threshLorenzoDetector = 20, maxNum = 1000,
#                                minNum = 100,
#                                FractEvoConstraint = True, 
#                                fractionFractEVO = 0.7)
# =============================================================================
    nBlobs = frac.analyze_blobs_Lorenzo(thresMinLorenzo = 100, maxBlobs = 350,
                              fractionFractEVO = 0.5, minBlobs = 0)
    nHoles = frac.analyze_holes_Lorenzo(thresMinLorenzo = 10, holeMinArea = 10,
                               maxNumOfHoles = 20,
                               minNumofHoles = 1,
                               FractEvoConstraint = True,
                               fractionFractEVO = 0.5 )
    asymmetryScore = frac.calculateAsymmetryIndex(image = None, extraWeight = False, 
                                 MinAreaSilh = 5000, 
                                 MaxAreaSilh = 45000,
                                 penalty = 5)
    clearSilhoutte = frac.clearSilhoutte(extraConstraint =  False, minDist = 20 , maxDist = 80)
    centreOfMass = frac.findCentreOfMass(distMINWeight = 0.20, distMAXWeight = 1,
                         fractionFractEVO = 0.5, FractEvoConstraint = True)
    fractalScore = frac.fractalDimMinkowskiBoxCount(target = 1.5)
    perimeterVsHolesScore = frac.perimeterCntVsHoles(thresMinLorenzo = 20, holeMinArea = 10, 
                            perimeterThresh = 750, numberOfHolesThresh = 5)
    
    areaLorenzoList.append(areaLorenzo)
    #nFeaturesList.append(nFeatures)
    nBlobsList.append(nBlobs)
    nHolesList.append(nHoles)
    asymmetryScoreList.append(asymmetryScore)
    clearSilhoutteList.append(clearSilhoutte)
    centreOfMassList.append(centreOfMass)
    fractalScoreList.append(fractalScore)
    perimeterVsHolesScoreList.append(perimeterVsHolesScore)
    A = 1; B = 1 ; C = 1 ; D = 1 ; E = 1 ; F = 1 ; G = 1 
    totalScore = A * areaLorenzo 
    #+ B * nFeatures 
    + C * nBlobs + D * asymmetryScore + E * clearSilhoutte + F * centreOfMass + G * fractalScore + perimeterVsHolesScore
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
                          totalScoreList[i]
                         ])
file_to_output.close()
