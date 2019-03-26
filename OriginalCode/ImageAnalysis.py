# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Standard imports
# Standard imports
import cv2
import numpy as np
#from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from MasksClass import MasksClass

from skimage.measure import compare_ssim as ssim
 

class ForegroundAnalysis ():
    
    def __init__ (self, imagepath, imageOutputPath,imageName):
        self.image = cv2.imread(imagepath)
        self.imagepath = imagepath
        self.totalPixels = self.image.size
        self.imageOutputPath = imageOutputPath
        self.imageName = imageName
        

    
    def utils_Max_std (self, xmin = 0, xmax = 255):
        
        maxStandDevNum = np.sqrt(self.totalPixels * ((xmax - xmin)/2)**2) / np.sqrt(self.totalPixels - 1)
        
        return (maxStandDevNum)
    
    def utils_write (self, suffix, image):
        cv2.imwrite ('{}{}_{}.jpg'.format(self.imageOutputPath,suffix, self.imageName), image )
        return 0
    
    def utils_get_contour_areas (self, contours):
        all_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            all_areas.append(area)
        return all_areas
    
    def chaosEdges (self, penalty = 4, MinAreaSilh = 10000, MaxAreaSilh = 21000):
        grayEdge = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
        mean, std = cv2.meanStdDev(grayEdge)
        # find Canny edges
        edged = cv2.Canny(grayEdge, 200 , 255)
         # threshold
        ret,thresh = cv2.threshold(grayEdge, 1, 255, cv2.THRESH_BINARY)
        # count the total area of main silhoutte
        totalAreaSilhoutte = (thresh > 0).sum()
        # count the total pixels of the edges
        edgedTotalPixels = (edged > 1).sum()
        
        ratioChaos = edgedTotalPixels / totalAreaSilhoutte
        
        if totalAreaSilhoutte < MinAreaSilh or totalAreaSilhoutte > MaxAreaSilh:
            return  (1 - ratioChaos) / penalty
        else:
            return 1 - ratioChaos
        
    def perimeterSilhoutte(self):
        Gray = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
        # threshold
        ret,ThresPerim = cv2.threshold(Gray, 1, 255, cv2.THRESH_BINARY)
        threshCopy = ThresPerim.copy()
        ing2, contours, hierarchy = cv2.findContours(threshCopy,
                                                     cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        OneBigContour = sorted_contours [:1]
        for cnt in OneBigContour:
            
            perimeterSilhoutte = cv2.arcLength(cnt,True)
        return  perimeterSilhoutte 
    
    
    def perimeterCntVsHoles(self):
        # calculate the perimeters eventually in combination with 
        # number of holes
        Gray = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
        # threshold
        ret,ThresPerim = cv2.threshold(Gray, 1, 255, cv2.THRESH_BINARY)
        threshCopy = ThresPerim.copy()
        ing2, contours, hierarchy = cv2.findContours(threshCopy,
                                                     cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        OneBigContour = sorted_contours [:1]
        numberOfHoles = self.analyze_holes_Lorenzo()
        for cnt in OneBigContour:
            
            perimeter = cv2.arcLength(cnt,True)
        
        ratio = perimeter / (self.image.shape[0] * self.image.shape[1])
        
        
        return (numberOfHoles*ratio)
    
    def analyse_clear_silh_Lor (self):
        grayBorder = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
         # threshold
        ret,threshBorder = cv2.threshold(grayBorder, 1, 255, cv2.THRESH_BINARY)
        
        rows = threshBorder.shape[0]
        cols = threshBorder.shape[1]
        
        border1 = (threshBorder[0: rows , 0] == 255).sum()
        border2 = (threshBorder[0: rows , cols - 1] == 255).sum()
        border3 = (threshBorder[0 , 0: cols] == 255).sum()
        border4 = (threshBorder[rows-1 , 0 : cols] == 255).sum()
        if (border1 + border2 + border3 + border4):
            return 0
        else:
            return 1
        
        pass
    
    def analyse_nfeatures_FAST(self, threshLorenzoDetector = 20, maxNum = 1000,
                               minNum = 100,
                               FractEvoConstraint = False, 
                               fractionFractEVO = 0.7):
        grayFast = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
        # create fast detector object
        fast = cv2.FastFeatureDetector_create(threshLorenzoDetector)
        #Obtain Key Points, by default non max suppression is On
        # to turn off set fast.setBool('nonmaxSuppression', False)
        
        keypoints = fast.detect(grayFast, None)
        
        cv2.drawKeypoints(self.image, keypoints, self.image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('fast', self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
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
        
    
    def analyze_area_Lorenzo_Adapted(self, threshForInflatedDetection = 100):
        grayArea = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
         # threshold
        ret,thresh = cv2.threshold(grayArea, 1, 255, cv2.THRESH_BINARY)
        # count the total area of main silhoutte
        totalAreaSilhoutte= (thresh > 0).sum()
         # threshold
        ret,thresh2 = cv2.threshold(grayArea, threshForInflatedDetection, 255, cv2.THRESH_BINARY)
        # collect the countours
         # find contours
        
        threshCopy = thresh2.copy()
        ing2, contours, hierarchy = cv2.findContours(threshCopy,
                                                     cv2.RETR_LIST,
                                                     cv2.CHAIN_APPROX_NONE)
        
                
        all_contous_areas = self.utils_get_contour_areas(contours)
        
        # this returns the sum of the areas of all the contours
        # that remain after the threshhold for inflated area
        inflatedAreaSum = sum(all_contous_areas)
        
        return totalAreaSilhoutte, inflatedAreaSum
    
    def analyze_holes_Lorenzo (self,  thresMinLorenzo = 10, holeMinArea = 10,
                               maxNumOfHoles = 20,
                               minNumofHoles = 1,
                               FractEvoConstraint = False,
                               fractionFractEVO = 0.2 ):
        # USED by Lorenzo to describe number of standalone shapes!!!!
        # ????????? Lorenzo called this holes in func but standaloneshape in 
        # Documentation
        
        grayHoles = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
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
        
        #for cont in contours:
        for  cnt ,area, hierarchy in zip (contours,all_contous_areas, hierarchies[0]):
            #print ('area')
            #print (area)
            #print ('hierarchy')
            #print (hierarchy[2])
        
            if hierarchy[2] == -1 and area > holeMinArea:
                counted.append(area)
                cv2.drawContours(self.image, [cnt], -1, (0, 255, 0), 1)
            else:
                noncounted.append(area)
        
        cv2.imshow('hol', self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
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
                r = (numberOfHolesLorenzo - mid) / numberOfHolesLorenzo
                #done this way to matches Lorenzo documentation
                return 1 - abs(r)
    
    
    def analyze_area_Lorenzo(self, thresMinLorenzo = 1 , thresMaxLorenzo = 256, 
                             maxArea = 21000,
                             minArea = 10000,
                             penalty = 4,
                             minMeanAreaLitlleCnt = 10,
                             minLenCnts = 3,
                             FractEvoConstraint = False,
                             fractionFractEVO = 3/2):
        # *********
        # there is an adapted version
        # *********
        
        grayArea = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
         # threshold
        ret,thresh = cv2.threshold(grayArea, thresMinLorenzo, thresMaxLorenzo, cv2.THRESH_BINARY)
        # count number of not black pixels
        totalAreaLorenzo = (thresh > 0).sum()
        # penalise noise silhoutte
        contourImage = grayArea.copy()
        blurred = cv2.GaussianBlur(contourImage, (5, 5), 0)
                
        ret,threshImage = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
        # determine the external silhoutte
        ing2,contours, hierarchy = cv2.findContours(threshImage, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_NONE)
        
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)
            cv2.drawContours(self.image, [cnt], -1, (255, 0, 255), 1)
        
        # penalise the shape with many little contours
        meanAreaLittleCnt = (sum(areas) - areas[-1]) / len(contours)
        #print (meanAreaLittleCnt, len(contours) )
        # Lorenzo means probably another set of images
        # since there is no relation to 2/3 of the whole 
        # in his code
        
        
        cv2.imshow ('threshImage', self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
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
            

    
    def analyze_blobs_Lorenzo(self, thresMinLorenzo = 100, maxBlobs = 350,
                              fractionFractEVO = 0.5, minBlobs = 0):
        #convert to gray
        grayBlob = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # threshold
        ret,thresh = cv2.threshold(grayBlob, thresMinLorenzo, 255, cv2.THRESH_BINARY)
        # find contours
        threshCopy = thresh.copy()
        ing2, contours, hierarchy = cv2.findContours(threshCopy,
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
            cv2.drawContours(self.image, [contour], -1, (255, 0, 255), 1)
            radius = (int(radius))
         
            cv2.circle(self.image,(int(x),int(y)),radius,(0, 255, 255), 1)
            
        
        cv2.imshow ('blob', self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
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
        
        
    
    def ratioConvexity (self, minPerimeter = 500, maxPerimeter = 1000,
                        penalty = 4):
        # gives the ratio of the distance in leght between the perimeters
        # of the silhoutte contours and the convex perimenters
        
        # find bigger external contour
        grayConv = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #ImageCopy = self.image.copy()
            # get the outer silhouette and max one shape only
        Imgblurred = cv2.GaussianBlur(grayConv, (5, 5), 0)
        ret,thresh = cv2.threshold(Imgblurred, 1, 255, cv2.THRESH_BINARY)
        ing2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                 
        if len(contours)> 1:
            # no single clear silhoutte
            contours = sorted(contours, key = cv2.contourArea, reverse = True)
            
        # calculate its perimeter
        perimeterExternalCnt = cv2.arcLength(contours[0], True)
        
        # calculate convexHull
        hull = cv2.convexHull(contours[0])
        
        #calculate perimenter convexHull
        perimeterHull = cv2.arcLength(hull, True)
        # draw contours
        #cv2.drawContours(ImageCopy, hull, -1, (0,255,0), 1)
        
        #cv2.imshow('contour', ImageCopy )
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        
        # determine the ratio
        diff = perimeterExternalCnt - perimeterHull
        PerimeterRatioConvexity = (diff/perimeterExternalCnt)
        # bigger the better for william
        if perimeterExternalCnt > maxPerimeter or perimeterExternalCnt < minPerimeter:
            PerimeterRatioConvexity = PerimeterRatioConvexity / penalty
            return  PerimeterRatioConvexity
        else:
            return PerimeterRatioConvexity
        
    
    def holesInSilhoutte (self, targetMinAreaCnt = 10 , minHolesNum = 10,
                          maxNumOfHoles = 10,
                          minSilhoutteArea = 10000):
        
        
        grayHole = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
        contourImage = grayHole.copy()
        blurred = cv2.GaussianBlur(contourImage, (5, 5), 0)
                
        ret,threshImage = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
        # determine the external silhoutte
        ing2,contours, hierarchy = cv2.findContours(threshImage, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        externalContours = len(contours)
        
        # determine the total numbers of corners detected inside the external 
        # corner
        
        ing3,contoursAll, hierarchyAll = cv2.findContours(threshImage, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_NONE)
        allContours = len(contoursAll)
        # calcualte the difference in percentage
        numberOfHoles = allContours - externalContours
        
        areas = []
        for cnt in contoursAll:
            area = cv2.contourArea(cnt)
            areas.append(area)
        
        totalSumAreas = sum(areas)
        
        MeanAreaInnerCnts = (totalSumAreas - areas[-1]) / len(areas)
        # establish limitation in size of the holes, number and how big is the 
        # main silhoutte - then set it as a percentage of the max of holes paramenter
        if (MeanAreaInnerCnts < targetMinAreaCnt and numberOfHoles > minHolesNum): 
            return 0 
        elif areas[-1] < minSilhoutteArea:
            return 0
        elif numberOfHoles > maxNumOfHoles:
            numberOfHoles = maxNumOfHoles
            return numberOfHoles / maxNumOfHoles
        else:
            return numberOfHoles / maxNumOfHoles
        
        
    def orbDetection (self, maxKeypoints = 5000,LimitPointScoreRange = 2800, 
                      LimitAreaVsOrb = 0.14, suffix = 'orb',
                      penalty = 2):
        
        GrayOrb = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        imageout = self.image.copy()
        
        # create ORB object and setup number of keypoints we desire
        orb = cv2.ORB_create(maxKeypoints)
        
        
        # determine Keypoints
        keypoints = orb.detect(GrayOrb, None)
        #obtain the descriptors
        keypoints, descriptors = orb.compute(GrayOrb, keypoints)
        numberOfKeypoints =  len(keypoints)
        # draw rich keypoints on input image
        imageout = cv2.drawKeypoints(self.image, keypoints,imageout, flags = 
                                  cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                                        
        # write
        #self.utils_write(suffix, imageout)
        
        # calculate the area of the entire silhoutte to use as weight
        contourImage = GrayOrb.copy()
        blurred = cv2.GaussianBlur(contourImage, (5, 5), 0)
                
        ret,threshImage = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
        # determine the area silhoutte
        totalPixels = (threshImage > 0).sum()
        orbVsArea = numberOfKeypoints/totalPixels
        # for william classification around 3000 is a good point
        
        # orbVsArea = 1 - (abs((numberOfKeypoints - 3000)) / 3000)
        
        if numberOfKeypoints > LimitPointScoreRange :
            return 0
        elif orbVsArea > LimitAreaVsOrb:
            orbVsArea = orbVsArea / penalty
            return orbVsArea
        else:
            
            return orbVsArea
         
    
    def clearSilhoutte (self, suffix = 'Silhoutte', 
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
            cv2.rectangle(ImageCopy,(x,y),(x+w,y+h),(0,255,0),2)
            
            # write rect on disk
            #self.utils_write (suffix, ImageCopy )
        else:
            x,y,w,h = cv2.boundingRect(contours[0])
            cv2.rectangle(ImageCopy,(x,y),(x+w,y+h),(0,255,0),2)
            
            # write rect on disk
            #self.utils_write (suffix, ImageCopy )
        
        # check the distances of the corners and the frames
        distLeftHighBoundingBox2LeftHighFrameCorner = dist.euclidean((0,0), (x, y))
        distRightHighBoundingBox2RightHighFrameCorner = dist.euclidean ((0, self.image.shape[1]), (x, y + w))
        distLeftLowBoundingBox2LeftLowFrameCorner = dist.euclidean((self.image.shape[0],0), (x + h, y))
        distRightLowBoundingBox2RightLowFrameCorner = dist.euclidean ((self.image.shape[0], self.image.shape[1]), (x + h, y + w))
        # calcualter the ratio of distances
        #DistCentre2Corner = dist.euclidean((self.image.shape[0]/2, self.image.shape[1]/2), (0,0))
        A = distLeftHighBoundingBox2LeftHighFrameCorner 
        B = distRightHighBoundingBox2RightHighFrameCorner
        C = distLeftLowBoundingBox2LeftLowFrameCorner 
        D = distRightLowBoundingBox2RightLowFrameCorner 
        
        # multiply the 4 distances together so that if one is 0
        # the rect touches the border it gives zero score
        
        ratioSilhoutte = (A + B + C + D ) /4
        
        if ratioSilhoutte > maxDist:
            return  0
        elif ratioSilhoutte < minDist:
            return 0
        else:
            mid = (maxDist + minDist) / 2
            diff = abs(ratioSilhoutte - mid) / mid
            return 1 - diff
        
    
    def findCentreOfMass(self, suffix = 'centreOfMass', distMINWeight = 0.20, distMAXWeight = 1,
                         fractionFractEVO = 0.5, FractEvoConstraint = False):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ImageCopy = self.image.copy()
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
        
        #print ('number of contours found'+ str(len(OneBigContour)))
        # loop over the contours - replaced with OneBigContour that is one only contour
        
        for c in OneBigContour:
            
        	# compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #print ('moment center {} , {} '.format(cX, cY))
         
        	# draw the contour and center of the shape on the image
            cv2.drawContours(ImageCopy, [c], -1, (0, 255, 0), 2)
            cv2.circle(ImageCopy, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(ImageCopy, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # write on disk
            #self.utils_write(suffix, ImageCopy )
            
            # area for weight
            area = cv2.contourArea(c)
            
            
          # how far is the centroid to the image centre
        ImageCentreX = self.image.shape[1] / 2
        ImageCentreY = self.image.shape[0] / 2
        
        distanceCentreOfMasstoCentreofImage = dist.euclidean((ImageCentreX, ImageCentreY), (cX, cY))
        #dstCentreofImagetoCorner = dist.euclidean((ImageCentreX, ImageCentreY), (0, 0))   
        
        #distanceRatioOfMassVsImageCentre = distanceCentreOfMasstoCentreofImage / dstCentreofImagetoCorner
        
        cv2.imshow('center', ImageCopy)
        cv2.waitKey()
        cv2.destroyAllWindows()
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
        
        
    def calculateAsymmetryIndex (self, image = None, extraWeight = True, 
                                 MinAreaSilh = 10000, 
                                 MaxAreaSilh = 21000,
                                 penalty = 5):
        # calculate the the hist correlation of flipped image
        # if image is not provided uses self.image
        if image is None:
            image = self.image
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        
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
                
        # calculate the hist of the two halfs
        #https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
        
        histHalfOrig = cv2.calcHist(imageCopy, [0,1,2], None, [4,4,4],
                                    [0,256,0,256,0,256])
        histHalfFlipLf = cv2.calcHist(imageFlipLf, [0,1,2], None, [4,4,4],
                                    [0,256,0,256,0,256])
        
        similarity = cv2.compareHist(histHalfOrig, histHalfFlipLf, 0)
        
        # Alternative Symmetry calculate the mean square error:
        # similarityMSE = self.mse(histHalfOrig, histHalfFlipLf)
        # alternative with SSIM
        similaritySSIM = self.compare_images (imageFlipLf, imageCopy)
        
        #print ('hist similarity: {}'.format(similarity))
        #☻print ('MSE similarity : {}'.format(similarityMSE))
        #print ('SSIM similarity: {}: '.format(similaritySSIM))
        histAsymmetry = 1 - similarity # not used only using ssim similarity score
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
    
    def mse (self,imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        	
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err
    
    def compare_images(self, imageA, imageB):
        
        #imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        #imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
        s = ssim(imageA,imageB, win_size=None, gradient=False, data_range=None, 
                 multichannel = True, gaussian_weights=False, full=False, dynamic_range = None)
        
        return s
        
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
            
        return 1 - distFromTarget
        
    
    def fourierOnAdaptiveThresh (self, startAngle, endAngle, suffix = 'Adapt_thresh'):
        imageBlurred = cv2.GaussianBlur(self.image, (3,3), 0)
        # calculate the threshold on the mean
        mean, standardDeviation = cv2.meanStdDev(imageBlurred)
        threshold = mean[0][0]
        ret,th1 = cv2.threshold(imageBlurred,threshold,255,cv2.THRESH_BINARY)
                
        meanDft, standardDeviationDft, imageFourierThresh = self.makeFourier(th1)
        
        rotation = 360 / (endAngle-startAngle)
        iteration = int(rotation)
        
        stdListRatio = []
        pixelsCountedInSliceRatio = []
        for cut in range(iteration):
            maskSlice = MasksClass(imageFourierThresh)
                       
            mask = maskSlice.createCakeSlicesMask(startAngle, endAngle)
            
            mean, std = cv2.meanStdDev(imageFourierThresh, mask = mask)
            startAngle += 360/rotation
            endAngle += 360/rotation
            
            #maxStdDev = self.utils_Max_std (0,255) # only in case of bigger than image 255 value
            # stdRatio = std[0][0] / maxStdDev # only if img 255 
            stdListRatio.append(std) #check if makes sense
            # to write the fourier
            # and to count the white over black in the slices
            imageGrayDft = np.array(imageFourierThresh * 255, dtype = np.uint8)
            #cv2.imwrite ('{}{}dft_{}.jpg'.format(self.imageOutputPath,suffix, self.imageName), imageGrayDft )
            sliced = cv2.bitwise_and(imageGrayDft, mask)
            meanSlice, stdSlice = cv2.meanStdDev(imageGrayDft, mask = mask)
            slicedTotalPixels = (sliced > 1).sum()
            slicedTotalFrequencyPixel = (sliced > meanSlice).sum() # threshhold with mean distribution
                        
            slicedRatioPixelCount = slicedTotalFrequencyPixel / slicedTotalPixels
            pixelsCountedInSliceRatio.append(slicedRatioPixelCount)
            
        # calculate the standard deviation of the standard deviation list
        # and the std deviation on the difference of pixel count for slices
        standardDeviationWithinTheStandardDev = np.std(stdListRatio)
        StdPixCountInSliceRatio = np.std(pixelsCountedInSliceRatio)
        
        return standardDeviationWithinTheStandardDev , StdPixCountInSliceRatio
    
        
    def fourierMeanPerSlices (self, startAngle, endAngle, suffix = 'default'):
        
        meanDft, standardDeviationDft, imagefromDft = self.makeFourier(image = self.image)
        
        rotation = 360 / (endAngle-startAngle)
        iteration = int(rotation)
        
        stdListRatio = []
        pixelsCountedInSliceRatio = []
        for cut in range(iteration):
            maskSlice = MasksClass(imagefromDft)
                       
            mask = maskSlice.createCakeSlicesMask(startAngle, endAngle)
            
            mean, std = cv2.meanStdDev(imagefromDft, mask = mask)
            startAngle += 360/rotation
            endAngle += 360/rotation
            
            #maxStdDev = self.utils_Max_std (0,255) # only in case of bigger than image 255 value
            # stdRatio = std[0][0] / maxStdDev # only if img 255 
            stdListRatio.append(std) #check if makes sense
            # to write the fourier
            # and to count the white over black in the slices
            imageGrayDft = np.array(imagefromDft * 255, dtype = np.uint8)
            #cv2.imwrite ('{}{}dft_{}.jpg'.format(self.imageOutputPath,suffix, self.imageName), imageGrayDft )
            sliced = cv2.bitwise_and(imageGrayDft, mask)
            meanSlice, stdSlice = cv2.meanStdDev(imageGrayDft, mask = mask)
            slicedTotalPixels = (sliced > 1).sum()
            slicedTotalFrequencyPixel = (sliced > meanSlice).sum() # threshhold with mean distribution
                        
            slicedRatioPixelCount = slicedTotalFrequencyPixel / slicedTotalPixels
            pixelsCountedInSliceRatio.append(slicedRatioPixelCount)
            
        # calculate the standard deviation of the standard deviation list
        # and the std deviation on the difference of pixel count for slices
        standardDeviationWithinTheStandardDev = np.std(stdListRatio)
        StdPixCountInSliceRatio = np.std(pixelsCountedInSliceRatio)
        
        return standardDeviationWithinTheStandardDev , StdPixCountInSliceRatio
    
    def areaForegroundVsbackground (self):
        imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # blur to increase extraction
        imageBlurred = cv2.GaussianBlur(imageGray, (3,3), 0)
        '''
        # without blur
        ret, threshImageBinary = cv2.threshold(imageGray, 1, 255, cv2.THRESH_BINARY)
        __, threshAdaptive = cv2.threshold(imageGray, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        '''
        ret, threshImageBinary = cv2.threshold(imageBlurred, 1, 255, cv2.THRESH_BINARY)
        # __, threshAdaptive = cv2.threshold(imageBlurred, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # count the number of pixel greater than 1 so in the area foreground
        totalPixelInForeground = (threshImageBinary > 1).sum()
        
        ratioForeVsBackground = totalPixelInForeground / threshImageBinary.size
        
        #cv2.imshow("bin", threshImageBinary )
        #cv2.imshow("adapt", threshAdaptive )
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return ratioForeVsBackground
    
    def calculateEntropy(self):
        
        # to check ...****
        imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # establish th enumber of bins
        # max vakue of entropy ranges from 0 to 8 in grayscale
        hist = cv2.calcHist(imageGray, [0], None, [256], [0, 256], 1, 0)
        
        hist /= imageGray.size
        
        entropy=np.sum(-hist*(np.log2(hist)))
        #print (entropy)
        return entropy
    
    
    def calculateRuleOfThird(self):
        #use mask
        #calculate the number of corners
        totalCorner =  self.calculateNumbersOfCorners()
        imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #image = image[:,:,0]
              
        # the conrenrHarris requires the array datatype to be float32
        imageGray = np.float32(imageGray)
        harrisCorner = cv2.cornerHarris(imageGray, 3, 3, 0.05)
        
        harrisNorm = cv2.normalize(harrisCorner, 0 , 255, cv2.NORM_MINMAX )
        harrisScaled = cv2.convertScaleAbs(harrisNorm)
        '''
        print (harrisScaled.size)
        print (harrisScaled.sum())
        '''
        rows = harrisScaled.shape[0]
        cols = harrisScaled.shape[1]
        harrisScaled[0: rows//12*2 , :] = 0
        harrisScaled[rows//12*10: rows , :] = 0
        harrisScaled[: , 0 : cols//12*2] = 0
        harrisScaled[: , cols//12*10 : cols ] = 0
        harrisScaled[rows//12*4 : rows//12*8, cols//12*4 : cols//12*8] = 0
        '''
        print (harrisScaled.size)
        print (harrisScaled.sum())
        '''
        #counting number of corners
        thresh = 25
        
        totalRuleOfThirdCorners = (harrisScaled > thresh).sum()
        RuleOfThird = totalRuleOfThirdCorners / totalCorner 
        '''
        totalRuleOfThirdCorners = 0
        rows, cols = harrisScaled.shape
        for row in range (0, rows):
            for col in range( 0, cols):
                if harrisScaled[row, col] > thresh :
                    totalRuleOfThirdCorner += 1
        print (totalRuleOfThirdCorner)
        '''
        
        return RuleOfThird
    
    def makeFourier(self, image):
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(imageGray),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
      
        cv2.normalize( magnitude_spectrum, magnitude_spectrum, alpha = 0 , beta = 1 , norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        imgDftGray = np.uint8(magnitude_spectrum)
        imgDftRGB = cv2.cvtColor(imgDftGray, cv2.COLOR_GRAY2RGB)
        
        meanDft, standardDeviationDft = cv2.meanStdDev(imgDftRGB)
        return meanDft[0][0], standardDeviationDft[0][0] , magnitude_spectrum
        
    
    def calculateNumbersOfCorners(self, minAreaForeground = 10000,
                                                maxAreaForeground = 21000, 
                                                penalty = 2.5,
                                                inflatedRatio = 0.11):
        # calculate the numbers of corners using Harris Corner
        # intersection of two edge
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # the conrenrHarris requires the array datatype to be float32
        gray = np.float32(gray)
        harrisCorner = cv2.cornerHarris(gray, 5, 5, 0.03)
        
        harrisNorm = cv2.normalize(harrisCorner, 0 , 255, cv2.NORM_MINMAX )
        harrisScaled = cv2.convertScaleAbs(harrisNorm)
        
        #counting number of corners
        thresh = 10
        totalCorner = 0
        
        rows, cols = harrisScaled.shape
        for row in range (0, rows):
            for col in range( 0, cols):
                if harrisScaled[row, col] > thresh :
                    totalCorner += 1
        # calculate the inflated ratio to determine another penalty system
        totalAreaSilhoutte, inflatedAreaSum = self.analyze_area_Lorenzo_Adapted()
        ratioInflatedAreaVsTotalArea =  inflatedAreaSum / totalAreaSilhoutte            
        #` penalaise image too big or too small by saturation boundaries
        
        if totalAreaSilhoutte < minAreaForeground or totalAreaSilhoutte > maxAreaForeground:
            
            return int(totalCorner / (penalty * 2))
        elif ratioInflatedAreaVsTotalArea < inflatedRatio:
            return int(totalCorner / (penalty ))
        else: 
            
            return totalCorner
               
    
    def calculateSaturation(self, MINMeanSatWholeimg = 0.09, MAXMeanSatWholeimg = 0.165, 
                            penalty = 2.5, inflatedRatio = 0.11):
        # Function for calculating the overall mean Saturation of the image
        # convert the image into LAB color space
        imageHSV = cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
        #create mask for extraxt the foreground
        maskObj = MasksClass(self.image)
        maskForeground = maskObj.createMaskFromThreshold(Min_threshold=1, Max_threshold=255)
        
        # calculate the mean on the saturation channel
        meanTot, standardDeviationTot = cv2.meanStdDev(imageHSV)
        meanFore, standardDeviationFore = cv2.meanStdDev(imageHSV, mask = maskForeground)
        # output the mean saturation of the whole image and the mean saturatio of the foreground
        meanTotSaturationPercentage = meanTot[1][0] / 255
        meanForeSaturationPercentage = meanFore[1][0] / 255
      
        # calculate the inflated ratio to determine another penalty system
        totalAreaSilhoutte, inflatedAreaSum = self.analyze_area_Lorenzo_Adapted()
        ratioInflatedAreaVsTotalArea =  inflatedAreaSum / totalAreaSilhoutte
        
        #` penalaise image too big or too small by saturation boundaries
        
        if meanTotSaturationPercentage < MINMeanSatWholeimg or meanTotSaturationPercentage > MAXMeanSatWholeimg:
            
            return meanForeSaturationPercentage / (penalty * 2)
        elif ratioInflatedAreaVsTotalArea < inflatedRatio:
            return meanForeSaturationPercentage / penalty
        else: 
            return meanForeSaturationPercentage

    def calculateContrast(self, minAreaForeground = 10000,
                                                maxAreaForeground = 21000, 
                                                penalty = 2.5,
                                                inflatedRatio = 0.11):
        #calculate the standard deviation 
        imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mean, standardDeviation = cv2.meanStdDev(imageGray)
        MaxStD = self.utils_Max_std(imageGray)
        
        contrast = standardDeviation[0][0] / MaxStD[0][0]
        
        totalAreaSilhoutte, inflatedAreaSum = self.analyze_area_Lorenzo_Adapted()
        ratioInflatedAreaVsTotalArea =  inflatedAreaSum / totalAreaSilhoutte
        
        if totalAreaSilhoutte < minAreaForeground or totalAreaSilhoutte > maxAreaForeground:
            return 0
        elif ratioInflatedAreaVsTotalArea < inflatedRatio:
    
            return  contrast / penalty 
        else: 
            return contrast
        
        
    
    def calculateBrightness(self, minAreaForeground = 1000,
                                                maxAreaForeground = 21000, 
                                                penalty = 2,
                                                inflatedRatio = 0.11):
        # Function for calculating the overall mean Brightness of the image
    	# convert the image into LAB color space
        imageLab = cv2.cvtColor(self.image,cv2.COLOR_BGR2Lab)
        # calculate the value from 0 to 100 of the average luminosity of the image
        totalPixel = 0
        totalBrightness = 0
        rows, cols, depth = imageLab.shape
        for row in range (0, rows):
            for col in range( 0, cols):
                totalPixel += 1
                pixel = imageLab[row, col, 0]
                totalBrightness += pixel
        brightnessScore = totalBrightness / totalPixel /100
        
        totalAreaSilhoutte, inflatedAreaSum = self.analyze_area_Lorenzo_Adapted()
        ratioInflatedAreaVsTotalArea =  inflatedAreaSum / totalAreaSilhoutte
        
        if totalAreaSilhoutte < minAreaForeground or totalAreaSilhoutte > maxAreaForeground:
            return 0
        elif ratioInflatedAreaVsTotalArea < inflatedRatio:
    
            return  brightnessScore / penalty 
        else: 
            return brightnessScore
        
    
    def image_colorfulness(self, minAreaForeground = 10000,
                                                maxAreaForeground = 21000, 
                                                penalty = 2.5,
                                                inflatedRatio = 0.11):
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
     
    	# derive the "colorfulness" metric and return i
    
        totalAreaSilhoutte, inflatedAreaSum = self.analyze_area_Lorenzo_Adapted()
        ratioInflatedAreaVsTotalArea =  inflatedAreaSum / totalAreaSilhoutte
        
        if totalAreaSilhoutte < minAreaForeground or totalAreaSilhoutte > maxAreaForeground:
            return  0 
        elif ratioInflatedAreaVsTotalArea < inflatedRatio:
            return ((stdRoot + (0.3 * meanRoot)) /100) / penalty
        else: 
            return (stdRoot + (0.3 * meanRoot)) /100
            
    
    def calculateWarmOrCold(self, minAreaForeground = 10000, maxAreaForeground = 21000, 
                            penalty = 5):
        # function to calculating if the image is more Warm or more cold
        # 1 is warm 0 is cold
        # depending on the Hue in the HSV colour space
        # cnvert the image into HSV
        imageHSV = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        #calculate the percentage, assuming warm from 150 to 0 and 0 to 30
        # and cold from 31 to 149
        totalPixel = 0
        warm = 0
        cold = 0
        classes =  [warm, cold]
        rows, cols, depth = imageHSV.shape
        #print (imageHSV.shape)
        # looping through the imageMat to get the Hue value per pixels and add to 
        # The correponding class (warm or cold) (indices [0] refers to H in HSV)
        
        
        for row in range (0, rows):
            for col in range( 0, cols):
                totalPixel += 1
                pixel = imageHSV[row, col, 0]
                if pixel > 0 and pixel < 30 or pixel > 150 :
                    classes[1] += 1
                else:
                    classes[0] += 1
        
        percentageWarm = (classes[1] / totalPixel) 
        
        grayWarm = cv2.cvtColor (self.image, cv2.COLOR_BGR2GRAY)
         # threshold
        ret,thresh = cv2.threshold(grayWarm, 1, 255, cv2.THRESH_BINARY)
        # count the total area of main silhoutte
        totalAreaSilhoutte = (thresh > 0).sum()
        
        if totalAreaSilhoutte < minAreaForeground or totalAreaSilhoutte > maxAreaForeground:
            return  percentageWarm / penalty
        else:
    
            return percentageWarm


