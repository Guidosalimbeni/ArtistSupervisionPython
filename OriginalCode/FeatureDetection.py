# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:22:49 2018

@author: Proprietario
"""

import cv2
import numpy as np
from skimage import measure

# feature detection class

class FeatureDetection():
    
    def __init__(self, imagepath, imagepathTemplate, imagepathTemplateCusp):
        self.image = cv2.imread(imagepath)
        self.templateTangent = cv2.imread(imagepathTemplate)
        self.templateCuspide = cv2.imread(imagepathTemplateCusp, 0)
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        
    
    def detectCircle (self):

        img = cv2.medianBlur(self.gray,5)
        copy = self.image.copy()

        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=10,maxRadius=60)

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
             # draw the outer circle
            cv2.circle(copy,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(copy,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow('detected circles',copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    def detectMultipleBrightSpot (self, minThresh = 200):
        
        blurred = cv2.GaussianBlur(self.gray,(11,11), 0)
        # threshold the image to reveal light regions in the
        # blurred image
        thresh = cv2.threshold(blurred, minThresh, 255, cv2.THRESH_BINARY)[1]
        # erosion and delation to remove noise
        thresh = cv2.erode(thresh, None, iterations = 2)
        thresh = cv2.dilate(thresh, None, iterations = 4)
        # perform a connected component analysis on the thresholded
        # image, then initialize a mask to store only the "large"
        # components
        labels = measure.label(thresh, neighbors=8, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")
         
        # loop over the unique components
        for label in np.unique(labels):
        	# if this is the background label, ignore it
        	if label == 0:
        		continue
         
        	# otherwise, construct the label mask and count the
        	# number of pixels 
        	labelMask = np.zeros(thresh.shape, dtype="uint8")
        	labelMask[labels == label] = 255
        	numPixels = cv2.countNonZero(labelMask)
        	# if the number of pixels in the component is sufficiently
        	# large, then add it to our mask of "large blobs"
        	if numPixels > 300:
        		mask = cv2.add(mask, labelMask)
        
        cv2.imshow('mask brightest spot', mask)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def detectBrightestSpot(self, radius = 5):
        grayImg = self.gray
        grayImg = cv2.GaussianBlur(grayImg, (radius,radius), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayImg)
        image = self.image.copy()
        cv2.circle(image, maxLoc, radius, (255, 0, 0), 2)
        
        cv2.imshow('bright', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        # TODO detect a list of he 3 brightest regions
    
    def detectBlur (self, thresh = 100):
        
        
        
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        fm = cv2.Laplacian(self.gray, cv2.CV_64F).var()
        text = "Not Blurry"
        
    	 # if the focus measure is less than the supplied threshold,
    	 # then the image should be considered "blurry"
        if fm < thresh:
        	text = "Blurry"
        
        # show the image
        blurdetect = self.image.copy()
        cv2.putText(blurdetect, "{}: {:.2f}".format(text, fm), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", blurdetect)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
       
    def extremePointsDetection (self):
        
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        mean = np.mean(blur)
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(blur, mean, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
         
        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        c = max(cnts, key=cv2.contourArea)

        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        image = self.image.copy()
        
        cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
        cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(image, extRight, 8, (0, 255, 0), -1)
        cv2.circle(image, extTop, 8, (255, 0, 0), -1)
        cv2.circle(image, extBot, 8, (255, 255, 0), -1)
         
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def orbDetection(self, maxKeypoints = 500):

        # create ORB object and setup number of keypoints we desire
        orb = cv2.ORB_create(maxKeypoints)
        # determine Keypoints
        keypoints = orb.detect(self.gray, None)
        #obtain the descriptors
        keypoints, descriptors = orb.compute(self.gray, keypoints)
        # draw rich keypoints on input image
        imageout = self.image.copy()
        imageout = cv2.drawKeypoints(imageout, keypoints,imageout, flags = 
                                  cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # mask creation
        blank = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
        
        coordinates = []
        for i, keypoint, in enumerate(keypoints):
            X, Y =  (keypoints[i].pt)
            x = int(X)
            y = int(Y)
            coordinates.append([y,x])
            
        for coord in coordinates:
            #blank[coord[0],coord[1] ] = 255
            cv2.circle(blank,( coord[1],coord[0]), 2, (255,255,255), -1)
        
        kernel = np.ones((5,5),np.uint8)
        blank = cv2.dilate(blank,kernel,iterations = 1)
        blank = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel)
        #4blank = cv2.dilate(blank,kernel,iterations = 2)
        blank = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel)
        
        blank = self.findCentreOfMass( image = blank, contours = None, numOfCntsMax = 3)
        
        cv2.imshow('Feature Method - ORB', imageout)
        cv2.imshow('mask orb', blank)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    def findTriangles(self):
        
        triangleImg = self.image.copy()
        
        kernel = np.ones((4, 4), np.uint8)
        dilation = cv2.dilate(self.gray, kernel, iterations=1)
        
        blur = cv2.GaussianBlur(dilation, (5, 5), 0)

        meanThresh = np.mean(self.gray)
        ret,thresh = cv2.threshold(blur, meanThresh, 255, cv2.THRESH_BINARY)
        
        # Now finding Contours         ###################
        _, contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coordinates = []
        for cnt in contours:
                # [point_x, point_y, width, height] = cv2.boundingRect(cnt)
            approx = cv2.approxPolyDP(
                cnt, 0.07 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                coordinates.append([cnt])
                cv2.drawContours(triangleImg, [cnt], 0, (0, 0, 255), 1)
        
        cv2.imshow("result.png", triangleImg)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    

    def cuspideDetect(self):
        
        # set as scaling consistent
        (tH, tW) = self.templateCuspide.shape[:2]
        found = None
        
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = cv2.resize(self.gray, None, fx = scale, fy = scale)
            r = self.gray.shape[1]/float(resized.shape[1])
            
            # if resized is smaller than the template break the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            
            # find template match
            result = cv2.matchTemplate(self.gray, self.templateCuspide, cv2.TM_CCOEFF)
            (_,maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # if we have found a new maximum correlation value, then ipdate
		     # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
                
                
        # unpack the bookkeeping varaible and compute the (x, y) coordinates
    	 # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
         
    	 # draw a bounding box around the detected result and display the image
        cuspideImg = self.image.copy()
        cv2.rectangle(cuspideImg, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        cv2.imshow("cuspide", cuspideImg)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def tangentDetect(self):
        
        edges = self.edgeDetection(scalarFactor = 1, meanShift = 0)
        #edges = self.edgeDetectionAuto()
        
        
        # first template
        template = np.zeros((12,12), np.uint8)
        template[0:10,0:2] = 255
        template[4:7, 0:10] = 255
        # w and h to also use later to draw the rectagles
        w, h = template.shape[::-1]
        # rotated template
        M = cv2.getRotationMatrix2D((w/2,h/2),90,1)
        template90 = cv2.warpAffine(template,M,(w,h))
        template180 = cv2.warpAffine(template90,M,(w,h))
        template270 = cv2.warpAffine(template180,M,(w,h))
        # run the matchtemplate
        result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF)
        result90 = cv2.matchTemplate(edges, template90, cv2.TM_CCOEFF)
        result180 = cv2.matchTemplate(edges, template180, cv2.TM_CCOEFF)
        result270 = cv2.matchTemplate(edges, template270, cv2.TM_CCOEFF)
        #find the points of match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        threshold = max_val * 0.90
        loc = np.where(result >= threshold)
        loc90 = np.where(result90 >= threshold)
        loc180 = np.where(result180 >= threshold)
        loc270 = np.where(result270 >= threshold)
        #convert edges for display
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        points = []
        for pt  in zip (*loc[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        for pt  in zip (*loc90[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        for pt  in zip (*loc180[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        for pt  in zip (*loc270[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
            
        # penalise the presence of tangents
        #TODO might only need the number of tangents
        score = np.exp(- len(points) / 10)
        
        leftCount = 0
        rightCount = 0
        for p in points:
            if p[0] < self.image.shape[0]:
                leftCount += 1
            else:
                rightCount += 1
        
        diff = abs(leftCount - rightCount)
        scoreDiffinSimmetry = np.exp(- diff / 5)
        
        cv2.imshow('ten', edges)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        return edges , score, scoreDiffinSimmetry
    
    def vertLinesParallelCount(self):
        
        edgeImg = self.edgeDetection(scalarFactor = 1, meanShift = 0)
        
        # first template
        heightTemplate = int(self.image.shape[0] *0.618 / 2)
        template = np.zeros((heightTemplate,10), np.uint8)
        template[:,4:7] = 255


        
        # w and h to also use later to draw the rectagles
        w, h = template.shape[::-1]
        # rotated template
        M = cv2.getRotationMatrix2D((w/2,h/2),90,1)
        template90 = cv2.warpAffine(template,M,(w,h))
        # run the matchtemplate
        result = cv2.matchTemplate(edgeImg, template, cv2.TM_CCOEFF)
        result90 = cv2.matchTemplate(edgeImg, template90, cv2.TM_CCOEFF)

        #find the points of match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        threshold = max_val * 0.90
        loc = np.where(result >= threshold)
        loc90 = np.where(result90 >= threshold)

        #convert edges for display
        edgeImg = cv2.cvtColor(edgeImg, cv2.COLOR_GRAY2BGR)
        
        points = []
        for pt  in zip (*loc[::-1]):
            cv2.rectangle(edgeImg, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        for pt  in zip (*loc90[::-1]):
            cv2.rectangle(edgeImg, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)

            
        # penalise the presence of tangents
        #TODO might only need the number of tangents
        score = np.exp(- len(points) / 10)
        
        leftCount = 0
        rightCount = 0
        for p in points:
            if p[0] < self.image.shape[0]:
                leftCount += 1
            else:
                rightCount += 1
        
        diff = abs(leftCount - rightCount)
        scoreDiffinSimmetry = np.exp(- diff / 5)
        
        cv2.imshow('ten', edgeImg)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        print (leftCount, rightCount )
        
        return edgeImg , score, scoreDiffinSimmetry
        
    def sift_detector(self, new_image = None, image_template = None):
        # Function that compares input image to template
        # It then returns the number of SIFT matches between them
        if new_image == None:
            image1 = self.edgeDetection()
            #image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        
        if image_template == None:
            image2 =  np.zeros((15,15), np.uint8)
            image2[0:15,0:3] = 255
            image2[6:9, 0:15] = 255
        
        # Create SIFT detector object #### paied paied no more free
        sift = cv2.SIFT()
    
        # Obtain the keypoints and descriptors using SIFT
        keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)
    
        # Define parameters for our Flann Matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
        search_params = dict(checks = 100)
    
        # Create the Flann Matcher object
        flann = cv2.FlannBasedMatcher(index_params, search_params)
    
        # Obtain matches using K-Nearest Neighbor Method
        # the result 'matchs' is the number of similar matches found in both images
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
        
        print(matches)
        
        # Store good matches using Lowe's ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m) 
    
        return len(good_matches)
    
    
    
    def fastDetection (self):
        
        # Create FAST Detector object
        fast = cv2.FastFeatureDetector_create()
        
        blank = np.zeros((self.image.shape[0], self.image.shape[1], 1), np.uint8)
        
        # Obtain Key points, by default non max suppression is On
        # to turn off set fast.setBool('nonmaxSuppression', False)
        keypoints = fast.detect(self.gray, None)
        print ("Number of keypoints Detected: ", len(keypoints))
        
        # Draw rich keypoints on input image
        blank = cv2.drawKeypoints(blank, keypoints, blank,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        coordinates = []
        for i, keypoint, in enumerate(keypoints):
            X, Y =  (keypoints[i].pt)
            x = int(X)
            y = int(Y)
            coordinates.append([y,x])

        pts = np.array(coordinates, np.int32)
        pts = pts.reshape((-1,1,2))
        #cv2.polylines(blank, [pts], False, (0,0,255), 1 )
        for coord in coordinates:
            blank[coord[0],coord[1] ] = 255
            cv2.circle(blank,( coord[1],coord[0]), 6, (255,255,255), -1)
            
        
        
        cv2.imshow('Feature Method - FAST', blank)
        cv2.imshow('original', self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
                
        
    
    def cornerDetection (self, maxCorners = 40 , minDistance = 6, midlineOnCornersCnt =  True):
        
        # thecv2.goodFeaturesToTrack(input image, maxCorners, qualityLevel, minDistance)
        
        # We specific the top 50 corners
        corners = cv2.goodFeaturesToTrack(self.gray, maxCorners, 0.01, minDistance )
        
        cornerimg = self.image.copy()
        
        cornersOntheLeft = 0
        cornersOntheRight = 0
        cornersOnTop = 0
        cornersOnBottom = 0
        # find the limit x and y of the detected corners
        listX = [corner[0][0] for corner in corners]
        listY = [corner[0][1] for corner in corners]
        minX = min(listX)
        maxX = max (listX)
        minY = min(listY)
        maxY = max (listY)

        for corner in corners:
            x, y = corner[0]
            x = int(x)
            y = int(y)
            if midlineOnCornersCnt:
                # find the middle x and middle y
                midx = minX + int((maxX - minX)/2)
                midy = minY + int((maxY - minY)/2)
                pass
            else:
                midx = int(self.image.shape[1] / 2)
                midy = int(self.image.shape[0] / 2)
                
            cv2.rectangle(cornerimg,(x-2,y-2),(x+2,y+2),(0,255,0), 1)
            if x < midx:
                cornersOntheLeft += 1
            if x > midx:
                cornersOntheRight += 1
            if y < midy:
                cornersOnTop += 1
            if y > midy:
                cornersOnBottom += 1
                
        
        print (cornersOntheLeft,
        cornersOntheRight,
        cornersOnTop,
        cornersOnBottom)
        cv2.imshow("Corners Found", cornerimg)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    
    def blobDetection (self):
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = False
        params.minArea = 10
        
        detector = cv2.SimpleBlobDetector_create(params)
        
        keypoints = detector.detect (gray)
        
        
        blobs = cv2.drawKeypoints(self.image, keypoints,  np.array([]),(0,255,255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('conv', blobs)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        
    
    def convexDetection (self, maxThresh = 255 ):
        
        imgCopy = self.image.copy()
        gray = cv2.cvtColor(imgCopy,cv2.COLOR_BGR2GRAY)
        meanThresh = np.mean(gray)
        ret,thresh = cv2.threshold(gray, meanThresh, maxThresh, cv2.THRESH_BINARY)
        ing2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        # sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse = False)
 
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(imgCopy, [hull], -1, (0,255,0),1)
            
        cv2.imshow('convex detection', imgCopy)
        cv2.waitKey()
        cv2.destroyAllWindows()

        
    
    def cntApproxDetection (self, scalarFactor = 1, meanShift = 0, method = 'edges'):
        
        if method == 'edges':
            edges = self.edgeDetection(scalarFactor = 1, meanShift = 0)
            ing2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
            
            cv2.imshow('edges befor approx', edges)
            cv2.waitKey()
            cv2.destroyAllWindows()
            
            sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
            
            # select only the bigger contours
            contoursSelection = sorted_contours[0:7]

            
            self.drawApproxContours(contoursSelection)
            
            return contoursSelection
        elif method =='thresh':
            copied = self.image.copy()
            gray = cv2.cvtColor(copied,cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            
            ret,thresh = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
            ing2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
            
            
            sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
            
            # select only the bigger contours
            contoursSelection = sorted_contours[0:7]
            
            
            self.drawApproxContours(contoursSelection)
            return contoursSelection
            
        
    
    def drawApproxContours(self, contours):
        
        copyImg = self.image.copy()
        # iterate trough each contour and compute the approx contour
        approxContours = []
        for cnt in contours:
            #calculate accuracy as a percent of contour perimeter
            accuracy = 0.003 * cv2.arcLength(cnt,True)
            approxCnt = cv2.approxPolyDP(cnt, accuracy, True)
            approxContours.append(approxCnt)
            
        cv2.drawContours(copyImg, approxContours, -1, (0,255,0), 2)
        cv2.imshow('contours_APPRX',copyImg)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    
    def contoursDetection (self, scalarFactor = 1, meanShift = 0):
        edges = self.edgeDetection(scalarFactor = 1, meanShift = 0)
        kernel = np.ones((5,5),np.uint8)
        edges = cv2.dilate(edges,kernel,iterations = 1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        ing2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        
        imgCopy = self.image.copy()
        cv2.drawContours(imgCopy, contours, -1, (255,180,255), -1)
       
        cv2.imshow('contours',imgCopy)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def edgeDetectionAuto(self, sigma = 0.33):
        
        # compute the median of the single channel pixel intensities
        v = np.median(self.image)
        
        copyforedge = self.image.copy()
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(copyforedge, lower, upper)
        
        # return the edged image
        return edged

        
    def edgeDetection(self, scalarFactor = 1, meanShift = 0):
        
        # edges history: edges can be defined as sudden changes(discontinuities)in an image
        # and they can encode just as much information as pixels
        # there are 3 main types of Edge Detection: Sobel, to emphasis vertical,horizontal edges
        # laplacian gets all orientations
        # canny (John F.Canny in 1986) Optimal due to low error rate,well defined edges
        # and accurate detection (1.Applies gaussian bluring, finds intesity gradient,
        # 3.. and 4... see Udemy course)
        
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        
        mean = np.mean(gray)
        mean += meanShift
        std = np.std(gray)

        minThres = int((mean - std) * scalarFactor)
        maxThresh = int((mean+std) * scalarFactor)
        edges = cv2.Canny(gray,minThres,maxThresh,apertureSize = 3)
        
        return edges
    
    def lineDetectionProbHough (self,  scalarFactor = 1, meanShift = 0):
        
        edges = self.edgeDetection(scalarFactor = scalarFactor, meanShift = meanShift)
        
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, 100, 1)
        
        copyImg = self.image.copy()
        
        totalLines = len(lines)
        
        verticalLines = 0
        horizontalLines = 0
        verticalLinesLeft = 0
        verticalLinesRight = 0
        horizontalLinesLeft = 0
        horizontalLinesRight = 0
        allX = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            allX.append(x1)
            allX.append(x2)
        # only horizzontal counts along the middle detection relevance
        midX = int((max(allX) - min(allX))/2) + min(allX)  
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2-x1) *180 / np.pi
            # horizzontal lines
            if angle == 0:
                cv2.line(copyImg, (x1, y1), (x2, y2), (0,0,255), 1)
                horizontalLines += 1
                if x1 < midX and x2 < midX:
                    horizontalLinesLeft += 1
                if x1 > midX and x2 > midX:
                    horizontalLinesRight += 1
            # vertical lines
            if angle == 90 or angle == -90 :
                cv2.line(copyImg, (x1, y1), (x2, y2), (0,255,0), 1)
                verticalLines += 1
                if x1 < midX and x2 < midX:
                    verticalLinesLeft += 1
                if x1 > midX and x2 > midX:
                    verticalLinesRight += 1
        diffVerticals = abs(verticalLinesLeft - verticalLinesRight)
        diffHorizontal = abs(horizontalLinesLeft -horizontalLinesRight )
        
        print (1 - (diffVerticals/verticalLines))
        print (1 - (diffHorizontal / horizontalLines))
      
        cv2.imshow('line', copyImg)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        
    
    def lineDetectionHough(self, scalarFactor = 1, meanShift = 0 ):
        # not useful
        
        copyImg2 = self.image.copy()
        
        edges = self.edgeDetection(scalarFactor = scalarFactor, meanShift = meanShift)
        
        lines = cv2.HoughLines(edges,1,np.pi/180,5, 10)
        
        for rho, theta in lines[0]:
            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
        
            cv2.line(copyImg2,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow('line2', copyImg2)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def findCentreOfMass(self, image = None, contours = None, numOfCntsMax = 7):
        
        if image is None:
            image = self.image
            
        imageCopy = image.copy()
        
        gray = cv2.cvtColor(imageCopy,cv2.COLOR_BGR2GRAY)
        
        #a get the outer silhouette and max one shape only
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
        ret,thresh = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY)
        
        if contours is None:
            ing2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
            contours = sorted_contours[0:numOfCntsMax]

        for c in contours:
            
        	# compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #print ('moment center {} , {} '.format(cX, cY))
         
        	# draw the contour and center of the shape on the image
            cv2.drawContours(imageCopy, [c], -1, (0, 255, 0), 1)
            cv2.circle(imageCopy, (cX, cY), 7, (255, 0, 0), -1)
            cv2.putText(imageCopy, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
        cv2.imshow('center on cnts', imageCopy)
        cv2.waitKey()
        cv2.destroyAllWindows()
      
        
        return imageCopy 
        
        