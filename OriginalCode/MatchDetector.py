# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 22:31:36 2018

@author: OWNER
"""
import numpy as np
import cv2

class MatchDetector ():
    
    def __init__ (self, imagepath, imageOutputPath,imageName, FaceClassifierPath):
        self.image = cv2.imread(imagepath)
        self.imagepath = imagepath
        self.totalPixels = self.image.size
        self.imageOutputPath = imageOutputPath
        self.imageName = imageName
        self.FaceClassifierPath = FaceClassifierPath
    
    def sift_detector(self):
        pass
    def orb_detector(self):
        # see tutorial on Udemy .. to implement with parallel lines
        # or tangents by first converting in grayscale or looking
        # to canny edges images
        pass
    
    def faceDetection (self):
        # point to the classiefier
        face_classifier = cv2.CascadeClassifier(self.FaceClassifierPath + str('haarcascade_frontalface_default.xml'))
        imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # the classifier returns the ROI as a tuple
        # it stores the top left coordinate and the bottom right coordinates
        faces = face_classifier.detectMultiScale(imageGray, 1.08, 3)
        # proper parameter should be 1,3 and 3 
        
        if faces is ():
            # good not to have faces
            return 1 
        else:
            
            return 1 - (len(faces) * 0.1)
        
        '''
        for (x, y, w, h) in faces:
            cv2.rectangle(self.image, (x,y), (x+w, y+h), (127,0,255), 2)
            cv2.imshow('face detection', self.image)
            cv2.waitKey()
        
            cv2.destroyAllWindows
        '''