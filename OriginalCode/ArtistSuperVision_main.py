# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:29:24 2018

Main in case no using the UI interface
this main is more used for testing at the moment

@author: Guido Salimbeni
"""



from ImageAnalysis import ForegroundAnalysis
from MatchDetector import MatchDetector
from FeatureDetection import FeatureDetection
from ColorAnalysis import ColorAnalysis
from ShapeDetection import ShapeDetection
from SaliencyMap import Saliency
from CompositionAnalysis import CompositionAnalysis
from FeatureAnalysis import FeatureAnalysis
from segmentationScikit import segmScikit
from imgsaliency import ImageSaliency
from imgsaliency2 import Image_Saliency_2
#from fractevo import FractEvo

import os, os.path

import cv2
import csv



# debugging path
#path = "D:\\aaa"
#☼ original path
path = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionMaya\\images"
imageOutputPath = 'D:\\google drive\\A PhD Project at Godlsmiths\\Artist supervision code\\images'
FaceClassifierPath = ''

imagepathTemplate = "D:\\aaa\\template\\template.jpg"
imagepathTemplateCusp = "D:\\aaa\\template\\template_cuspide_up.jpg"

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
# print in the console the total number of images
print ("Program Finished. This is the tot number of images : {}".format( totalNumOfImages))

imageOutputPath = 'D:\\'

# make the calculation 
# TODO make the write images by classification scores here (not need to do afterward with the csv and Jupyter)
for i in range(totalNumOfImages):
    imagepath = imagepaths[i]
    imageName = imageNames[i]
    
    
    featureDetection = FeatureDetection(imagepath, imagepathTemplate, imagepathTemplateCusp)
    # the smaller the scalar factor the more the edges are detected
    # mean shift increase or decrease the mean of the threshold for edge detection
    # negative shift get more edges in the shadow
    # positive shift get more edges in the light
    #featureDetection.lineDetectionProbHough( scalarFactor = 1, meanShift = 0)
    #featureDetection.lineDetectionHough()
    #cnts = featureDetection.cntApproxDetection(scalarFactor = 1, meanShift = 0, method = 'edges')
    #cnts2 = featureDetection.cntApproxDetection(scalarFactor = 1, meanShift = 0, method = 'thresh')
    #featureDetection.contoursDetection()
    
    #featureDetection.convexDetection( maxThresh = 255 )
    #featureDetection.blobDetection()
    #featureDetection.tangentDetect()
    #featureDetection.vertLinesParallelCount()
    #featureDetection.sift_detector(new_image = None, image_template = None)
    #featureDetection.ORB_detector_template() NOT WORKING
    #featureDetection.cornerDetection()
    #featureDetection.fastDetection()
    #featureDetection.orbDetection()
    #6featureDetection.extremePointsDetection()
    #featureDetection.detectBlur()
    #featureDetection.edgeDetectionAuto()
    #featureDetection.detectBrightestSpot()
    
    #featureDetection.detectMultipleBrightSpot()
    #featureDetection.cuspideDetect()
    #featureDetection.findCentreOfMass(contours=cnts2)
    #featureDetection.detectCircle()
    #featureDetection.findTriangles()
    

    # TODO check all the imagepathTemplateCusp that are irrelevant after new implementation 
    shapeDetection = ShapeDetection(imagepath, imagepathTemplate, imagepathTemplateCusp)
    #shapeDetection.shapePolygonsNum()
    #shapeDetection.superPixelSegmentation()
    #shapeDetection.accessingSuperpixel()
    #shapeDetection.thresholdTest()
    #shapeDetection.drawPolyScores()
    #shapeDetection.hog_detection_segm()
    #shapeDetection.grabcut()
    
    
    colorDetection = ColorAnalysis(imagepath, imagepathTemplate)
    #colorDetection.kmeanscolor() 
    #colorDetection.colorfulnessDetection()
    #colorDetection.colorfulnessBySuperpixels()
    #colorDetection.colorQuantization()
    
    
    saliencyDetection = Saliency(imagepath)
    #saliencyDetection.get_saliency_map()
    imageSalinecy = saliencyDetection.get_proto_objects_map()
    
    #featureDetection.findCentreOfMass(image = imageSalinecy)
    
    segmentationTest = segmScikit(imagepath)
    #segmentationTest.makeSegmentation()
    
    
    sal2 = ImageSaliency(imagepath)
    #sal2.runSaliency()
    
    
# =============================================================================
#     salbis = Image_Saliency_2(imagepath)
#     #salbis.runSaliency2()
# =============================================================================
    
    
    # compositional analysis testing before apply to interface
    
    comp = CompositionAnalysis(imagepath)
    #comp.superPixelSegmentation( num_segments = 60)
    #comp.maskFromSuperpixelSegmentation(num_segments = 4)
    #comp.ruleOfThirdDistance(segmentation = 'inner', minArea = True, numberOfCnts = 50, areascalefactor = 5000, distanceMethod = 'segment')
    #comp.ruleOfThirdDistance(segmentation = 'inner', minArea = True, numberOfCnts = 3, areascalefactor = 10000, distanceMethod = 'lines')
    #comp.VisualBalanceForeground(numberOfCnts = 40, areascalefactor = 3000, segmentation = 'inner')
    #comp.diagonalsDistance()
    #comp.triangleAreaGoldenRatio()
    #comp._fourTriangleGuidelines()
    
    #comp.fourTriangleDistance(segmentation = 'ORB', edged = True, edgesdilateOpen = True)
    #comp.bigTriangleComposition()
    #comp.bigTriangleCompositionAdapted(segmentation = 'inner', minArea = True, numberOfCnts = 10, areascalefactor = 2000, distanceMethod = 'segment')
    #comp.fourTriangleCompositionAdapted(segmentation = 'inner', minArea = True, numberOfCnts = 50, areascalefactor = 3000, distanceMethod = 'segment')
    #comp.diagonalDistanceBalance(segmentation = 'inner', minArea = True, numberOfCnts = 40, areascalefactor = 2000, distanceMethod = 'segment')
    #comp.numberEdgesConvexCnt()
    
    #comp.displayandScoreExtremePoints()
    
    #comp._orbSegmentationConnectedLines()
    #comp.goldenSpiralFixDetection()
    #comp._zigzagCntsArea()
    #comp.cornerDetectionVisualBalance()
    #comp._saliencySegmentation( method = cv2.RETR_EXTERNAL,  factor = 3)
    #comp._thresholdSegmentation(factor = 1.2)
    #comp.goldenProportionOnCnts( numberOfCnts = 25, method = cv2.RETR_CCOMP)
    #comp.fourierOnEdgesDisplay()
    #comp.numOfTangentandBalance()
    #comp.circleComposition()
    #comp.rectangularComposition()
    #comp.calculateRuleOfThirdOnOrb()
    #comp.HOGcompute()
    #comp._orbSegmentation(maxKeypoints = 100, edged = True, edgesdilateOpen = False, method = cv2.RETR_EXTERNAL)
    #comp.grabcutOnOrb()
    #comp._innerCntsSegmentation(numberOfCnts = 20, method = cv2.RETR_CCOMP, minArea = 5)
    # this next replace the previous class that i had put in another module and now i fused into compositional class for semplicity
    #comp.collectScoresImage()
    #comp._borderCut()
    # color related feature extraction
    #comp.calculateDiagonalAsymmetry()
    #comp.calculateHistBalance()
    #comp.calculateWarmOrColdBalance()
    
    
    
    # feature analysis for implementation in the UI
    feat = FeatureAnalysis(imagepath)
    #feat.calculateWarmOrCold()
    #feat.displayScoresImage()
    #feat.calculateAsimmetry()
    #feat.areaForegroundVsbackground()
    
    ############################### FRACTEVO MUTATOR
    
    fractevo = ForegroundAnalysis(imagepath, imageOutputPath, imageName= 'bla')
    #fractevo.analyze_area_Lorenzo()
    #fractevo.calculateAsymmetryIndex()
    #fractevo.analyze_blobs_Lorenzo()
    #fractevo.findCentreOfMass()
    #fractevo.analyse_nfeatures_FAST()
    #fractevo.analyze_holes_Lorenzo()
# =============================================================================
#     
#     frac = FractEvo(imagepath, imageOutputPath, imageName= 'bla')
#     frac.analyze_area_Lorenzo()
#     
#     
# =============================================================================
    
    


