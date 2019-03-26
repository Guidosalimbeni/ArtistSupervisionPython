
"""
Created on Thu Mar  1 13:29:24 2018

Main in case no using the UI interface
this main is more used for testing at the moment

@author: Guido Salimbeni
"""




from CompositionAnalysis import CompositionAnalysis

import os, os.path

import cv2

path = "D:\\aaa"

imageOutputPath = 'D:\\google drive\\A PhD Project at Godlsmiths\\Artist supervision code\\images'


# =============================================================================
# parser = argparse.ArgumentParser()
# parser.add_argument('-f', '--filepath')
# parser.add_argument('-b', '--bar_value')
# args = parser.parse_args()
# 
# print (args.my_foo)
# print (args.bar_value)
# =============================================================================


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


for i in range(totalNumOfImages):
    imagepath = imagepaths[i]
    imageName = imageNames[i]
    
 
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
    maskForeground, scoreVisualBalance, SlopeVisualBalance = comp.VisualBalanceForeground()
    
    
    cv2.imshow('segm', maskForeground)
    cv2.waitKey()
    cv2.destroyAllWindows()


