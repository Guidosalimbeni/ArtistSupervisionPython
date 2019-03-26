# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:46:26 2018

@author: Proprietario
"""

# Main 
# run the colorfulness and display it
# display the colorfulness score on the image

from ImageAnalysis import ForegroundAnalysis
from MatchDetector import MatchDetector
import os, os.path

import csv

# Load our input image
#imagepath = 'input_image.jpg'
#pathtest = 'D:\\loadedimages\\00008.png'
# load the images
imagepaths = []
imageNames = []
#☼ original path
path = "D:\\google drive\\organic\\Machine Learning\\MLCapture256x256"
# *********** PATH WHERE I DELETED ALL NuLL SCORE GIVEN BY WILLIAM WIHT A QUESTION MARK AND THE ZEROS (cause judge were from 1 to 5)
#path = "D:\\google drive\\A PhD Project at Godlsmiths\\Artist supervision code\\MLCapture256x256"
# *************************************************************************
# debugging path

#path = 'D:\\loadedimages'

#--------------------------------------------------------------------------
imageOutputPath = 'D:\\google drive\\A PhD Project at Godlsmiths\\Artist supervision code\\images'
FaceClassifierPath = ''

valid_images = [".jpg", ".png", ".tga", ".gif"]
for f in os.listdir(path):

    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    if ext.lower() not in valid_images:
        continue
    imagepaths.append(os.path.join(path,f))
    imageNames.append(name)
# run the calculation
totalNumOfImages = len(imagepaths)    

standardDeviationWithinTheStandardDevList = []
StdPixCountInSliceRatioList = []
ratioForeVsBackgroundList = []
fractalMinkowskiList = []
standardDeviationWithinTheStandardDevThreshList = []
StdPixCountInSliceRatioThreshList = []
ssimAsymmetryList = []
distanceRatioOfMassVsImageCentreList = []
ratioSilhoutteList = []
orbVsAreaList = []
faceScoreList = []
percentageWarmList = []
numberOfHolesList = []
PerimeterRatioConvexityList = []
scoreBrightnessList = []
ColorfulnessList = []
contrastScoreList = []
meanSaturationList = []
totalCornerList = []
ruleofthirdList = []
chaosEdgesList = []
n_blobs_LorenzoList = []
totalAreaLorenzoWeigthedList = []
inflatedAreaSumList = []
numberOfHolesLorenzoList = []
numberOfFeaturesLorList = []
analyse_clear_silh_LorList = []
perimeterCntVsHolesScoreList = []
perimeterSilhoutteList = []
fractEvoScoreList = []

for i in range(totalNumOfImages):
    imagepath = imagepaths[i]
    imageName = imageNames[i]
    # foreground analysis
    imageF = ForegroundAnalysis(imagepath, imageOutputPath, imageName)
    
    # analysis of Foregroud ------ TO IMOROVE - TODO - WORK ON TRESHOLD OF THE DFT
    # fourier analysis
    standardDeviationWithinTheStandardDev, StdPixCountInSliceRatio = imageF.fourierMeanPerSlices(22.5,67.5) #still to change rotation
    standardDeviationWithinTheStandardDevList.append(standardDeviationWithinTheStandardDev)
    StdPixCountInSliceRatioList.append(StdPixCountInSliceRatio)
    # Fourier Analysis on Threshold images
    standardDeviationWithinTheStandardDevThresh, StdPixCountInSliceRatioThresh = imageF.fourierOnAdaptiveThresh(22.5,67.5) #still to change rotation
    standardDeviationWithinTheStandardDevThreshList.append(standardDeviationWithinTheStandardDev)
    StdPixCountInSliceRatioThreshList.append(StdPixCountInSliceRatio)
    # ********************************************************************** TODO ABOVE
    
    # area vs total image size ratio # NOT IN USE since other area analysis are more weighted for Mutator
    # bit this can be used for other analysis
    ratioForeVsBackground = imageF.areaForegroundVsbackground() # from 0 to 1
    ratioForeVsBackgroundList.append(ratioForeVsBackground)
    # fractal ratio
    fractlaMinkowski = imageF.fractalDimMinkowskiBoxCount(target = 1.5)
    fractalMinkowskiList.append(fractlaMinkowski)
    #asymmetry
    ssimAsymmetry = imageF.calculateAsymmetryIndex(extraWeight = True, MinAreaSilh = 5000, 
                                                   MaxAreaSilh = 21000, penalty = 2)
    ssimAsymmetryList.append(ssimAsymmetry)
    #centre of mass
    distanceRatioOfMassVsImageCentre = imageF.findCentreOfMass(distMINWeight = 0.20, 
                                                               distMAXWeight = 1)
    distanceRatioOfMassVsImageCentreList.append(distanceRatioOfMassVsImageCentre)
    # clear silhoutte
    ratioSilhoutte = imageF.clearSilhoutte(minDist = 35 , maxDist = 85)
    ratioSilhoutteList.append(ratioSilhoutte)
    # ORB detection 
    # weighted with area and set how far is from the target 
    orbVsArea = imageF.orbDetection(maxKeypoints = 5000, 
                                    LimitPointScoreRange = 2800,
                                    LimitAreaVsOrb = 0.14,
                                    penalty = 2) 
    orbVsAreaList.append(orbVsArea)
    # warm
    percentageWarm = imageF.calculateWarmOrCold( minAreaForeground = 1000,
                                                maxAreaForeground = 21000, 
                                                penalty = 2)
    percentageWarmList.append(percentageWarm)
    # number os standalone shapes as for Lorenzo reward hoels in shapes
    # weighted ***** to use
    numberOfHoles = imageF.holesInSilhoutte(targetMinAreaCnt = 20,
                                            minHolesNum = 10,
                                            minSilhoutteArea = 5000,
                                            maxNumOfHoles = 10)
    numberOfHolesList.append(numberOfHoles)
    # convexity ratio 
    PerimeterRatioConvexity  = imageF.ratioConvexity(minPerimeter = 250,
                                                     maxPerimeter = 1000,
                                                     penalty = 4)
    PerimeterRatioConvexityList.append(PerimeterRatioConvexity)
    #brightness
    scoreBrightness = imageF.calculateBrightness(minAreaForeground = 5000,
                                                maxAreaForeground = 21000, 
                                                penalty = 2,
                                                inflatedRatio = 0.11)
    scoreBrightnessList.append(scoreBrightness)
    # Colorfulness
    Colorfulness = imageF.image_colorfulness(minAreaForeground = 5000,
                                                maxAreaForeground = 21000, 
                                                penalty = 2,
                                                inflatedRatio = 0.11)
    ColorfulnessList.append(Colorfulness)
    # contrast calculated on std and maxStdDev
    contrastScore = imageF.calculateContrast(minAreaForeground = 5000,
                                                maxAreaForeground = 21000, 
                                                penalty = 2,
                                                inflatedRatio = 0.11)
    contrastScoreList.append(contrastScore)
    # saturation on mean
    meanSaturation = imageF.calculateSaturation(MINMeanSatWholeimg = 0.09, 
                                                MAXMeanSatWholeimg = 0.165, 
                                                penalty = 2,
                                                inflatedRatio = 0.11)
    meanSaturationList.append(meanSaturation)
    # total number of corners
    totalCorner = imageF.calculateNumbersOfCorners(minAreaForeground = 5000,
                                                   maxAreaForeground = 21000, 
                                                   penalty = 2,
                                                   inflatedRatio = 0.11)
    totalCornerList.append(totalCorner)
    # ruleofthird
    ruleofthird = imageF.calculateRuleOfThird()
    ruleofthirdList.append(ruleofthird)
    # chaos calculated on density of edges detected
    chaosEdges = imageF.chaosEdges(penalty = 2, 
                                   MinAreaSilh = 5000, 
                                   MaxAreaSilh = 21000)
    chaosEdgesList.append(chaosEdges)
    
    # LORENZO TRANSLATE IN PYTHON
    n_blobs_Lorenzo = imageF.analyze_blobs_Lorenzo(thresMinLorenzo = 120, 
                                                   maxBlobs = 350)
    n_blobs_LorenzoList.append(n_blobs_Lorenzo)
    # area Lorenzo *****  there is an adapted version below on inflated 
    # area This is weighted consider also the relation of cnt shapes!!
    totalAreaLorenzoWeigthed = imageF.analyze_area_Lorenzo(thresMinLorenzo=1, thresMaxLorenzo = 256,
                                                   maxArea = 21000, minArea = 5000,
                                                   penalty = 2,
                                                   minMeanAreaLitlleCnt = 10,
                                                   minLenCnts = 3)
    totalAreaLorenzoWeigthedList.append(totalAreaLorenzoWeigthed)
    # area Lorenzo Alternative TOTAL AREA SILHOUTTE NOT USED HERE IN THIS SCRIPT BUT USED INTERNALLY AS A FUNCTION
    totalAreaSilhoutte, inflatedAreaSum = imageF.analyze_area_Lorenzo_Adapted(threshForInflatedDetection = 100)
    inflatedAreaSumList.append(inflatedAreaSum)
    # Holes implememted from Lorenzo even if not included in the original lorenzo JSON file
    numberOfHolesLorenzo = imageF.analyze_holes_Lorenzo(thresMinLorenzo = 10, 
                                                        holeMinArea = 10,
                                                        maxNumOfHoles = 20)
    numberOfHolesLorenzoList.append(numberOfHolesLorenzo)
    # number of FeatureN by Lorenzo that uses FAST
    numberOfFeaturesLor = imageF.analyse_nfeatures_FAST(threshLorenzoDetector = 28,
                                  maxNum = 1000)
    numberOfFeaturesLorList.append(numberOfFeaturesLor)
    # clear silhoutte Lorenzo. Not touching border frames
    analyse_clear_silh_Lor = imageF.analyse_clear_silh_Lor()
    analyse_clear_silh_LorList.append(analyse_clear_silh_Lor)
    # perimeter of silhoutte and weighted with holes and total frame size
    
    # TO CHECK IF IT IS WORKING ***********************************************
    perimeterCntVsHolesScore = imageF.perimeterCntVsHoles()
    perimeterCntVsHolesScoreList.append(perimeterCntVsHolesScore)
    # lenght of perimenter main silhoutte
    perimeterSilhoutte = imageF.perimeterSilhoutte()
    perimeterSilhoutteList.append(perimeterSilhoutte)
    #TODO # entropy

    
    # implement from workflowy on momenti of image there is inertia??? interessante
    
    
    # face detection if 1 means no faces if less there is somenthing
    ImageD = MatchDetector(imagepath, imageOutputPath, imageName,FaceClassifierPath)
    faceScore = ImageD.faceDetection()
    faceScoreList.append(faceScore)
    
    # FRACTEVO SCORING ****************************************
    FractEvoArea = imageF.analyze_area_Lorenzo( thresMinLorenzo = 1 , thresMaxLorenzo = 256, 
                             maxArea = 65536,
                             minArea = 0,
                             penalty = 3,
                             minMeanAreaLitlleCnt = 1,
                             minLenCnts = 1,
                             FractEvoConstraint = True,
                             fractionFractEVO = 0.66)
    FractEvoAsymmetry = imageF.calculateAsymmetryIndex(extraWeight = False, MinAreaSilh = 100, 
                                                   MaxAreaSilh = 65000, penalty = 1)
    # adapted of Lorenzo increased to 100 the max blobs and put thresh to 1
    FractEvo_N_blobs = imageF.analyze_blobs_Lorenzo(thresMinLorenzo = 1, 
                                                   maxBlobs = 100,
                                                   fractionFractEVO = 0.5,
                                                   minBlobs = 0)
    FractEvo_clear_silh = imageF.analyse_clear_silh_Lor()
    FractEvoCentreOfMass = imageF.findCentreOfMass(distMINWeight = 0.20, 
                                                               distMAXWeight = 1,
                                                               fractionFractEVO = 0.5,
                                                               FractEvoConstraint = True)
    FractEvoNumOfGoodFeatures = imageF.analyse_nfeatures_FAST(threshLorenzoDetector = 28,minNum = 100,
                                  maxNum = 1000,FractEvoConstraint = True, fractionFractEVO = 0.7)
    
    FractEvoFractal = imageF.fractalDimMinkowskiBoxCount(target = 1.5)
    # Holes are the StandAlone Shapes of Lorenzo Documentation
    # reward shapes whose num of inner holes are closer to 1 in a range from 0 to 5
    FractEvoStandAloneShapes = imageF.analyze_holes_Lorenzo(thresMinLorenzo = 10, 
                                                        holeMinArea = 40,
                                                        maxNumOfHoles = 5,
                                                        minNumofHoles = 1,
                                                        FractEvoConstraint = True,
                                                        fractionFractEVO = 0.2)
    
    fractEvoScore = (FractEvoArea + FractEvoAsymmetry + FractEvo_N_blobs +
    FractEvo_clear_silh + FractEvoCentreOfMass + FractEvoNumOfGoodFeatures +
    FractEvoFractal + FractEvoStandAloneShapes)
    fractEvoScoreList.append(fractEvoScore)
    
    # end FRACtevo Scoring Lorenzo ***********************************
    
    
    del (imageF)

print ("Program Finished. This is the tot number of images : {}".format( totalNumOfImages))

# prefix N_ means not ready yet for Mutator classification !!!!!!!!!!!!!!!!
# prefis Y_ means ready to use for Mutator classification !!!!!!!!
# write the csv file
file_to_output = open('scoringMutatorAllImagesInOriginalFolder27Feb2018.csv', 'w', newline='')
csv_writer = csv.writer(file_to_output, delimiter = ',')
csv_writer.writerow (['file',
                      'N_stdWithinSlicedStds', 
                      'N_StdPixCountInSliceRatio', 
                      'N_ratioForeVsBackground',
                      'Y_fractalMinkowski',
                      'N_standardDeviationWithinTheStandardDevThresh',
                      'N_StdPixCountInSliceRatioThresh',
                      'Y_ssimAsymmetry',
                      'Y_distanceRatioOfMassVsImageCentre',
                      'Y_ratioSilhoutte',
                      'Y_orbVsArea',
                      'N_faceScore',
                      'Y_percentageWarm',
                      'Y_numberOfHolesWeighted',
                      'Y_PerimeterRatioConvexity',
                      'Y_scoreBrightness',
                      'Y_Colorfulness',
                      'Y_contrastScore',
                      'Y_meanSaturation',
                      'Y_totalCorner',
                      'Y_ruleofthird',
                      'Y_chaosEdges',
                      'Y_n_blobs_Lorenzo',
                      'Y_totalAreaLorenzoWeigthed',
                      'Y_inflatedAreaSum',
                      'Y_numberOfHolesLorenzo',
                      'Y_numberOfFeaturesLor',
                      'Y_analyse_clear_silh_Lor',
                      'Y_perimeterCntVsHolesScore',
                      'perimeterSilhoutte',
                      'fractEvoScore'
                      ])

for i in range(totalNumOfImages):
    csv_writer.writerow ([imageNames[i],
                          standardDeviationWithinTheStandardDevList[i],
                          StdPixCountInSliceRatioList[i],
                          ratioForeVsBackgroundList[i],
                          fractalMinkowskiList[i],
                          standardDeviationWithinTheStandardDevThreshList[i],
                          StdPixCountInSliceRatioThreshList[i],
                          ssimAsymmetryList[i],
                          distanceRatioOfMassVsImageCentreList[i],
                          ratioSilhoutteList[i],
                          orbVsAreaList[i],
                          faceScoreList[i],
                          percentageWarmList[i],
                          numberOfHolesList[i],
                          PerimeterRatioConvexityList[i],
                          scoreBrightnessList[i],
                          ColorfulnessList[i],
                          contrastScoreList[i],
                          meanSaturationList[i],
                          totalCornerList[i],
                          ruleofthirdList[i],
                          chaosEdgesList[i],
                          n_blobs_LorenzoList[i],
                          totalAreaLorenzoWeigthedList[i],
                          inflatedAreaSumList[i],
                          numberOfHolesLorenzoList[i],
                          numberOfFeaturesLorList[i],
                          analyse_clear_silh_LorList[i],
                          perimeterCntVsHolesScoreList[i],
                          perimeterSilhoutteList[i],
                          fractEvoScoreList[i]])
file_to_output.close()
print(' wrote first csv file')

# selected features csv file output 

file_to_output = open('FeaturesScoresW.csv', 'w', newline='')
csv_writer = csv.writer(file_to_output, delimiter = ',')
csv_writer.writerow (['file', 
                      'N_ratioForeVsBackground',
                      'Y_fractalMinkowski',
                      'Y_ssimAsymmetry',
                      'Y_distanceRatioOfMassVsImageCentre',
                      'Y_ratioSilhoutte',
                      'Y_orbVsArea',
                      'N_faceScore',
                      'Y_percentageWarm',
                      'Y_numberOfHolesWeighted',
                      'Y_PerimeterRatioConvexity',
                      'Y_scoreBrightness',
                      'Y_Colorfulness',
                      'Y_contrastScore',
                      'Y_meanSaturation',
                      'Y_totalCorner',
                      'Y_ruleofthird',
                      'Y_chaosEdges',
                      'Y_n_blobs_Lorenzo',
                      'Y_totalAreaLorenzoWeigthed',
                      'Y_inflatedAreaSum',
                      'Y_numberOfHolesLorenzo',
                      'Y_numberOfFeaturesLor',
                      'Y_analyse_clear_silh_Lor',
                      'Y_perimeterCntVsHolesScore',
                      'perimeterSilhoutte',
                      'fractEvoScore'
                      ])

for i in range(totalNumOfImages):
    csv_writer.writerow ([imageNames[i],
                          ratioForeVsBackgroundList[i],
                          fractalMinkowskiList[i],
                          ssimAsymmetryList[i],
                          distanceRatioOfMassVsImageCentreList[i],
                          ratioSilhoutteList[i],
                          orbVsAreaList[i],
                          faceScoreList[i],
                          percentageWarmList[i],
                          numberOfHolesList[i],
                          PerimeterRatioConvexityList[i],
                          scoreBrightnessList[i],
                          ColorfulnessList[i],
                          contrastScoreList[i],
                          meanSaturationList[i],
                          totalCornerList[i],
                          ruleofthirdList[i],
                          chaosEdgesList[i],
                          n_blobs_LorenzoList[i],
                          totalAreaLorenzoWeigthedList[i],
                          inflatedAreaSumList[i],
                          numberOfHolesLorenzoList[i],
                          numberOfFeaturesLorList[i],
                          analyse_clear_silh_LorList[i],
                          perimeterCntVsHolesScoreList[i],
                          perimeterSilhoutteList[i],
                          fractEvoScoreList[i]])
file_to_output.close()

print(' wrote second csv file')
# TODO rmemenber to ad write into the class to write the DFT transform along 
# the image



''' 
#code example to read csv file
data = open('example.csv', encoding = 'utf-8')
csv_data = csv.reader(data)
data_lines = list(csv_data)
#code example to write csv file
file_to_output = open('new.csv', 'w', newline='')
csv_writer = csv_writer(file_to_output, delimiter = ',')
csv_writer.writerow (['col1', 'col2'])
'''


