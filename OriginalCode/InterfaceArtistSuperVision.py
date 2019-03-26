# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:50:52 2018

@author: Proprietario
"""

# USAGE
# tkinter_test.py

# import the necessary packages
#from tkinter import *
from tkinter import Frame,Tk,Button,Label, messagebox, StringVar, Entry
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import csv
import os, os.path
import shutil

#from ColorAnalysis import ColorAnalysis
from CompositionAnalysis import CompositionAnalysis
from CompositionAnalysis import AutoScoreML
#from FeatureAnalysis import FeatureAnalysis

class InterfaceArtistSupervision():
    
    def __init__(self):
        print ('ui created')
        
        
        self.pressed = 0
        self.scoredImagesNames = []
        
        self.preRoot = Tk()
        self.preRoot.title("Select Dir")
        self.preRoot.geometry('350x200')
        
        self.btndir = Button(text="Select directory", command=self.select_dir)
        self.btndir.pack(side="left", fill="both", expand="yes", padx="10", pady="10")
        
        self.preRoot.mainloop()
        
        
    def loadOutDirWindow(self):
        
        self.outRoot = Tk()
        self.outRoot.title("Select OutDir")
        self.outRoot.geometry('350x200')
        
        
        self.btnOutdir = Button( text="Select Output Dir", command=self.select_out_dir)
        self.btnOutdir.pack(side="left", fill="both", expand="yes", padx="10", pady="10")
        
        self.outRoot.mainloop()
    
    
    def loadMainRoot(self):

        self.panelA = None
        self.panelB = None
        #self.panelC = None
        self.panelD1 = None
        self.panelD = None
        #self.panelE = None
        self.panelF = None
        #self.panelG = None
        #self.panelH = None
        self.panelI = None
        #self.panelL = None
        self.panelM = None
        self.panelN = None
        self.panelO = None
        #self.panelP = None
        self.panelQ = None
        self.panelR = None
        self.panelS = None
        self.panelT = None
        self.imageU = None
        self.imageV = None
        
        
        # initialize the window toolkit along with the two image panels
        self.root = Tk() # create the window
        self.root.title("Artist Supervision UI")
           
        self.openfile()
        
        self.frame1 = Frame(self.root)
        self.frame1.pack(side = 'top')
        self.frame2 = Frame(self.root)
        self.frame2.pack(side = 'top')
        self.frame2A = Frame(self.root)
        self.frame2A.pack(side = 'top')
        self.frame3 = Frame(self.root)
        self.frame3.pack(side = 'top')
        self.frame4 = Frame(self.root)
        self.frame4.pack(side = 'bottom')
        
        self.btn1 = Button(self.frame3, text = '1', command = lambda : self.moveRated(score = 1))
        self.btn1.pack(side="left", expand="yes", padx="10", pady="10")
        self.btn2 = Button(self.frame3, text = '2', command = lambda : self.moveRated(score = 2))
        self.btn2.pack(side="left", expand="yes", padx="10", pady="10")
        self.btn3 = Button(self.frame3, text = '3', command = lambda : self.moveRated(score = 3))
        self.btn3.pack(side="left", expand="yes", padx="10", pady="10")
        self.btn4 = Button(self.frame3, text = '4', command = lambda : self.moveRated(score = 4))
        self.btn4.pack(side="left", expand="yes", padx="10", pady="10")
        self.btn5 = Button(self.frame3, text = '5', command = lambda : self.moveRated(score = 5))
        self.btn5.pack(side="left", expand="yes", padx="10", pady="10")
        self.btn6 = Button(self.frame3, text = '6', command = lambda : self.moveRated(score = 6))
        self.btn6.pack(side="left", expand="yes", padx="10", pady="10")
        self.btn7 = Button(self.frame3, text = '7', command = lambda : self.moveRated(score = 7))
        self.btn7.pack(side="left", expand="yes", padx="10", pady="10")
        self.btn8 = Button(self.frame3, text = '8', command = lambda : self.moveRated(score = 8))
        self.btn8.pack(side="left", expand="yes", padx="10", pady="10")
        self.btn9 = Button(self.frame3, text = '9', command = lambda : self.moveRated(score = 9))
        self.btn9.pack(side="left", expand="yes", padx="10", pady="10")
        self.btn10 = Button(self.frame3, text = '10', command = lambda : self.moveRated(score = 10))
        self.btn10.pack(side="left", expand="yes", padx="10", pady="10")
        
        

        # auto score button
        self.autoML = StringVar()
        self.autoML.set('Do you want an hint?')
        self.autoMLLabel = Label(self.frame4, textvariable = self.autoML)
        self.autoMLLabel.pack(side="right", expand="yes", padx="10", pady="10")
        
        self.btnMACHINELEARNING = Button(self.frame4, text = 'ML auto Score', command = self.autoscoreDnnClassifier)
        self.btnMACHINELEARNING.pack(side="right", expand="yes", padx="10", pady="10")
        
        # force to close the csv file if not finished the database
        self.btnClose = Button(self.frame4, text = 'close csv', command = self.closeScoreFile)
        self.btnClose.pack(side="right", expand="yes", padx="10", pady="10")
        
        self.btnNext = Button(self.frame4, text = 'next Image', command = self.next_image)
        self.btnNext.pack(side="right", expand="yes", padx="10", pady="10")
        
        self.compositionUserJudge = StringVar()
        self.entryComp = Entry(self.frame4, textvariable = self.compositionUserJudge)
        self.entryComp.pack(side="left", expand="yes", padx="10", pady="10")
        
        self.scoredYesNo = StringVar()
        self.scoredYesNo.set('NOT scored Yet')
        self.scoredYesNoLabel = Label(self.frame4, textvariable = self.scoredYesNo)
        self.scoredYesNoLabel.pack(side="right", expand="yes", padx="10", pady="10")
        

        
        self.next_image()
        
        # kick off the GUI
        self.root.mainloop()
        
    def select_dir(self):
        
        self.directory = filedialog.askdirectory()
        self.loadImagesInList()
        self.preRoot.destroy()
        self.loadOutDirWindow()
        

    def loadImagesInList(self):
        
        # load and reload after moving and deleting rated images
        
        self.imagepaths = []
        self.imageNames = []
        self.imagefullNames = []
        
        valid_images = [".jpg", ".png", ".tga", ".gif"]
        for f in os.listdir(self.directory):
        
            ext = os.path.splitext(f)[1]
            name = os.path.splitext(f)[0]
            full_name = name + ext
            if ext.lower() not in valid_images:
                continue
            self.imagepaths.append(os.path.join(self.directory,f))
            self.imageNames.append(name)
            self.imagefullNames.append(full_name)
        # run the calculation
        print('total number of images: ', len(self.imagepaths)  )     
        
    def select_out_dir(self):
        
        self.output_path = filedialog.askdirectory()
        self.outRoot.destroy()
        self.loadMainRoot()
        

    def next_image(self):
        
        # reload the image in the folder in order to get updated if inserted new images to the dataset at runtime
        self.loadImagesInList()
        # reset the autoML to question:
        self.autoML.set('Do you want an hint?')
        # open a file chooser dialog and allow the user to select an input
        # image

        if (self.pressed ) == len(self.imagepaths):
            
            self.pressed = 0
  
        self.path = self.imagepaths[self.pressed]
        # ensure the next is working and updating the index
        self.pressed += 1
        
        self.name = self.imageNames[self.pressed -1]
        if self.name in self.scoredImagesNames:
            self.scoredYesNo.set('Already Scored')
            #print (self.name)
            #print (self.scoredImagesNames)
        else: 
            self.scoredYesNo.set('NOT Scored')
        
        # ensure a file path was selected
        if len(self.path) > 0:

            #Panel A - Original
            image = cv2.imread(self.path)
             
            #Panel B
            # Visual Balance Foreground
            comp = CompositionAnalysis(imagepath = self.path)
            CenterOfMassDisplayImage, self.scoreVisBalance, self.slopeVisualBalance = comp.VisualBalanceForeground(numberOfCnts = 50, kernel = 3,areascalefactor = 10000, segmentation = 'inner')
            # Panel C 
            # edged display
            #edgedDisplay = comp._edgeDetection( scalarFactor = 1, meanShift = 0, edgesdilateOpen = False)
# =============================================================================
#             rectCompImg, self.rectCompScore = comp.rectangularComposition()
#             
# =============================================================================
            # Panel D 1
            # Rule of Third (from paper 41) - segment on inner contours and distance to segments method
            imageDisplayRuleOfThirdThresh, self.ScoreRuleOfThirdThresh = comp.ruleOfThirdDistance(segmentation = 'inner', 
                                                                                                  minArea = True, numberOfCnts = 50, areascalefactor = 10000, distanceMethod = 'segment')
            # Panel D 
            # Big Triangle Adapted inner contours and segment method
            imageDisplayBigTriangleScore, self.ScoreBigTriangle = comp.bigTriangleCompositionAdapted(segmentation = 'inner', 
                                                                                                minArea = True, numberOfCnts = 50, areascalefactor = 10000, distanceMethod = 'segment')
            # Panel F 
            # Diagonal this replace the old version and work for balance in consideration of dutch painter
            diagonal, self.diagonalasymmetryBalance = comp.diagonalDistanceBalance(segmentation = 'inner', minArea = True, numberOfCnts = 50, areascalefactor = 10000, distanceMethod = 'segment')
            # Panel G +  h
            # IF INSERTED AGAIN NEED TO ADD THE SCORES TO CSV THAT I DELETED
# =============================================================================
#             # Triangle Composition (from paper 41) orb segmentation method
#             triangleDisplay, blankOrb, self.scoreGoldenTriangle = comp.triangleAreaGoldenRatio(segmentation = 'ORB', minArea = False, 
#                              numberOfCnts = 10, areascalefactor = 3000)
# =============================================================================
            #Panel H circel comp
# =============================================================================
#             CircleCompImg, self.circleCompScore = comp.circleComposition()
#             
# =============================================================================
            # Panel I
            displayImgTriaDist, self.scoreFourTriangleDistance= comp.fourTriangleCompositionAdapted(segmentation = 'inner', 
                                                                                                  minArea = True, numberOfCnts = 50, areascalefactor = 10000, 
                                                                                                  distanceMethod = 'segment')
            
            # Panel L composition of a big main triangle (solidity base of triangle low)
# =============================================================================
#             displayImgBigTriangleComp, self.bigTriangleCompScore = comp.bigTriangleComposition(segmentation = 'ORB')
# =============================================================================
            # Panel M number of edges of convex hull
            HullimgCopy, self.scoreHullBalance, self.firstPointLeftY,self.secondtPointLeftY,self.thirdPointLeftY,self.firstPointRightY,self.secondtPointRightY,self.thirdPointRightY = comp.numberEdgesConvexCnt ( minArea = True, 
                             numberOfCnts = 8, areascalefactor = 1000 )
            # Panel N tangents detector and display
            edgesAndTangentImg , self.scoreNumTangents = comp.numOfTangentandBalance ()
            # Panel O extreme points and score distances from half sides
            extremePointsImg, self.DistExtLeftToHalf,self.DistExtRightToHalf,self.DistExtTopToHalf, self.DistExtBotToHalf, self.DistLeftBorder, self.DistRightBorder, self.DistTopBorder, self.DistBotBorder= comp.displayandScoreExtremePoints()
            # Panel P ZigZag and golden rectangle
# =============================================================================
#             copyZigZag, self.ratioGoldenRectangleZigZagOrb, sorted_contours, self.zigzagPerimetScore = comp._zigzagCntsArea(drawLabel = True)
# =============================================================================
            # Panel Q holds the golden ratio spiral display and number of keypoints toucing the spiral as a ratio 
            goldenImgDisplay, self.scoreSpiralGoldenRatio = comp.goldenSpiralFixDetection ( displayall = False, displayKeypoints = True, maxKeypoints = 100, edged = True)
            # panel R stores the corners detection and visual balance on horiz and vertical considering the mid line the mid of the orb detections
            cornerimg, self.scoreHorizzontalCorners, self.scoreVerticalCorners = comp.cornerDetectionVisualBalance (maxCorners = 40 , minDistance = 6, midlineOnCornersCnt =  True)
            # Panel S for the balance between vertical and horizontal lines along the ROI
            vertHorizontalLinesBalanceImg, self.verticalLinesBalanceScore, self.horizontalLinesBalanceScore = comp.vertAndHorizLinesBalance()
            # Panel T divine proportion of all inner contours previous to next difference from 1.618
            goldenPropImg, self.scoreProportionAreaVsGoldenRatio= comp.goldenProportionOnCnts(numberOfCnts = 25, method = cv2.RETR_CCOMP, minArea = 2)
            # Panel U for dft
            fourierImg, self.dftDiffScoreLfRt= comp.fourierOnEdgesDisplay()
            # Panel E 
            # blackboard extra features analysis
            #blackboard = FeatureAnalysis(self.path )
            #blackboard, self.ssimAsymmetry, self.fractalScoreFromTarget, self.ratioForeVsBackground = comp.collectScoresImage()
            self.ssimAsymmetry, self.fractalScoreFromTarget, self.ratioForeVsBackground, self.diagonalAsymmetry, self.histHueCorrelationBalance, self.warmColorBalance = comp.collectScoresImage()
            #### onlyscore no images to display
            self.ratioRuleOfThirdPixels = comp.ruleOfThirdOnThreshPixelsCount()

            ### Panle V run the scoring synthesis for displaying purpose only
            scoringSynthesiImg = comp.synthesisScores()
            
            
            # CONVERSIONS *********************************************************************
            image = self.conversionToDisplayInTkinter(image)
            #edgedDisplay = cv2.cvtColor(edgedDisplay, cv2.COLOR_GRAY2BGR)
            #edgedDisplay =self.conversionToDisplayInTkinter(edgedDisplay)
            CenterOfMassDisplayImage = self.conversionToDisplayInTkinter(CenterOfMassDisplayImage)
            imageDisplayRuleOfThirdThresh = self.conversionToDisplayInTkinter(imageDisplayRuleOfThirdThresh)
            imageDisplayBigTriangleScore = self.conversionToDisplayInTkinter(imageDisplayBigTriangleScore)
            diagonal = self.conversionToDisplayInTkinter(diagonal)
            #blackboard = self.conversionToDisplayInTkinter(blackboard)
            #blankOrb = self.conversionToDisplayInTkinter(blankOrb) # not used
            #triangleDisplay = self.conversionToDisplayInTkinter(triangleDisplay)
            displayImgTriaDist = self.conversionToDisplayInTkinter(displayImgTriaDist)
            #displayImgBigTriangleComp = self.conversionToDisplayInTkinter(displayImgBigTriangleComp)
            HullimgCopy = self.conversionToDisplayInTkinter(HullimgCopy)
            edgesAndTangentImg = self.conversionToDisplayInTkinter(edgesAndTangentImg)
            extremePointsImg = self.conversionToDisplayInTkinter(extremePointsImg)
            #copyZigZag = self.conversionToDisplayInTkinter(copyZigZag)
            goldenImgDisplay = self.conversionToDisplayInTkinter(goldenImgDisplay)
            cornerimg = self.conversionToDisplayInTkinter(cornerimg)
            vertHorizontalLinesBalanceImg = self.conversionToDisplayInTkinter(vertHorizontalLinesBalanceImg)
            goldenPropImg = self.conversionToDisplayInTkinter(goldenPropImg)
            fourierImg = self.conversionToDisplayInTkinter(fourierImg)
            #CircleCompImg = self.conversionToDisplayInTkinter(CircleCompImg)
            scoringSynthesiImg = self.conversionToDisplayInTkinter(scoringSynthesiImg)
            #rectCompImg = self.conversionToDisplayInTkinter(rectCompImg)
            
            # if the panels are None, initialize them
            if self.panelA is None or self.panelB is None:
                # the first panel will store our original image
                self.panelA = Label(master = self.frame1, image=image)
                self.panelA.image = image
                self.panelA.pack(side="left", padx=5, pady=10)
                
                # while this panel will store the rule of third threshold method
                self.panelD1 = Label(master = self.frame1, image=imageDisplayRuleOfThirdThresh)
                self.panelD1.image = imageDisplayRuleOfThirdThresh
                self.panelD1.pack(side="right", padx=5, pady=10)
                # while this panel will store the rule of third saliency map segmentation method
                self.panelD = Label(master = self.frame1, image=imageDisplayBigTriangleScore)
                self.panelD.image = imageDisplayBigTriangleScore
                self.panelD.pack(side="right", padx=5, pady=10)

                # while this panel will store the triangle comp orb map segmentation method
# =============================================================================
#                 self.panelG = Label(master = self.frame1, image=triangleDisplay)
#                 self.panelG.image = triangleDisplay
#                 self.panelG.pack(side="right", padx=5, pady=10)
# =============================================================================
                # hull convexity number of edges
                self.panelM = Label(master = self.frame1, image=HullimgCopy)
                self.panelM.image = HullimgCopy
                self.panelM.pack(side="right", padx=5, pady=10)
                # panel o this will store the extreme points and distance from half sides
                self.panelO = Label(master = self.frame1, image=extremePointsImg)
                self.panelO.image = extremePointsImg
                self.panelO.pack(side="right", padx=5, pady=10)
                
                #zig Zag and golden rectangle
# =============================================================================
#                 self.panelP = Label(master = self.frame2,image=copyZigZag)
#                 self.panelP.image = copyZigZag
#                 self.panelP.pack(side="right", padx=5, pady=10)
# =============================================================================
                # while this panel will store the rule of third saliency map segmentation method
                self.panelF = Label(master = self.frame2, image=diagonal)
                self.panelF.image = diagonal
                self.panelF.pack(side="right", padx=5, pady=10)
                # golden spiral and number of keypoints on spiral (loop over 4 spiral)
                self.panelQ = Label(master = self.frame2, image=goldenImgDisplay)
                self.panelQ.image = goldenImgDisplay
                self.panelQ.pack(side="right", padx=5, pady=10)
                # while the second panel will the visual Balance Hoizzontal
                self.panelB = Label(master = self.frame2,image=CenterOfMassDisplayImage)
                self.panelB.image = CenterOfMassDisplayImage
                self.panelB.pack(side="right", padx=5, pady=10)
                # while this panel will store the tangents detector
                self.panelN = Label(master = self.frame2, image=edgesAndTangentImg)
                self.panelN.image = edgesAndTangentImg
                self.panelN.pack(side="right", padx=5, pady=10)
                # while this panel will store the triangle comp orb map segmentation method
                self.panelI = Label(master = self.frame2, image=displayImgTriaDist)
                self.panelI.image = displayImgTriaDist
                self.panelI.pack(side="right", padx=5, pady=10)
                
                # panel for big triangle
# =============================================================================
#                 self.panelL = Label(master = self.frame2, image=displayImgBigTriangleComp)
#                 self.panelL.image = displayImgBigTriangleComp
#                 self.panelL.pack(side="right", padx=5, pady=10)
# =============================================================================
            
                # this displays the edges
# =============================================================================
#                 self.panelC = Label(master = self.frame2, image=rectCompImg)
#                 self.panelC.image = rectCompImg
#                 self.panelC.pack(side="left", padx=5, pady=10)
# =============================================================================
                #TODO create a separate function so that each subfunction call the same segmentation parameters for segmentation
                # orb detection from triangle comp
# =============================================================================
#                 self.panelH = Label(master = self.frame2, image=CircleCompImg)
#                 self.panelH.image = CircleCompImg
#                 self.panelH.pack(side="left", padx=5, pady=10)
# =============================================================================
                # panel U fourier
                self.panelU = Label(master = self.frame2A, image=fourierImg)
                self.panelU.image = fourierImg
                self.panelU.pack(side="right", padx=5, pady=10)
                # panel T golden proportion of all shapes
                self.panelT = Label(master = self.frame2A, image=goldenPropImg)
                self.panelT.image = goldenPropImg
                self.panelT.pack(side="right", padx=5, pady=10)
                # panel S vertical and horizontal lines balance along the mid of ROI
                self.panelS = Label(master = self.frame2A, image=vertHorizontalLinesBalanceImg)
                self.panelS.image = vertHorizontalLinesBalanceImg
                self.panelS.pack(side="right", padx=5, pady=10)
                # panel R corners and visual balance horizz and vertical with mid line in the mid of area of orbs detections
                self.panelR = Label(master = self.frame2A, image=cornerimg)
                self.panelR.image = cornerimg
                self.panelR.pack(side="right", padx=5, pady=10)

# =============================================================================
#                 # while this panel will store all theextra features
#                 self.panelE = Label(master = self.frame2A, image=blackboard)
#                 self.panelE.image = blackboard
#                 self.panelE.pack(side="right", padx=5, pady=10)
# =============================================================================
                # scoring display
                self.panelV = Label(master = self.frame2A, image=scoringSynthesiImg)
                self.panelV.image = scoringSynthesiImg
                self.panelV.pack(side="left", padx=5, pady=10)
            
            else:
                self.panelA.configure(image=image)
                self.panelB.configure(image=CenterOfMassDisplayImage)
                #self.panelC.configure(image=rectCompImg)
                self.panelD1.configure(image=imageDisplayRuleOfThirdThresh)
                self.panelD.configure(image=imageDisplayBigTriangleScore)
                #self.panelE.configure(image=blackboard)
                self.panelF.configure(image=diagonal)
                #self.panelG.configure(image=triangleDisplay)
                #self.panelH.configure(image=CircleCompImg)
                self.panelI.configure(image = displayImgTriaDist)
                #self.panelL.configure(image = displayImgBigTriangleComp)
                self.panelM.configure(image = HullimgCopy)
                self.panelN.configure(image = edgesAndTangentImg)
                self.panelO.configure(image = extremePointsImg)
                #self.panelP.configure(image = copyZigZag)
                self.panelQ.configure(image = goldenImgDisplay)
                self.panelR.configure(image = cornerimg)
                self.panelS.configure(image = vertHorizontalLinesBalanceImg)
                self.panelT.configure(image = goldenPropImg)
                self.panelU.configure(image = fourierImg)
                self.panelV.configure(image = scoringSynthesiImg)
                self.panelA.image = image
                self.panelB.image = CenterOfMassDisplayImage
                #self.panelC.image = rectCompImg
                self.panelD1.image = imageDisplayRuleOfThirdThresh
                self.panelD.image = imageDisplayBigTriangleScore
                #self.panelE.image = blackboard
                self.panelF.image = diagonal
                #self.panelG.image = triangleDisplay
                #self.panelH.image = CircleCompImg
                self.panelI.image = displayImgTriaDist
                #self.panelL.image = displayImgBigTriangleComp
                self.panelM.image = HullimgCopy
                self.panelN.image = edgesAndTangentImg
                self.panelO.image = extremePointsImg
                #self.panelP.image = copyZigZag
                self.panelQ.image = goldenImgDisplay
                self.panelR.image = cornerimg
                self.panelS.image = vertHorizontalLinesBalanceImg
                self.panelT.image = goldenPropImg
                self.panelU.image = fourierImg
                self.panelV.image = scoringSynthesiImg
            
        
    def openfile (self):
        
        
        
        self.file_to_output = open('scoringApril30_B.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.file_to_output, delimiter = ',')
        self.csv_writer.writerow (['file',
                                   'judge',
                                   'CompositionUserChoice',
                                   'scoreVisBalance',
                                   'slopeVisualBalance',
                                   'ScoreRuleOfThirdThresh',
                                   'ScoreBigTriangle',
                                   'diagonalasymmetryBalance',
                                   'scoreFourTriangleDistance',
                                   'scoreHullBalance',
                                   'scoreNumTangents',
                                   'scoreSpiralGoldenRatio',
                                   'scoreHorizzontalCorners',
                                   'scoreVerticalCorners',
                                   'verticalLinesBalanceScore',
                                   'horizontalLinesBalanceScore',
                                   'scoreProportionAreaVsGoldenRatio',
                                   'dftDiffScoreLfRt',
                                   'ssimAsymmetry',
                                   'diagonalAsymmetry',
                                   'histHueCorrelationBalance',
                                   'warmColorBalance',
                                   'ratioForeVsBackground',
                                   'fractalScoreFromTarget',
                                   'ratioRuleOfThirdPixels',
                                   'DistLeftBorder', 
                                   'DistRightBorder', 
                                   'DistTopBorder', 
                                   'DistBotBorder',
                                   'DistExtLeftToHalf','DistExtRightToHalf','DistExtTopToHalf', 'DistExtBotToHalf',
                                   'firstPointLeftY','secondtPointLeftY','thirdPointLeftY','firstPointRightY','secondtPointRightY','thirdPointRightY'
                                   ])

    
    def writeScore(self):
        
        CompositionUserChoice = self.compositionUserJudge.get()
        if CompositionUserChoice == '':
            CompositionUserChoice = 'other'

        self.csv_writer.writerow ([self.name,
                                   self.score,
                                   CompositionUserChoice,
                                   self.scoreVisBalance,
                                   self.slopeVisualBalance,
                                   self.ScoreRuleOfThirdThresh,
                                   self.ScoreBigTriangle,
                                   self.diagonalasymmetryBalance,
                                   self.scoreFourTriangleDistance,
                                   self.scoreHullBalance,
                                   self.scoreNumTangents,
                                   self.scoreSpiralGoldenRatio,
                                   self.scoreHorizzontalCorners,
                                   self.scoreVerticalCorners,
                                   self.verticalLinesBalanceScore,
                                   self.horizontalLinesBalanceScore,
                                   self.scoreProportionAreaVsGoldenRatio,
                                   self.dftDiffScoreLfRt,
                                   self.ssimAsymmetry,
                                   self.diagonalAsymmetry,
                                   self.histHueCorrelationBalance,
                                   self.warmColorBalance,
                                   self.ratioForeVsBackground,
                                   self.fractalScoreFromTarget,
                                   self.ratioRuleOfThirdPixels,
                                   self.DistLeftBorder, 
                                   self.DistRightBorder, 
                                   self.DistTopBorder, 
                                   self.DistBotBorder,
                                   self.DistExtLeftToHalf,self.DistExtRightToHalf,self.DistExtTopToHalf, self.DistExtBotToHalf,
                                   self.firstPointLeftY,self.secondtPointLeftY,self.thirdPointLeftY,self.firstPointRightY,self.secondtPointRightY,self.thirdPointRightY
                                   ])
        
        self.entryComp.delete(0)
        
        if len(self.scoredImagesNames) == len(self.imagepaths):
            messagebox.showinfo("COMPLETED Scoring", "Well Done! I will close csv and quit")
            self.closeScoreFile()
            self.root.destroy()
            
        else: 
            self.next_image()
        
    def closeScoreFile(self):
        
        self.file_to_output.close()
        
        print(' wrote first csv file')
    
    def autoscoreDnnClassifier (self):
        
        # create the dataframe of features extracted
        extractedFeatures = [(self.scoreVisBalance,
                               self.slopeVisualBalance,
                               self.ScoreRuleOfThirdThresh,
                               self.ScoreBigTriangle,
                               self.diagonalasymmetryBalance,
                               self.scoreFourTriangleDistance,
                               self.scoreHullBalance,
                               self.scoreNumTangents,
                               self.scoreSpiralGoldenRatio,
                               self.scoreHorizzontalCorners,
                               self.scoreVerticalCorners,
                               self.verticalLinesBalanceScore,
                               self.horizontalLinesBalanceScore,
                               self.scoreProportionAreaVsGoldenRatio,
                               self.dftDiffScoreLfRt,
                               self.ssimAsymmetry,
                               self.diagonalAsymmetry,
                               self.histHueCorrelationBalance,
                               self.warmColorBalance,
                               self.ratioForeVsBackground,
                               self.fractalScoreFromTarget,
                               self.ratioRuleOfThirdPixels,
                               self.DistLeftBorder, 
                               self.DistRightBorder, 
                               self.DistTopBorder, 
                               self.DistBotBorder,
                               self.DistExtLeftToHalf,self.DistExtRightToHalf,self.DistExtTopToHalf, self.DistExtBotToHalf,
                               self.firstPointLeftY,self.secondtPointLeftY,self.thirdPointLeftY,self.firstPointRightY,self.secondtPointRightY,self.thirdPointRightY
                               )]
        
        # call the model with the features extracted and collect the prediction
        ml = AutoScoreML(extractedFeatures)
        res = ml.autoScoreML()
        
        # encode the prediction into bad, good, very good
        if res == 0:
            self.autoML.set( 'bad')
        if res == 1:
            self.autoML.set ( 'quite good')
        if res == 2:
            self.autoML.set ('very good!')

            
    def conversionToDisplayInTkinter(self, image):
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        return image
            
    def moveRated(self, score):
        # wrote the score into the csv file and move the image
        # to avoid double rating 
        # create one folder for copy and one folder for sorted 01_filename.jpg etc
        if score == 1:
            self.score = 1
            self.CopyRenameWriteScore(score = 1)   
        elif score == 2:
            self.score = 2
            self.CopyRenameWriteScore(score = 2)  
        elif score == 3:
            self.score = 3
            self.CopyRenameWriteScore(score = 3)
        elif score == 4:
            self.score = 4
            self.CopyRenameWriteScore(score = 4)
        elif score == 5:
            self.score = 5
            self.CopyRenameWriteScore(score = 5)
        elif score == 6:
            self.score = 6
            self.CopyRenameWriteScore(score = 6)
        elif score == 7:
            self.score = 7
            self.CopyRenameWriteScore(score = 7)
        elif score == 8:
            self.score = 8
            self.CopyRenameWriteScore(score = 8)
        elif score == 9:
            self.score = 9
            self.CopyRenameWriteScore(score = 9)
        elif score == 10:
            self.score = 10
            self.CopyRenameWriteScore(score = 10)
        
    
    def CopyRenameWriteScore (self, score):
        
        self.name = self.imageNames[self.pressed -1]
        
        if self.name in self.scoredImagesNames:
            
            messagebox.showinfo("Scored", "alredy rated I move next")
            self.next_image()
            return
        else:

            self.scoredImagesNames.append(self.name)
        
        score = str(score)
        strScore = score.zfill(2)
        # the pressed index is still not updated by the pressing of button
        self.copy_rename(old_file_name = self.imagefullNames[self.pressed - 1],
               new_file_name = strScore + '_' + self.imagefullNames[self.pressed - 1],
               scr_file_path = self.imagepaths[self.pressed - 1],
               output_path = self.output_path)
        self.writeScore()
            
    
    def copy_rename(self, old_file_name, new_file_name, scr_file_path, output_path):
        
        dst_dir = output_path
        shutil.copy(scr_file_path, dst_dir)
        
        dst_file = os.path.join(output_path, old_file_name)
        new_dst_file_name = os.path.join (dst_dir, new_file_name)
        
        os.rename(dst_file, new_dst_file_name)
            

# TODO considering creating a dictionary instead csv and write a json file 
# TODO this way it could allow to overwrite score and alert if a file is already
        # rated instead of moving file around.
        # also it will allow to pen a previous jason file into a dictionary and
        # continue from there



UI = InterfaceArtistSupervision()










