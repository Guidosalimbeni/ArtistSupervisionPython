# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:43:30 2018

@author: OWNER
"""

import numpy as np
import cv2
import os


path = "D:\\google drive\\A PhD Project at Godlsmiths\\MutatorML\\MLCaptureG"

start = 0
number_of_images_to_edge = 500

# =============================================================================
# outputPathA = "D:\\google drive\\A PhD Project at Godlsmiths\\MutatorML\\pix2pixImages\\A\\"
# outputPathB = "D:\\google drive\\A PhD Project at Godlsmiths\\MutatorML\\pix2pixImages\\B\\"
# outputPathA_B = "D:\\google drive\\A PhD Project at Godlsmiths\\MutatorML\\pix2pixImages\\A_B\\train\\"
# =============================================================================
# =============================================================================
# maskpath = "D:\\google drive\\A PhD Project at Godlsmiths\\MutatorML\\MLCaptureGMask"
# =============================================================================

outputPathA_B = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionMaya\\images\\pix2pix\\A_B\\train\\"

pathA = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionMaya\\images\\pix2pix\\A\\"
pathB = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionMaya\\images\\pix2pix\\B\\"

def imagelist (path):
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
    

    
    return imagepaths, imageNames, totalNumOfImages

imageMasks = []
valid_images_mask = [".jpg", ".png", ".tga", ".gif"]
# =============================================================================
# imagepaths = []
# imageNames = []
# valid_images = [".jpg", ".png", ".tga", ".gif"]
# for f in os.listdir(path):
# 
#     ext = os.path.splitext(f)[1]
#     name = os.path.splitext(f)[0]
#     if ext.lower() not in valid_images:
#         continue
#     imagepaths.append(os.path.join(path,f))
#     imageNames.append(name)
# # store the total number of images
# totalNumOfImages = len(imagepaths) 
# 
# imageMasks = []
# valid_images_mask = [".jpg", ".png", ".tga", ".gif"]
# =============================================================================
# =============================================================================
# for f in os.listdir(maskpath):
# 
#     ext = os.path.splitext(f)[1]
#     name = os.path.splitext(f)[0]
#     if ext.lower() not in valid_images_mask:
#         continue
#     imageMasks.append(os.path.join(maskpath,f))
# =============================================================================


def image2edges(image):
    
    gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    
        # threshold
    ret,ThresPerim = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    threshCopy = ThresPerim.copy()
    ing2, contours, hierarchy = cv2.findContours(threshCopy,
                                                 cv2.RETR_LIST,
                                                 cv2.CHAIN_APPROX_SIMPLE)
    
    edged = np.zeros((image.shape[0], image.shape[1], 3), dtype = 'uint8')
    
    edged[:,:] = 255
    
    for cnt in contours:
        cv2.drawContours(edged, [cnt], -1, (0, 0, 0), 1)
    
    return edged

def background2white(image, maskImg):
    
    ret,thresh = cv2.threshold(maskImg, 10, 255, cv2.THRESH_BINARY)
    thresh[np.where((image>[0,0,0]).all(axis=2))] = [255,255,255]
    
    #image[np.where((image==[0,0,0]).all(axis=2))] = [255,255,255]
    image[np.where((thresh==[0,0,0]).all(axis=2))] = [255,255,255]
    
    return image


def a_and_b (image, edged):
    
    result = np.zeros((image.shape[0] , image.shape[1] * 2, 3), dtype = 'uint8')
    
    result = np.concatenate([edged, image], 1)
    
    return result
    

# =============================================================================
# for imagepath, name, mask in zip(imagepaths[start:number_of_images_to_edge], imageNames[start:number_of_images_to_edge], imageMasks[start:number_of_images_to_edge]):
#     
#     image = cv2.imread(imagepath)
#     maskImg = cv2.imread(mask)
#     edged = image2edges(image)
#     image = background2white(image, maskImg)
#     a_b = a_and_b (image, edged)
#     
#     cv2.imwrite(outputPathA + name + ".png", edged)
#     cv2.imwrite(outputPathB + name + ".png", image)
#     cv2.imwrite(outputPathA_B + name + ".png", a_b)
# =============================================================================
    
imagepathsA, imageNamesA, totalNumOfImages = imagelist (pathA)
imagepathsB, imageNamesB, totalNumOfImages = imagelist (pathB)
for imagepathA, nameA, imagepathB in zip(imagepathsA[start:number_of_images_to_edge], imageNamesA[start:number_of_images_to_edge], imagepathsB[start:number_of_images_to_edge]):
    
    imageA = cv2.imread(imagepathA)
    imageB = cv2.imread(imagepathB)

    a_b = a_and_b (imageB, imageA)
    

    cv2.imwrite(outputPathA_B + nameA + ".png", a_b)

# =============================================================================
# for imagepath, name in zip(imagepaths[start:number_of_images_to_edge], imageNames[start:number_of_images_to_edge]):
#     
#     image = cv2.imread(imagepath)
#     
#     edged = image2edges(image)
#     image = background2white(image, maskImg)
#     a_b = a_and_b (image, edged)
#     
#     cv2.imwrite(outputPathA + name + ".png", edged)
#     cv2.imwrite(outputPathB + name + ".png", image)
#     cv2.imwrite(outputPathA_B + name + ".png", a_b)
# =============================================================================
        