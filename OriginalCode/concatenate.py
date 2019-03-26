# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 09:30:06 2018

@author: OWNER
"""
import cv2
import numpy as np
import os

patha = "C:\\Users\\OWNER\\Downloads\\New folder (2)\\morandi\\"

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


imagepaths, imageNames, totalNumOfImages = imagelist (patha)

print (totalNumOfImages)

# 256 x 256

#blank = np.zeros((1536,1536), dtype = np.uint8)

# =============================================================================
# newnames = []
# for name in imageNames:
#     names = name.split('_')
#     a = names[0]
#     b = a.zfill(3)
#     new = b + '-'+ name
#     newnames.append(new)
# 
# newnames = sorted(newnames)
# splitted = []
# for n in newnames:
#      a = n.split('-')
#      splitted.append(a)
# 
# newnames = []
# for n in splitted:
#      a = n[1]
#      newnames.append(a)
# 
# print (path + newnames[0] + '.png')
# =============================================================================

newnames = sorted(imagepaths)
path =''
for i, path in enumerate(newnames[0:24]):
    
    
    
    image = cv2.imread(path + newnames[i] )
    image1 = cv2.imread(path + newnames[i + 1])
    image2 = cv2.imread(path + newnames[i + 2])
    image3 = cv2.imread(path + newnames[i + 3])
    image4 = cv2.imread(path + newnames[i + 4])
    image5 = cv2.imread(path + newnames[i + 5])
    a = np.concatenate((image, image1,image2, image3,image4, image5), axis = 1)
    image6 = cv2.imread(path + newnames[i + 6])
    image7 = cv2.imread(path + newnames[i + 7])
    image8 = cv2.imread(path + newnames[i + 8])
    image9 = cv2.imread(path + newnames[i + 9])
    image10 = cv2.imread(path + newnames[i + 10])
    image11 = cv2.imread(path + newnames[i + 11])
    b = np.concatenate((image6, image7,image8, image9,image10, image11), axis = 1)
    
    z = np.concatenate((a,b), axis = 0)
    
    image12 = cv2.imread(path + newnames[12])
    image13 = cv2.imread(path + newnames[i + 13])
    image14 = cv2.imread(path + newnames[i + 14])
    image15 = cv2.imread(path + newnames[i + 15])
    image16 = cv2.imread(path + newnames[i + 16])
    image17 = cv2.imread(path + newnames[i + 17])
    c = np.concatenate((image12, image13,image14, image15,image16, image17), axis = 1)
    image18 = cv2.imread(path + newnames[i + 18])
    image19 = cv2.imread(path + newnames[i + 19])
    image20 = cv2.imread(path + newnames[i + 20])
    image21 = cv2.imread(path + newnames[i + 21])
    image22 = cv2.imread(path + newnames[i + 22])
    image23 = cv2.imread(path + newnames[i + 23])
    d = np.concatenate((image18, image19,image20, image21,image22, image23), axis = 1)
    
    y = np.concatenate((c,d), axis = 0)
    
    r = np.concatenate((z,y), axis = 0)

cv2.imshow('a', r)

cv2.waitKey()
cv2.destroyAllWindows()