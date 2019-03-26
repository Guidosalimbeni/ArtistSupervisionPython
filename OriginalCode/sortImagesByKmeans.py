# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:06:53 2018

@author: OWNER
"""

import pandas as pd
import numpy as np
import os, os.path
import csv
import shutil
import matplotlib.pyplot as plt

path = "D:\\google drive\\organic\\Machine Learning\\MLCapture256x256"

output_path = "D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\sortedKmeansApr2018"

imagepaths = []
imageNames = []
imagefullNames = []
valid_images = [".jpg", ".png", ".tga", ".gif"]
for f in os.listdir(path):

    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    full_name = name + ext
    if ext.lower() not in valid_images:
        continue
    imagepaths.append(os.path.join(path,f))
    imageNames.append(name)
    imagefullNames.append(full_name)
    
# path to the sorted kmeans
df = pd.read_csv ("D:\google drive\A PhD Project at Godlsmiths\ArtistSupervisionProject\code\Kmeans_judge_pred.csv")
print (df.columns)

def copy_rename( old_file_name, new_file_name, scr_file_path, output_path):
    #src_dir = path
    dst_dir = output_path
    shutil.copy(scr_file_path, dst_dir)
    
    dst_file = os.path.join(output_path, old_file_name)
    new_dst_file_name = os.path.join (dst_dir, new_file_name)
    os.rename(dst_file, new_dst_file_name)
 # this works even if the dataframe is shorter than the folder of image since it is a copy by name functions
# so even if I deleted the rows correponding to the double judgements from william the copy still works
# only the final folder will have less images
 
for i, pred in zip(df.id, df.pred):
    copy_rename(old_file_name = imagefullNames[i-1],
               new_file_name = str(int(pred)) + '_' + imagefullNames[i-1],
               scr_file_path = imagepaths[i-1],
               output_path = output_path)
    
    # print (index,int(judge))