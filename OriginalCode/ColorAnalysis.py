# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 23:04:40 2018
Class for implementing the scores for the composition UI and also the display image
with all the scores
@author: Proprietario
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.cluster import MiniBatchKMeans
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io

#TODO create a mask for the compositional analysis once ready for color
#TODO the brekel and Maya  can output the extracted foreground or the mask . MAybe also unity

class ColorAnalysis():
    
    def __init__(self, imagepath, imagepathTemplate):
        self.image = cv2.imread(imagepath)
        self.templateTangent = cv2.imread(imagepathTemplate, 0)
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
    
    def colorfulnessBySuperpixels(self, num_segments = 4):

        # load the image in OpenCV format so we can draw on it later, then
        # allocate memory for the superpixel colorfulness visualization
        orig = self.image.copy()
        vis = np.zeros(orig.shape[:2], dtype="float")
        # load the image and apply SLIC superpixel segmentation to it via
        # scikit-image
        copyImage = self.image.copy()
        segments = slic(img_as_float(copyImage), n_segments=num_segments, slic_zero=True)
        # loop over each of the unique superpixels
        for v in np.unique(segments):
            # construct a mask for the segment so we can compute image
            # statistics for *only* the masked region
            mask = np.ones(self.image.shape[:2])
            mask[segments == v] = 0
            # compute the superpixel colorfulness, then update the
            # visualization array
            C = self.segment_colorfulness( mask)
            vis[segments == v] = C       
        
        # scale the visualization image from an unrestricted floating point
        # to unsigned 8-bit integer array so we can use it with OpenCV and
        # display it to our screen
        vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
         
        # overlay the superpixel colorfulness visualization on the original
        # image
        alpha = 0.6
        overlay = np.dstack([vis] * 3)
        output = self.image.copy()
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        # show the output images
        cv2.imshow("Input", orig)
        cv2.imshow("Visualization", vis)
        cv2.imshow("Output", output)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def segment_colorfulness(self, mask):
        
    	 # split the image into its respective RGB components, then mask
    	 # each of the individual RGB channels so we can compute
    	 # statistics only for the masked region
    	 (B, G, R) = cv2.split(self.image.astype("float"))
    	 R = np.ma.masked_array(R, mask=mask)
    	 G = np.ma.masked_array(B, mask=mask)
    	 B = np.ma.masked_array(B, mask=mask)
     
    	 # compute rg = R - G
    	 rg = np.absolute(R - G)
     
    	 # compute yb = 0.5 * (R + G) - B
    	 yb = np.absolute(0.5 * (R + G) - B)
     
    	 # compute the mean and standard deviation of both `rg` and `yb`,
    	 # then combine them
    	 stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
    	 meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
     
    	 # derive the "colorfulness" metric and return it
    	 return stdRoot + (0.3 * meanRoot)
    
    
    def colorfulnessDetection (self):
        
        #https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf 
        
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
        
        # display the colorfulness score on the image
        C = stdRoot + (0.3 * meanRoot)
        imageColorfulness = self.image.copy()
        cv2.putText(imageColorfulness, "{:.2f}".format(C), (20, 20), 
		 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        '''
        # show image
        cv2.imshow('colorfulness', imageColorfulness)
        cv2.waitKey()
        cv2.destroyAllWindows()
        '''
        colorfulnessScore = stdRoot + (0.3 * meanRoot)
        
        # derive the "colorfulness" metric and return it
        return colorfulnessScore, imageColorfulness
    
    def colorQuantization (self, n_clusters = 5):
        
        # https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
        
        (h, w) = self.image.shape[:2]
        
        # convert the image from the RGB color space to the L*a*b*
        # color space -- since we will be clustering using k-means
        # which is based on the euclidean distance, we'll use the
        # L*a*b* color space where the euclidean distance implies
        # perceptual meaning
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        # reshape the image into a feature vector so that k-means
        # can be applied
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        
        # apply k-means using the specified number of clusters and
        # then create the quantized image based on the predictions
        clt = MiniBatchKMeans(n_clusters = n_clusters)
        labels = clt.fit_predict(image)
        print('this are the labels: ' , labels)
        print (len(labels))
        quant = clt.cluster_centers_.astype("uint8")[labels]
        # reshape the feature vectors to images
        quant = quant.reshape((h, w, 3))
        image = image.reshape((h, w, 3))
        # convert from L*a*b* to RGB
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        # display the images and wait for a keypress
        cv2.imshow("image", np.hstack([image, quant]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def kmeanscolor (self):
        
        #https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
        imageCopy = self.image.copy()

        # reshape the image to be a list of pixels
        imageCopy = imageCopy.reshape((imageCopy.shape[0] * imageCopy.shape[1], 3))

        # cluster the pixel intensities
        
        clt = KMeans(n_clusters = 3)
        clt.fit(imageCopy)
        
        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        hist = self._centroid_histogram(clt)
        
        bar = self._plot_colors(hist, clt.cluster_centers_)
         
        # show our color bart
        
        cv2.imshow('bar', bar)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    
    def _centroid_histogram(self, clt):
    	# grab the number of different clusters and create a histogram
    	# based on the number of pixels assigned to each cluster
    	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
     
    	# normalize the histogram, such that it sums to one
    	hist = hist.astype("float")
    	hist /= hist.sum()
     
    	# return the histogram
    	return hist
    
    def _plot_colors(self, hist, centroids):
    	# initialize the bar chart representing the relative frequency
    	# of each of the colors
    	bar = np.zeros((50, 300, 3), dtype = "uint8")
    	startX = 0
     
    	# loop over the percentage of each cluster and the color of
    	# each cluster
    	for (percent, color) in zip(hist, centroids):
    		# plot the relative percentage of each cluster
    		endX = startX + (percent * 300)
    		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
    			color.astype("uint8").tolist(), -1)
    		startX = endX
    	
    	# return the bar chart
    	return bar