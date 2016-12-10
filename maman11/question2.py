# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:34:48 2016

@author: Dotan
"""

import cv2
import matplotlib.pyplot as plt
import  numpy as np
import matplotlib.mlab as mlab
import math
import matplotlib as mpl
from skimage.measure import compare_mse as mse
import scipy
import matplotlib.gridspec as gridspec

from scipy import signal 

def color2bayer(img):
    imgMosaic = np.zeros(img.shape[:2], 'uint8')

    imgMosaic[0::2,0::2] =img[0::2,0::2,0] # red
    imgMosaic[0::2,1::2]= img[0::2,1::2,1] # green1
    imgMosaic[1::2,0::2] =  img[1::2,0::2,1] #green2
    imgMosaic[1::2,1::2] = img[1::2,1::2,2] # blue
    
    return imgMosaic    
    
def demosaic_blinear(mosaic):
    
    mosaicDouble = np.double(mosaic)
    redMask = np.zeros(mosaicDouble.shape,'double')
    greenMask = np.zeros(mosaicDouble.shape,'double')
    blueMask = np.zeros(mosaicDouble.shape,'double')

    redMask[0::2,0::2] = mosaicDouble[0::2,0::2]
    greenMask[0::2,1::2] = mosaicDouble[0::2,1::2]
    greenMask[1::2,0::2] = mosaicDouble[1::2,0::2]
    blueMask[1::2,1::2] = mosaicDouble[1::2,1::2]

    redKernel = blueKernel = np.mat([[1,2,1],[2,4,2],[1,2,1]]) / 4.0
    greenKernel = np.mat([[0,1,0],[1,4,1],[0,1,0]]) / 4.0

    demosaic = np.zeros([mosaic.shape[0], mosaic.shape[1], 3])
    # reconstruct missing colors :
    demosaic[:,:,0]= cv2.filter2D(redMask,-1,redKernel)
    demosaic[:,:,1]= cv2.filter2D(greenMask,-1,greenKernel)
    demosaic[:,:,2] = cv2.filter2D(blueMask,-1,blueKernel)
    
    return np.uint8(demosaic)

def demosaic_freeman(mosaic, k):
    mosaicDouble = np.double(mosaic)
    redMask = np.zeros(mosaicDouble.shape,'float64')
    greenMask = np.zeros(mosaicDouble.shape,'float64')
    blueMask = np.zeros(mosaicDouble.shape,'float64')

    redMask[0::2,0::2] = mosaicDouble[0::2,0::2]
    greenMask[0::2,1::2] = mosaicDouble[0::2,1::2]
    greenMask[1::2,0::2] = mosaicDouble[1::2,0::2]
    blueMask[1::2,1::2] = mosaicDouble[1::2,1::2]

    redKernel = blueKernel = np.mat([[1.0,2.0,1.0],[2.0,4.0,2.0],[1.0,2.0,1.0]]) / 4.0
    greenKernel = np.mat([[0,1.0,0],[1.0,4.0,1.0],[0,1.0,0]]) / 4.0

    # reconstruct missing colors :
    resR= cv2.filter2D(redMask,-1,redKernel)
    resG= cv2.filter2D(greenMask,-1,greenKernel)
    resB = cv2.filter2D(blueMask,-1,blueKernel)
    
    resR = np.add(resG,signal.medfilt2d(np.subtract(resR, resG),(k,k)))
    resB = np.add(resG,signal.medfilt2d(np.subtract(resB , resG),(k,k)))
    
    # fix edge case for casting- because numpy casting to uint8 don't set negative values to
    # zero and post 255 to 255, instead numpy doing cycles around 0-255, so I fixed it manually:
    resR[resR > 255] = 255;
    resR[resR < 0] = 0;
    resB[resB > 255] = 255;
    resB[resB < 0] = 0;
    
    return cv2.merge([np.uint8(resR), np.uint8(resG), np.uint8(resB)])


def create_freeman_report(img) :
    
    _mosaic = color2bayer(img)
    kernels = [1,3,5,7,9,11]
    MSEs=[0,0,0,0,0,0]
    VARs=[0,0,0,0,0,0]
    MAXs=[0,0,0,0,0,0]
    
    plt.figure()
    gs = gridspec.GridSpec(2, 6)

    for i in xrange(len(kernels)):
        
        res = demosaic_freeman(_mosaic,kernels[i])
        plt.subplot(gs[0,i]),plt.axis('off'), plt.imshow(res,'gray',vmin=0, vmax=255),plt.title("kernel : " +str( kernels[i]))
        seColor = np.power(np.subtract(np.float64(res),np.float64(img)), 2)
        VARs[i]= np.sqrt(seColor).var()
        MAXs[i]= np.sqrt(seColor).max()
        MSEs[i] = seColor.mean() 
    
    plt.hold('on')
    plt.subplot(gs[1,0:2]), plt.plot(kernels,MSEs), plt.title("MSE vs kernel size")
    plt.subplot(gs[1,2:4]), plt.plot(kernels,VARs), plt.title("err Variance vs kernel size")
    plt.subplot(gs[1,4:6]), plt.plot(kernels,MAXs), plt.title("MAX error vs kernel size")

###### SCRIPT STARTS HERE    
#cleanup
plt.close('all')

# p1  create mosaic image from colored image:
img = plt.imread('thirdAlternative.JPG')

mosaic = color2bayer(img)

plt.figure(1)
plt.subplot(121), plt.axis('off'), plt.imshow(img, 'gray', vmin=0, vmax=255), plt.title('Input image')
plt.subplot(122), plt.axis('off'), plt.imshow(mosaic, 'gray', vmin=0, vmax=255), plt.title('Mosaic')
plt.show()

# p2 - do demosaic using linear convolotion and check the loss 
demosaic = demosaic_blinear(mosaic)

# compare source and recunstructed using mse: 
seColor = np.power(np.subtract(np.float64(demosaic),np.float64(img)), 2)

serr = seColor[:,:,0] + seColor[:,:,1] + seColor[:,:,2]
plt.figure(2)

plt.subplot(231),plt.axis('off'),plt.imshow(img, 'gray'), plt.title('origin')
plt.subplot(232),plt.axis('off'),plt.imshow(demosaic, 'gray'), plt.title('after Demosaic')
plt.subplot(233),plt.axis('off'),plt.imshow(np.sqrt(serr), 'gray'), plt.title('err')

plt.subplot(234),plt.axis('off'),plt.imshow(img[1000:1025,976:1001,:], 'gray'), plt.title('origin area')
plt.subplot(235),plt.axis('off'),plt.imshow(demosaic[1000:1025,976:1001,:], 'gray'), plt.title('area after Demosaic ')
plt.subplot(236),plt.axis('off'),plt.imshow(serr[1000:1025,976:1001], 'gray'), plt.title('area err^2')

print "Colored MSE :", seColor.mean()
print "Total MSE: ", serr.mean()

# take sqrt to get the real max/var and not the max/var of the sqare
print "Total Err Var: ", np.sqrt(serr).var()
print "Colored Err Var: ", np.sqrt(seColor).var()
print "Total Max Err: ", np.sqrt(serr).max()
print "Colored Max Err: ", np.sqrt(seColor).max()

# p3 - implement improved freeman demosaic and check for best median kernel size
freeman = demosaic_freeman(mosaic,5)

plt.figure(3)
plt.subplot(131), plt.axis('off'),plt.imshow(img), plt.title('origin')
plt.subplot(132), plt.axis('off'),plt.imshow(demosaic), plt.title('blinear')
plt.subplot(133), plt.axis('off'),plt.imshow(freeman), plt.title('freeman')

#### NOTE  THIS CODE IS IN COMMENT BECAUSE IT RUNS SLOWLY, uncomment what you want to test:
    
#create_freeman_report(img)

# p4 - try with some other images
#create_freeman_report(plt.imread('q4a.JPG'))
#create_freeman_report(plt.imread('q4b.JPG'))
#create_freeman_report(plt.imread('q4c.JPG'))
#create_freeman_report(plt.imread('q4d.JPG'))