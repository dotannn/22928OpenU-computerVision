# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:35:57 2016

@author: Dotan
"""

import cv2
import matplotlib.pyplot as plt
import  numpy as np
import matplotlib.mlab as mlab
import math
from skimage.measure import compare_ssim as ssim

#cleanup
plt.close('all')

# A - create 100x100 matrix where each element have a noraml dist with mean=3 and std=2
img=  np.random.normal(3,2,[100,100])

# A - show the matrix as a grayscale image: 
plt.figure(1),plt.axis('off'), plt.imshow( img, 'gray',  vmin=0, vmax=1),plt.title('rand Image')
plt.show()
# B - sort matrix elements and plot the sorted vector : 
sorted_vec = np.sort(img.flatten())
plt.figure(2)
plt.plot(sorted_vec)
plt.show()
# C - plot the histogram of "A" matrix using 32 bins, compare to distribution func:
plt.figure(3)
n, bins, patches = plt.hist(img.ravel(),32,normed=1, facecolor='green', alpha=0.75)
y = mlab.normpdf( bins, 3, 2)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.show()

# D - quant the img  to 256 values  lowest to 0 and max to 255s show tihe new histogram
img_quant = ((img  - img.min() ) / (img.max() - img.min())) * 255
img_quant= np.uint8(img_quant)

plt.figure(4),plt.axis('off'), plt.imshow( img_quant, 'gray',  vmin=0, vmax=255),plt.title('Quant Image')
plt.show()

# E -  what is the mean and std after quantization? 
print "mean: ", img_quant.mean()
print "std: ", img_quant.std()

# F -read a colored image from memory and display it in color and as grayscale
plt.figure(5)

cImage= plt.imread('coloredImage.JPG')
grayImage = cv2.cvtColor(cImage, cv2.COLOR_RGB2GRAY)
plt.subplot(121),plt.axis("off"),plt.imshow(cImage, 'gray'),plt.title('ORIGINAL - COLORED')
plt.subplot(122),plt.axis("off"), plt.imshow(grayImage,'gray'),plt.title('GRAY')
plt.show()

# G - crop the image 
croppedImg = grayImage[100:500,150:600]

plt.figure(6)
plt.axis("off")
plt.imshow(croppedImg, 'gray',vmin=0, vmax=255 ), plt.title('CROPPED')
plt.show()

# H - add aditive gausian noise with mean 0 and variance 3 
noise = np.random.normal(0,math.sqrt(3),croppedImg.shape)

noisedImg = croppedImg + noise
#prevent bad casting errors:
noisedImg[noisedImg  > 255] = 255;
noisedImg[noisedImg < 0] = 0;

noisedImg  = np.uint8(noisedImg)

plt.figure(7)
plt.subplot(131), plt.axis("off"),plt.imshow(croppedImg, 'gray',vmin=0, vmax=255 ), plt.title('CROPPED')
plt.subplot(132), plt.axis("off"),plt.imshow(noise, 'gray'), plt.title('Noise')
plt.subplot(133), plt.axis("off"),plt.imshow(noisedImg, 'gray',vmin=0, vmax=255 ), plt.title('CROPPED + Noise')
plt.show()

# I - apply gausian filter to smooth  the image:
blur = cv2.GaussianBlur(noisedImg,(3,3), 2)

plt.figure(8)

plt.subplot(121), plt.axis("off"),plt.imshow(noisedImg, 'gray',vmin=0, vmax=255 ), plt.title('Noisy image')
plt.subplot(122), plt.axis("off"),plt.imshow(blur, 'gray',vmin=0, vmax=255 ), plt.title('Gaussian blurred image')
plt.show()


rOpt = [0,1,2,3,4,5]
ssimRes = [0, 0,0,0,0,0]

for rIdx in xrange(len(rOpt)):
    r = rOpt[rIdx]
    #work with automatic sigma based on kernel size
    blur = cv2.GaussianBlur(noisedImg,(2*r +1,2*r +1), -1)
    ssimRes[rIdx]=ssim(croppedImg, blur)

plt.figure(9)
plt.plot(rOpt,ssimRes), plt.title('ssim vs gaussian blur radius')
plt.show()
        





