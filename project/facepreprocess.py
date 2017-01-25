#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:33:59 2017

@author: dotan
"""
import progressbar
import cv2
import numpy as np
from scipy import ndimage
import math
import scipy.fftpack


class FacePreprocessor:
    def __init__(self, preprocess_type):
        self.ptype = preprocess_type
        
    def TanTriggs(self, img, alpha = 0.1, tau = 10.0, gamma = 0.2, sigma0 = 1.0, sigma1 = 2.0):
        # convert to float32
        X = np.float32(img)
        
        # gamma correct
        I = np.power(X,gamma)
        
        # calc DoG:
        dog = np.asarray(ndimage.gaussian_filter(I,sigma1) - ndimage.gaussian_filter(I, sigma0))
        
        dog = dog / np.power(np.mean(np.power(np.abs(dog), alpha)), 1.0/ alpha)
        dog = dog / np.power(np.mean(np.power(np.minimum(np.abs(dog), tau), alpha)), 1.0/alpha)
        dog =  tau*np.tanh(dog/tau)
        
        # normalize
        return cv2.normalize(dog,dog,0,255,cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    def adjust(self,img):
        return cv2.normalize(img,img,0,255,cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    def lowpassfilter(size, cutoff, n):
        pass
        
    def homomorphic(self, img, boost = 8, cutoff=0.12, order=2, lhist_cur=0.2, uhist_cur=0.2):
        #http://digital.cs.usu.edu/~xqi/Promotion/IETCV.FR.14.pdf
        
        [rows, cols] = img.shape
        
        img = self.adjust(img)
        
        img  = np.float32(img) / 255

        fftlog = scipy.fftpack.fft2(np.log(img+0.01))
        
        h = (1-1/boost)*( 1.0 - np.exp()) + 1/boost;
        
        
        res= np.exp(np.real(scipy.fftpack.ifft2(fftlog.copy() * h)))
        
        return self.adjust(res)
    
    
    def gradientfaces(self,img, sigma = 0.75):
        # https://www.idc-online.com/technical_references/pdfs/electronic_engineering/Illumination%20Insensitive.pdf

        # adjust img
        imgAdjust = self.adjust(img)
        img  = np.float32(img) / 255
        # run gaussian
        xf = ndimage.gaussian_filter(imgAdjust,sigma)
        
        # construct derivatives of gaussians in x and y directions
        W = max(1,np.floor((7.0/2.0) / sigma))
        [Xw,Yw]= np.meshgrid(np.arange(-W, W), np.arange(-W, W))
        
        # build derivative filters
        Gx = -2*Xw*np.exp(-(Xw**2+Yw**2)/(2*sigma**2));
        Gy = -2*Yw*np.exp(-(Xw**2+Yw**2)/(2*sigma**2));
        
        # Compute gradientfaces 
        res = math.atan2(cv2.filter2D(xf,-1, Gy),cv2.filter2D(xf,-1, Gx))
        return self.adjust(res)
        
    
    def preprocess(self, imgs):
        if self.ptype == 'tantriggs':
            processedImgs = []
            print 'preprocess tantriggs'
            bar = progressbar.ProgressBar()
            for idx in bar(range(len(imgs))):
                processed = self.TanTriggs(imgs[idx])
                processedImgs.append(processed)
            
            return processedImgs
            
        elif self.ptype == 'gradientfaces':
            processedImgs = []
            print 'preprocess gradientfaces'
            bar = progressbar.ProgressBar()
            for idx in bar(range(len(imgs))):
                processed = self.gradientfaces(imgs[idx])
                processedImgs.append(processed)
            
            return processedImgs
        elif self.ptype == 'homomorphic':
            processedImgs = []
            print 'preprocess homomorphic'
            bar = progressbar.ProgressBar()
            for idx in bar(range(len(imgs))):
                processed = self.homomorphic(imgs[idx])
                processedImgs.append(processed)
            
            return processedImgs
        else:
            print 'preprocess - doing nothing'
            return imgs
                
        pass