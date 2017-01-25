#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:16:42 2017

@author: dotan
"""

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

class FVF:
    def __init__(self):
        self.maxFeatures = 10^6
        self.scaleFactor = 1.4142
        self.n_scales = 5
        self.pca_reduction = 64
        self.gmm =  GaussianMixture(n_components=512,
                   covariance_type='diag', max_iter=20, random_state=0)
        pass
    
    def dsifts(self,samples, step_size=1):
        eps = 1e-7
        sift = cv2.xfeatures2d.SIFT_create()
        sifts = []
        n_features = self.maxFeatures/ len(samples)
        for sample in samples :  
            
            w = sample.shape[1]
            h = sample.shape[0]

            # extract key points (dense)
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, sample.shape[0], step_size) 
                for x in range(0, sample.shape[1], step_size)]
    
            kp2, dense_sift = sift.compute(sample,kp)
          
            # make rootSIFT
            dense_sift /= (dense_sift.sum(axis=1, keepdims=True) + eps)
            dense_sift= np.sqrt(dense_sift)
            
            #augment locations:
            pts = np.float32([kp2[idx].pt for idx in range(len(kp2))])
            
            # normalize between -0.5 to 0.5
            pts[:,0] =  pts[:,0] / w - 0.5
            pts[:,1] =  pts[:,1] / h - 0.5
            
            dense_sift = np.hstack((dense_sift,pts))
            
            if dense_sift.shape[0] > n_features:
                np.random.shuffle(dense_sift)
                features = dense_sift[:n_features,:]
            else :
                features = dense_sift

            sifts.append(features )
        # stack all train sifts in numpy matrix and return
        return sifts, len(kp)
    
    def fit(self,X,y):
        sifts = self.dsifts(X)
        
        # PCA to 'pca_reduction'
        
        