#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:29:59 2017

@author: dotan
"""

import numpy as np
from sklearn.decomposition import PCA

class FaceRepresentation:
    def __init__(self, representation_type):
        self.reptype = representation_type
        self.uniform_lut =np.array([
        0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
        14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
        58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
        58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
        58,58,58,50,51,52,58,53,54,55,56,57])
        
    def train(self, imgs, y):
        if self.reptype == 'pcaAll':
            self.pca =PCA()
            imgs_flatten = [ img.flatten() for img in imgs]
            self.pca.fit(imgs_flatten)
        elif self.reptype == 'pca2048':
            imgs_flatten = [ img.flatten() for img in imgs]
            self.pca =PCA(n_components=2048)
            self.pca.fit(imgs_flatten)
            
        elif self.reptype == 'pca1024':
            imgs_flatten = [ img.flatten() for img in imgs]
            self.pca =PCA(n_components=1024)
            self.pca.fit(imgs_flatten)
            
        elif self.reptype == 'pca512':
            imgs_flatten = [ img.flatten() for img in imgs]
            self.pca =PCA(n_components=512)
            self.pca.fit(imgs_flatten)
            
        elif self.reptype == 'pca256':
            imgs_flatten = [ img.flatten() for img in imgs]
            self.pca =PCA(n_components=256)
            self.pca.fit(imgs_flatten)
        
            
    def spatial_hists(self, img,grid_size=8, uniform=False):
        h = int(np.ceil(img.shape[0] / grid_size))
        w = int(np.ceil(img.shape[1] / grid_size))
        hists =[]
        if uniform == True:
            uniformed_img = self.uniform_lut[np.uint8(img)]
        for i in range(0,img.shape[0], h):
            for j in range(0,img.shape[1], w):
                if uniform==False : 
                    region = img[i:min(i+h,img.shape[0]),j:min(j+w,img.shape[1])]
                    hist = np.histogram(region.flatten(), bins=256, range=(0, 256), normed=True)[0]
                else :
                    region = uniformed_img[i:min(i+h,img.shape[0]),j:min(j+w,img.shape[1])]
                    hist = np.histogram(region.flatten(), bins=59, range=(0, 59), normed=True)[0]
                    pass
                hists.extend(hist)
        
        return np.asarray(hists).ravel()
        
    def represent(self,imgs):
        res = []
        if self.reptype == 'flatten':
            res = [img.ravel() for img in imgs]
            return np.vstack(res)
        elif self.reptype == 'spatialhist8':
            res =[]    
            for img in imgs:
                res.append(self.spatial_hists(img,8))
            return np.vstack(res)
        elif self.reptype == 'spatialhist_u8':
            res =[]    
            for img in imgs:
                res.append(self.spatial_hists(img,8, uniform=True))
            return np.vstack(res)
        elif self.reptype == 'spatialhist12':
            res =[]    
            for img in imgs:
                res.append(self.spatial_hists(img,12))
            return np.vstack(res)
        elif self.reptype == 'spatialhist_u12':
            res =[]    
            for img in imgs:
                res.append(self.spatial_hists(img,12,uniform=True))
            return np.vstack(res)
        elif self.reptype == 'spatialhist4':
            res =[]    
            for img in imgs:
                res.append(self.spatial_hists(img,4,uniform=False))
            return np.vstack(res)
        elif self.reptype == 'spatialhist_u4':
            res =[]    
            for img in imgs:
                res.append(self.spatial_hists(img,4,uniform=True))
            return np.vstack(res)
        elif self.reptype.find('pca') != -1 : 
            imgs_flatten = [ img.flatten() for img in imgs]
            return self.pca.transform(imgs_flatten)
                
                  