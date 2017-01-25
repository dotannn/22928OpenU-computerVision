#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:55:05 2017

@author: dotan
"""
import numpy as np
import cv2
import progressbar

class FaceOperator:
    def __init__(self, operator_type):
        self.optype = operator_type
    
        #https://www.tutorialspoint.com/dip/krisch_compass_mask.htm
        # define Krisch Compass Masks
        self.krischMasks = [
                np.float32([[-3, -3, 5],
                            [-3, 0, 5],
                            [-3, -3, 5]]),
                
                np.float32([[-3, 5, 5],
                            [-3, 0, 5],
                            [-3, -3, -3]]),
                       
                np.float32([[5, 5, 5],
                            [-3, 0, -3],
                            [-3, -3, -3]]),
                       
                np.float32([[5, 5, -3],
                            [5, 0, -3],
                            [-3, -3, -3]]),
                       
                np.float32([[5, -3, -3],
                            [5, 0, -3],
                            [5, -3, -3]]),
                       
                np.float32([[-3, -3, -3],
                            [5, 0, -3],
                            [5, 5, -3]]),
                       
                np.float32([[-3, -3, -3],
                            [-3, 0, -3],
                            [5, 5, 5]]),
                       
                np.float32([[-3, -3, -3],
                            [-3, 0, 5],
                            [-3, 5, 5]])
                       ]
                    
    
                
    def lbp(self, img,radius):
        #http://www.ee.oulu.fi/mvg/files/pdf/pdf_494.pdf
        res = np.zeros(img.shape, 'uint8')
        for i in range(radius,img.shape[0]-radius):
            for j in range(radius,img.shape[1]-radius):
                center = img[i,j]
                val = int(img[i-radius, j] > center)
                val = val + int(img[i-radius, j+radius] > center) * 2
                val = val + int(img[i, j+radius] > center) * (2**2) 
                val = val + int(img[i+radius, j+radius] > center) * (2**3) 
                val = val + int(img[i+radius, j] > center) * (2**4) 
                val = val + int(img[i+radius, j-radius] > center) * (2**5) 
                val = val + int(img[i, j-radius] > center) * (2**6)    
                val = val + int(img[i-radius, j-radius] > center) * (2**7)
                res[i,j] =  val
        return res
    
    def tplbp(self, img):
        #http://www.openu.ac.il/home/hassner/projects/Patchlbp/WolfHassnerTaigman_ECCVW08.pdf
        res = np.zeros(img.shape, 'uint8')
        radius = 2
        for i in range(radius,img.shape[0]-radius):
            for j in range(radius,img.shape[1]-radius):        
                val = 0
                val = val | ((img[i,j] - img[i  ,j-2]) > (img[i,j] - img[i-2,j ])) * 1;
                val = val | ((img[i,j] - img[i-1,j-1]) > (img[i,j] - img[i-1,j+1])) * 2;
                val = val | ((img[i,j] - img[i-2,j  ]) > (img[i,j] - img[i  ,j+2])) * 4;
                val = val | ((img[i,j] - img[i-1,j+1]) > (img[i,j] - img[i+1,j+1])) * 8;
                val = val | ((img[i,j] - img[i  ,j+2]) > (img[i,j] - img[i+1,j  ])) * 16;
                val = val | ((img[i,j] - img[i+1,j+1]) > (img[i,j] - img[i+1,j-1])) * 32;
                val = val | ((img[i,j] - img[i+1,j  ]) > (img[i,j] - img[i  ,j-2])) * 64;
                val = val | ((img[i,j] - img[i+1,j-1]) > (img[i,j] - img[i-1,j-1])) * 128;
                res[i,j]= val
        return res
                
    def eldp(self,img):
        #http://digital.cs.usu.edu/~xqi/Promotion/IETBio.FaceRecognition.15.pdf
        # run gaussian blur on image
        smooth = cv2.GaussianBlur(img, (7,7),sigmaX=1, sigmaY=1)
        smooth = np.float32(smooth)
    
        # calculate edge responses using krisch masks
        ks = []
        for mask in self.krischMasks:
            ks.append(cv2.filter2D(smooth, -1, mask))
        
        # calc ELDP from edges
        res = np.zeros(img.shape,'uint8')
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                val =0
                for k in range(len(ks)):
                    val = val  + np.int(ks[k][i,j]>0) * np.power(2,k)
                    res[i,j] = val

        return res
        
    def transform(self, imgs):
        print 'transform using ' + self.optype + 'method'
        bar = progressbar.ProgressBar()
        res = []
        if self.optype == 'eldp':
            for imgIdx in bar(range(len(imgs))):  
                res.append(self.eldp(imgs[imgIdx]))
            return res
        elif self.optype =='lbp':
            for imgIdx in bar(range(len(imgs))):  
                res.append(self.lbp(imgs[imgIdx],1))
            return res
        elif self.optype =='tplbp':
            for imgIdx in bar(range(len(imgs))):  
                res.append(self.tplbp(imgs[imgIdx]))
            return res
        else :
            return imgs
            
        
        