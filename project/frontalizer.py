#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:24:11 2017

@author: dotan
"""

import numpy as np
import dlib
import cv2
import progressbar
import math
import frontalize
import scipy.io as io
import camera_calibration as calib

#https://ai2-s2-pdfs.s3.amazonaws.com/500b/92578e4deff98ce20e6017124e6d2053b451.pdf

class FaceFrontalizer:
    def __init__(self, alignment_type):
        self.atype = alignment_type
        self.landmarkDetector = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        self.clahe = cv2.createCLAHE( clipLimit=10.0,  tileGridSize=(12,12))
        self.model3D = frontalize.ThreeD_Model("./frontalization_models/model3Ddlib.mat", 'model_dlib')
        
    def detect_landmark(self,img, bb=None ):
        if bb == None : 
            rect = dlib.rectangle(0,0,img.shape[1],img.shape[0])
        else :
            rect = dlib.rectangle(bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3])
        
        points = self.landmarkDetector(self.clahe.apply(img), rect)
        
        lmarks = []
        for i in range(68):
            lmarks.append((points.part(i).x, points.part(i).y,))
            
        lmarks = np.asarray(lmarks, dtype='float32')
    
        return lmarks
        
        
    def hassner(self, img, bb=None):
        
        
        imgc = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #https://arxiv.org/pdf/1411.7964v1.pdf
        eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
        
        lmarks = self.detect_landmark(img,bb)
        
        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(self.model3D, lmarks)
        # perform frontalization
        frontal_raw, frontal_sym = frontalize.frontalize(imgc, proj_matrix, self.model3D.ref_U, eyemask)
        
        frontal_raw =cv2.cvtColor(frontal_raw, cv2.COLOR_RGB2GRAY)
        
        return frontal_raw
        
        
    def rotation(self, img, bb=None, scale_factor=44.0):
        
        
        lmarks = np.array(self.detect_landmark(img,bb))
        
        eye_l = (lmarks[37] + lmarks[38] + lmarks[40] + lmarks[41]) * 0.25
        eye_r = (lmarks[43] + lmarks[44] + lmarks[46] + lmarks[47]) * 0.25

        xdist = eye_r[0]-eye_l[0]
        ydist = eye_r[1]-eye_l[1]
        angle = math.atan(ydist/xdist) * (180/ np.pi)
        
        scale = scale_factor / xdist
        
        center=(img.shape[0]/2, img.shape[1]/2)
        rot = cv2.getRotationMatrix2D(center, angle,scale)
        
        return cv2.warpAffine(img,rot,(img.shape[1],img.shape[0]), borderMode=cv2.BORDER_CONSTANT)
        
    def affine(self, img, bb=None):
        
        landmarks = self.detect_landmark(img, bb)
        
        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(INNER_EYES_AND_BOTTOM_LIP) 
        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                       np.float32(img.shape) * MINMAX_TEMPLATE[npLandmarkIndices])
    
        aligned = cv2.warpAffine(img, H, (img.shape[1],img.shape[0]), borderMode=cv2.BORDER_CONSTANT)
        return aligned
    
    def frontalize_img(self, img, bb=None):
        if self.atype == 'affine':
            return self.affine(img,bb)
        elif self.atype =='rotation' :
            return self.rotation(img,bb)
        elif self.atype =='hassner' :
            return self.hassner(img,bb)
        else :
            return img
            
    def frontalize(self,imgs):
        print 'align faces'
        aligned =[]
    
        bar = progressbar.ProgressBar()
        for imgIdx in bar(range(len(imgs))):
            img = imgs[imgIdx]
            aligned.append(self.frontalize_img(img))
        
        return aligned
            
    
            
        
        