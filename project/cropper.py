#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 22:58:08 2017

@author: dotan
"""
import dlib
import progressbar
import cv2
import math
import numpy as np

import frontalizer;reload(frontalizer)

class Cropper:
    def __init__(self, res=(100,100), crop_type='resize', alignment_type='none'):
        self.frontalizer = frontalizer.FaceFrontalizer(alignment_type)
        self.res = res
        self.croptype = crop_type
        self.cascades = []
        self.cascades.append(dlib.get_frontal_face_detector())
        self.cascades.append(cv2.CascadeClassifier('./lbpcascade_frontalface.xml'))
        self.cascades.append(cv2.CascadeClassifier('./lbpcascade_profileface.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_frontalface_default.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_frontalface_alt_tree.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_profileface.xml'))
        self.clahe = cv2.createCLAHE( clipLimit=10.0,  tileGridSize=(12,12))
    
    def detect_face(self,image):
        for cascade in self.cascades:
            if type(cascade) is dlib.dlib.fhog_object_detector:
                bbs = cascade(image)
                if len(bbs) ==0 :
                    bbs = cascade(self.clahe.apply(image))
                    if len(bbs) > 0 :
                        return [bbs[0].left(), bbs[0].top(), bbs[0].right()-bbs[0].left(),bbs[0].bottom()-bbs[0].top()]
                    else :
                        bbs = cascade(cv2.equalizeHist(image))
                        if len(bbs) > 0 :
                            return [bbs[0].left(), bbs[0].top(), bbs[0].right()-bbs[0].left(),bbs[0].bottom()-bbs[0].top()]
                else :
                    return [bbs[0].left(), bbs[0].top(), bbs[0].right()-bbs[0].left(),bbs[0].bottom()-bbs[0].top()]
            else :
                bbs =  cascade.detectMultiScale(
                        image,
                        scaleFactor = 1.1,
                         minNeighbors = 5,
                        minSize = (120, 120),
                        flags = cv2.CASCADE_SCALE_IMAGE
                        )
                if len(bbs) > 0:
                    return list(bbs[0].astype('int'))
                else :
                    bbs =  cascade.detectMultiScale(
                        self.clahe.apply(image),
                        scaleFactor = 1.1,
                         minNeighbors = 5,
                        minSize = (120, 120),
                        flags = cv2.CASCADE_SCALE_IMAGE
                        )
                    if len(bbs) > 0 :
                        return list(bbs[0].astype('int'))
                    else :
                        bbs =  cascade.detectMultiScale(
                        cv2.equalizeHist(image),
                        scaleFactor = 1.1,
                         minNeighbors = 5,
                        minSize = (120, 120),
                        flags = cv2.CASCADE_SCALE_IMAGE
                        )
                        if len(bbs) > 0 :
                            return list(bbs[0].astype('int'))
        return None
    
        
    def crop(self, imgs):
        croppedImgs =[]
        print 'detect and crop faces2'
        bar = progressbar.ProgressBar()
        for imgIdx in bar(range(len(imgs))):
            processedImg = imgs[imgIdx]
        
            newbb = self.detect_face(processedImg)
        
            if newbb != None:
                bb = newbb
            
            
            if (self.frontalizer.atype=='affine'):
                processedImg = self.frontalizer.frontalize_img(processedImg, bb)   
                cropped = processedImg[:,0:480]
                croppedResized= cv2.resize(cropped,(self.res[1],self.res[0]))
                croppedImgs.append(croppedResized)
                continue
            elif (self.frontalizer.atype=='hassner'):
                pass
            elif (self.frontalizer.atype=='rotation'):
                pass
            else:
                pass
        return croppedImgs
#            bbUse = np.copy(bb)
#      
#            if self.croptype=='around':
#                difX = self.res[1] - bb[2]
#                difY = self.res[0] - bb[3]
#                bbUse[2] = self.res[1]
#                bbUse[3] = self.res[0]
#            
#                bbUse[0] = max(bb[0] - math.floor(difX/2),0)
#                if (bbUse[0] + bbUse[2]) > processedImg.shape[1] :
#                    dif = processedImg.shape[1] - (bbUse[0] + bbUse[2])       
#                    bbUse[0] += dif
#
#                bbUse[1] = max(bb[1] - math.floor(difY/2),0)
#                if (bbUse[1] + bbUse[3]) > processedImg.shape[0] :
#                    dif = processedImg.shape[0] - (bbUse[1] + bbUse[3])
#                    bbUse[1] += dif

#            elif self. croptype=='resize':
#                difX = int( float(bb[2]*0.1))
#                difY = int( float(bb[3]*0.15))
#                bbUse[2] = int( float(bb[2]*1.1))
#                bbUse[3] = int( float(bb[3]*1.15))
#                bbUse[0] = max(bb[0] - math.floor(difX/2),0)
#                if (bbUse[0] + bbUse[2]) > processedImg.shape[1] :
#                    dif = processedImg.shape[1] - (bbUse[0] + bbUse[2])       
#                    bbUse[0] += dif
#
#                bbUse[1] = max(bb[1] - math.floor(difY/2),0)
#                if (bbUse[1] + bbUse[3]) > processedImg.shape[0] :
#                    dif = processedImg.shape[0] - (bbUse[1] + bbUse[3])
#                    bbUse[1] += dif
#
#                pass
#        
#
#            cropped = processedImg[bbUse[1]:bbUse[1]+bbUse[3], bbUse[0]:bbUse[0]+bbUse[2]]
#            if self.croptype=='resize':
#                croppedResized= cv2.resize(cropped,(self.res[1],self.res[0]))
#            else :
#                croppedResized = cropped#cv2.resize(cropped,self.res)
#            croppedImgs.append(croppedResized)
#
#        return croppedImgs
#        