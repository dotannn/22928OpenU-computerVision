# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 00:51:17 2016

@author: Dotan
"""
import telegram

TOKEN = '313645318:AAFDuySwzhlaD7EUhJ885bB1seta4-D11qo'
bot = telegram.Bot(token=TOKEN)
chat_id = 199392648

import time
import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn
import matplotlib.cm as cm
from scipy import interp
from load_data import load_data
from sklearn.metrics.pairwise import euclidean_distances
from skimage import feature

import cropper; reload(cropper)
import facesclassifier; reload(facesclassifier)
import facepreprocess; reload(facepreprocess)
import facerepresentation; reload(facerepresentation)
import faceoperator; reload(faceoperator)
import frontalizer; reload(frontalizer)

basedir = '/media/dotan/Data/datasets/proj/'
n_train = 65

train_data, test_data, labelnames = load_data(basedir + 'ExtendedYaleB/',n_train)
#
train_imgs, ytrain = zip(*train_data)
test_imgs, ytest = zip(*test_data)


                                                  
crop = cropper.Cropper((48,42),'resize','none')
#
train_cropped_imgs = crop.crop(train_imgs)
#
test_cropped_imgs = crop.crop(test_imgs)
#
#train_window_cropped =[]
#test_window_cropped = []
#
#for img in train_imgs:
#    train_window_cropped(img[0:300,0:250])
#
#for img in test_imgs:
#    test_window_cropped(img[0:300,0:250])
#plt.imshow(train_imgs[1])
#
#faces77 =cv2. face.createLBPHFaceRecognizer(grid_x=4,grid_y=4)
#faces77.train(train_window_cropped,np.int32(ytrain))

def chi2_distance(histA, refs):
        eps = 1e-10
        
        d = np.zeros((len(refs),1))
        
        for idx in range(len(refs)):
            histB = refs[idx]
            m = (histA - histB )/ 2
            # compute the chi-squared distance
            d[idx] = 0.5 * np.sum([((histA - m) ** 2) /(m+eps)])
        return d
        

def lbp(img,radius=1):
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
        
def spatial_hist( img,grid_size_x=8,grid_size_y=8, bins=256):
        h = int(np.ceil(img.shape[0] / grid_size_y))
        w = int(np.ceil(img.shape[1] / grid_size_x))
        hists =[]

        for i in range(0,img.shape[0], h):
            for j in range(0,img.shape[1], w):
                region = img[i:min(i+h,img.shape[0]),j:min(j+w,img.shape[1])]
                hist = np.histogram(region.flatten(), bins=bins, range=(0, bins), normed=True)[0] 
                hists.extend(hist)
        
        return np.asarray(hists).ravel()
 

train_desc = []
for img in train_cropped_imgs:
    lbpimg = feature.local_binary_pattern(img,8,2, method='uniform')
    lbpimg= lbpimg[2:-2,2:-2]
    desc =spatial_hist(lbpimg,1,lbpimg.shape[0], 10 )
    train_desc.append(desc)

test_desc = []
for img in test_cropped_imgs:
    lbpimg = feature.local_binary_pattern(img,8,2, method='uniform')
    lbpimg= lbpimg[2:-2,2:-2]
    desc =spatial_hist(lbpimg,1,lbpimg.shape[0], 10 )
    test_desc.append(desc)
    
ypredict = np.zeros((len(test_desc),1),'int')

for idx in range(len(test_desc)):
    dist = chi2_distance(test_desc[idx], train_desc)
    yIdx = np.argmin(dist)
    ypredict[idx] = ytrain[yIdx]

text= "crop resized 1x1grid: " + str(metrics.accuracy_score(ytest,ypredict))
bot.sendMessage(chat_id=chat_id, text=text)



#
#train_hists = []
#test_hists = []
#for img in train_imgs:
#    img_lbp=  lbp(img,1)
#    lbp_hist = spatial_hist(img_lbp)
#    train_hists.append(lbp_hist)
#
#    
#for img in test_imgs:
#    img_lbp=  lbp(img,1)
#    lbp_hist = spatial_hist(img_lbp)
#    test_hists.append(lbp_hist)
#    
#ypredict77 = np.zeros((len(test_hists)),'int')
#dists= euclidean_distances(np.vstack(test_hists),np.vstack(train_hists))
#
#for idx in range(len(test_hists)):
#    dist = dists[idx]
#    yIdx = np.argmin(dist)
#    ypredict77[idx] = ytrain[yIdx]
#
#
#text= "my lbph, no crop 8x8grid: " + str(metrics.accuracy_score(ytest,ypredict77))
#bot.sendMessage(chat_id=chat_id, text=text)
##faces0 = cv2.face.createLBPHFaceRecognizer()
##
##faces0.train(train_imgs, np.int32(ytrain))
#
ypredict0 = []
for img in test_imgs :
    ypredict0.append(faces0.predict(img.copy()))
##    
##text= "No crop : " + str(metrics.accuracy_score(ytest,ypredict0))
##bot.sendMessage(chat_id=chat_id, text=text)
##
##
##faces1 =cv2. face.createLBPHFaceRecognizer(grid_x=1,grid_y=1)
## 
##faces1.train(train_cropped_imgs, np.int32(ytrain))
##
#ypredict1 = []
#for img in test_cropped_imgs :
#    ypredict1.append(faces1.predict(img.copy()))

#text= "crop resized 1x1grid: " + str(metrics.accuracy_score(ytest,ypredict1))
#bot.sendMessage(chat_id=chat_id, text=text)

#
#crop = cropper.Cropper((350,300),'rotation')
#train_cropped_imgs = crop.crop(train_imgs)
#test_cropped_imgs = crop.crop(test_imgs)
#faces2 = cv2.face.createLBPHFaceRecognizer()
#
#faces2.train(train_cropped_imgs, np.int32(ytrain))
#ypredict2 = []
#for img in test_cropped_imgs :
#    ypredict2.append(faces2.predict(img.copy()))
#    
#text= "crop+rotate : " + str(metrics.accuracy_score(ytest,ypredict2))
#
#bot.sendMessage(chat_id=chat_id, text=text)