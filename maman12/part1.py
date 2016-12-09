#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:41:55 2016

@author: dotan
"""

### imports
import time
import cPickle, gzip, numpy
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.cm as cm

### Helper functions:
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)
        
def showImg(img, title): 
    plt.imshow(img,'gray'), plt.title(title),plt.axis('off')
    
def run10Knn(X, y, Xtest, ytest ):
    # fit using KNN-classifier,  check for best k (based on f1-score)
    k_score= np.zeros([10,1])
    k = range(1,11)

    # knn with different K :
    for curK in k:
            clf = KNeighborsClassifier(curK)
            with Timer('Fit k=' + str(curK) + ':'):
                clf.fit(X,y)
        
            with Timer('Predict k=' + str(curK) + ':'):
                pred = clf.predict(Xtest)

            #k_f1[curK-1] = metrics.f1_score(ytest,pred, average='weighted')
            k_score[curK-1] = metrics.accuracy_score(ytest,pred, average='weighted')
    # show k vs f1 score  graph
    plt.figure()
    plt.plot(k, k_score)

    

###  START OF SCRIPT :
plt.close('all')

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
# separete to data and labels:
x_train, y_train = train_set
x_test, y_test = test_set

# show first 12 images in db:
plt.figure(0)
rows = 3    
cols = 4

for i in range(rows):
    for j in range(cols):
        idx = i *cols + j
        im = x_train[idx].reshape([28,28])
        label = y_train[idx]
        plt.subplot(rows,cols,idx+1)
        showImg(im, label)
        

plt.show()
        
        
# how many images there are for each digit?
plt.figure(1)
h = plt.hist(y_train,range(11))
for i in range(10):
    print str(i) + ": " +str(int(h[0][i]))

plt.show()


## run on data itself : 
#run10Knn(x_train, y_train, x_test, y_test)
#
## do PCA on the train dataset:
#pca = PCA()
#pca.fit(x_train)
#
## draw graph of total variance as function of p-components
#acc_var = np.cumsum(pca.explained_variance_)
#plt.figure(2)
#plt.plot(acc_var)
#
## how many component needed for 95% or 80% variance:
#acc_var_ratio = np.cumsum(pca.explained_variance_ratio_)
#var95=  [ n for n,i in enumerate(acc_var_ratio) if i>0.95 ][0]
#var80=  [ n for n,i in enumerate(acc_var_ratio) if i>0.90 ][0]
#
## plot mean image and 6 first priciple components:
#plt.figure(3)
#plt.subplot(2,4,1), showImg(pca.mean_.reshape([28,28]),'mean image')
#plt.subplot(2,4,2), showImg(pca.components_[0].reshape([28,28]),'1st component')
#plt.subplot(2,4,3), showImg(pca.components_[1].reshape([28,28]),'2nd component')
#plt.subplot(2,4,4), showImg(pca.components_[2].reshape([28,28]),'3rd component')
#plt.subplot(2,4,5), showImg(pca.components_[3].reshape([28,28]),'4th component')
#plt.subplot(2,4,6), showImg(pca.components_[4].reshape([28,28]),'5th component')
#plt.subplot(2,4,7), showImg(pca.components_[5].reshape([28,28]),'6th component')
#plt.subplot(2,4,8), showImg(pca.components_[6].reshape([28,28]),'7th component')
#plt.show()
#
## transform to 2 dimantions and scatter plot result:
#pca = PCA(n_components=2)
#x_train2dim = pca.fit_transform(x_train)
#
#digits = range(10)
#colors = cm.rainbow(np.linspace(0, 1, len(digits)))
#plt.figure(4)
#for d, c in zip(digits, colors):
#    plt.scatter(x_train2dim[y_train==d,0],x_train2dim[y_train==d,1], color=c)
#    plt.hold('on')
#
#plt.hold('off')
#plt.show()
#
#
#comps = [2,10,20]
#
#for cc in comps:
#    pca = PCA(n_components=cc)
#    pca.fit(x_train)
#    x_train_pca = pca.transform(x_train)
#    x_test_pca = pca.transform(x_test)
#    run10Knn(x_train_pca,y_train, x_test_pca, y_test)
#    plt.title(str(cc) + ' compnents')
#
## g 
pcaAll = PCA()
pcaAll.fit(x_train)
#
##choose random sample
#sampleIdx = np.uint(np.random.uniform(0,len(x_train)))
#sample = x_train[sampleIdx:sampleIdx+1]
#ks = [2,5,10,50,100,150,200]
#
#recon = np.zeros([len(ks), len(x_train[0])])
#plt.figure(8)
#plt.subplot(2,4,1), plt.imshow(sample.reshape([28,28]),'gray'), plt.axis('off'), plt.title('original')
#for idx in range(len(ks)):
#    num_dim = ks[idx]
#    dec = np.dot( sample- pcaAll.mean_ , pcaAll.components_[:num_dim].T)
#    rec = np.dot(dec, pcaAll.components_[:num_dim]) + pcaAll.mean_
#    plt.subplot(2,4,idx+2), plt.imshow(rec.reshape([28,28]),'gray'), plt.axis('off'),
#    plt.title('dims= ' + str(num_dim))
#
#plt.show()
# h1:
class PCAPredictor:
    def __init__(self):
        pass 
    
    def fit(self, X,y):
        self.pcaModels_ = []
        for idx in range(10):
            self.pcaModels_.append(PCA())
            self.pcaModels_[idx].fit(x_train[y_train==idx])
    def predict(self,X) : 
        mses= []
        for model in self.pcaModels_:
            tmp = model.transform(X)
            recon = model.inverse_transform(tmp)
            mse = ((X- recon) ** 2).mean(axis=1)
            mses.append(mse)
        return np.argmin(mses,0)
        
pcaPredictor = PCAPredictor()

plt.figure(9)
pcaPredictor.fit(x_train, y_train)

for idx in range(10):
    model = pcaPredictor.pcaModels_[idx]
    for pcIdx in range(6):
        plt.subplot(11,6,6*idx + pcIdx + 1), showImg(model.components_[pcIdx].reshape([28,28]),str(pcIdx+1))

for pcIdx in range(6):
    plt.subplot(11,6,60 + pcIdx + 1),showImg(pcaAll.components_[pcIdx].reshape([28,28]),str(pcIdx+1))

plt.show()

print metrics.classification_report(pcaPredictor.predict(test_set[0]), test_set[1])
#h2 
# transTest=
#sampleIdx = np.uint(np.random.uniform(0,len(x_train)))
    
