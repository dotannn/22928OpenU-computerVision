#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:41:55 2016

@author: dotan
"""

### imports
import time
import cPickle, gzip
import numpy as np
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

def transformPCA( vec, pca, n_comp):
    return np.dot( vec- pca.mean_ , pca.components_[:n_comp].T)
    
def invTransormPCA(vec, pca, n_comp):
    return np.dot(vec, pca.components_[:n_comp]) + pca.mean_

###  START OF SCRIPT :
plt.close('all')

# intro: Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
# separete to data and labels:
x_train, y_train = train_set
x_test, y_test = test_set

## intro :
# how many images there are for each digit?
plt.figure(0)
h = plt.hist(y_train,range(11))
for i in range(10):
    print str(i) + ": " +str(int(h[0][i]))

plt.show()

# show first 12 images in db:
plt.figure(1)
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
        
## A - run knn on data itself find best k: 

# define a function that runs 10 knn (k=1..10) and plot k vs accuracy graph
# (its a function for future use)
def run10Knn(X, y, Xtest, ytest):
    # fit using KNN-classifier,  check for best k (based on f1-score)
    k_scores= np.zeros([10,1])
    k = range(1,11)

    # knn with different K :
    for curK in k:
            # set the n_jobs to -1 to maximize CPU usage, try algorithm='ball_tree' to make
            # it go faster
            clf = KNeighborsClassifier(n_neighbors=curK, n_jobs=-1, algorithm='ball_tree')
            with Timer('Fit and calc accuracy score for k=' + str(curK) + ':'):
                k_scores[curK-1]= clf.fit(X,y).score(Xtest,ytest)
            
    # show k vs  mean-accuracy score graph
    plt.plot(k, k_scores), plt.xlabel('k'), plt.ylabel('mean-accuracy')
    plt.title('num of nearest neighbors vs mean accuracy score')
    plt.show()

# run the function on the data itself:
with Timer('run 10 times KNN on image data(witouth dim-reduction)'):
    plt.figure(2)
    run10Knn(x_train, y_train, x_test, y_test)


## B do PCA on the train dataset and draw 6 first principle components
pca = PCA()
pca.fit(x_train)
# plot mean image and 6 first priciple components:
plt.figure(3)
plt.subplot(2,4,1), showImg(pca.mean_.reshape([28,28]),'mean image')
plt.subplot(2,4,2), showImg(pca.components_[0].reshape([28,28]),'1st component')
plt.subplot(2,4,3), showImg(pca.components_[1].reshape([28,28]),'2nd component')
plt.subplot(2,4,4), showImg(pca.components_[2].reshape([28,28]),'3rd component')
plt.subplot(2,4,5), showImg(pca.components_[3].reshape([28,28]),'4th component')
plt.subplot(2,4,6), showImg(pca.components_[4].reshape([28,28]),'5th component')
plt.subplot(2,4,7), showImg(pca.components_[5].reshape([28,28]),'6th component')
plt.subplot(2,4,8), showImg(pca.components_[6].reshape([28,28]),'7th component')
plt.show()

# C draw graph of total variance as function of p-components
acc_var = np.cumsum(pca.explained_variance_ratio_) * 100.0
plt.figure(4)
plt.plot(acc_var), plt.xlabel('number of Eigenvectors included'), plt.ylabel('% variability captured')
plt.title('% of variability of data Captured vs. Num of Eigenvectors')
plt.show()

## D how many component needed for 95% or 80% variance:
var95=  [ n for n,i in enumerate(acc_var) if i>95 ][0]
var80=  [ n for n,i in enumerate(acc_var) if i>80 ][0]

## E transform to 2 dimantions and scatter plot result:
x_train2dim = transformPCA(x_train,pca,2)

digits = range(10)
colors = cm.rainbow(np.linspace(0, 1, len(digits)))
plt.figure(5)
for d, c in zip(digits, colors):
    plt.scatter(x_train2dim[y_train==d,0],x_train2dim[y_train==d,1], color=c, label=str(d))
    plt.hold('on')
plt.legend()
plt.hold('off')
plt.show()

# F - find best K  for Knn classifier running on PCA transformed train data with
# 2,10 and 20 components
comps = [2,10,20]
for ccIdx in range(len(comps)):
    cc = comps[ccIdx]
    x_train_pca = transformPCA(x_train,pca,cc)
    x_test_pca = transformPCA(x_test,pca,cc)
    plt.figure(6+ccIdx)
    run10Knn(x_train_pca,y_train, x_test_pca, y_test)
    plt.title('# of NN vs accuracy : ' + str(cc) + ' compnents')
    plt.show()

## G - project random image to 2,5,10,50,100,150 dims and reconstruct 
#choose random sample
sampleIdx = np.uint(np.random.uniform(0,len(x_train)))
sample = x_train[sampleIdx:sampleIdx+1]
ks = [2,5,10,50,100,150,200]

recon = np.zeros([len(ks), len(x_train[0])])
plt.figure(9)
plt.subplot(2,4,1), showImg(sample.reshape([28,28]),'original')
for idx in range(len(ks)):
    num_dim = ks[idx]
    proj = transformPCA(sample,pca, num_dim)
    rec = invTransormPCA(proj,pca, num_dim)
    plt.subplot(2,4,idx+2), showImg(rec.reshape([28,28]), 'dims= ' + str(num_dim))

plt.show()
## H :
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

plt.figure(10)
pcaPredictor.fit(x_train, y_train)

for idx in range(10):
    model = pcaPredictor.pcaModels_[idx]
    for pcIdx in range(6):
        plt.subplot(11,6,6*idx + pcIdx + 1), showImg(model.components_[pcIdx].reshape([28,28]),str(pcIdx+1))

for pcIdx in range(6):
    plt.subplot(11,6,60 + pcIdx + 1),showImg(pca.components_[pcIdx].reshape([28,28]),str(pcIdx+1))

plt.show()

y_predict = pcaPredictor.predict(test_set[0])
print metrics.classification_report(y_predict, test_set[1])
print metrics.accuracy_score(y_predict, test_set[1])
