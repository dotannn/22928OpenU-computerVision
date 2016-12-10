# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 00:51:17 2016

@author: Dotan
"""

import time
import cv2
import os
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn
import matplotlib.cm as cm

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def calcBOWDescriptor( features, codebook ):
    label = codebook.predict(features)
    hist, bins = np.histogram(label,100,[1,101])
    return hist

def calcSifts(samples, step_size):
    sift = cv2.xfeatures2d.SIFT_create()
    train_sifts = []
    first = True
    for sample in samples :
        gray= sample[0]
    
        # extract key points (dense)
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
            for x in range(0, gray.shape[1], step_size)]

        if (debug and first):
            first = False
            plt.figure(99)
            plt.imshow(cv2.drawKeypoints(gray,kp,None))
            plt.show();
    
        kp2, dense_sift = sift.compute(gray,kp)
        train_sifts.append(dense_sift)
    
    # stack all train sifts in numpy matrix and return
    return train_sifts, len(kp)
    
def load_data( basedir , class_a, class_b, train_test_ratio):
    # load and split data 
    files = glob.glob(basedir + '/*.jpg')
    classes = []
    imgs = []
    if os.name =='nt':
        sep = '\\'
    else :
        sep = '/'
    for filename in files :
        # load image and convert to gray    
        img = plt.imread(filename)
        gray= cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
        imgs.append(gray)
        if filename.startswith(basedir +sep +class_a):
            classes.append(0)
        elif filename.startswith(basedir +sep +class_b):
            classes.append(1)

    # attach samples to label
    dataset = zip(imgs, classes)
    
    # shuffle for random train and test
    dataset = sklearn.utils.shuffle(dataset)
    
    # return train and test :
    n_train = int(len(dataset) * train_test_ratio)
    return dataset[:n_train], dataset[n_train:] 

    
###  START OF SCRIPT :

# cleanup : 
plt.close('all')

## params 
basedir = 'data'
class_a = 'opencountry'
class_b = 'coast'
train_test_ratio = 0.80
dsift_step_size = 15
debug = False
n_clusters = 150
kmeans_tol = 1e-3 # 1
regTerms = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10 ]

# load and split data:
train_data, test_data = load_data(basedir,class_a, class_b, train_test_ratio)

## TRAIN 
# do dense-sift on all train data 
with Timer('do dense-sift on all train data'):
    sifts, n_kp= calcSifts(train_data, dsift_step_size)
        
# perform kmeans on all samples (stack all sifts in one matrix)
with Timer('Fit kmeans for vector-quantization'):
    codebook = KMeans(n_clusters= n_clusters, verbose=1, tol=kmeans_tol, n_init=1)
    codebook.fit(np.vstack(sifts))

# create centers histogram for each image :
with Timer('calc bow-descriptors of train'):
    sample_desc = [] 
    for sample in sifts:
        sampleHist = calcBOWDescriptor(sample, codebook )
        sample_desc.append(sampleHist)
                    
    X_train = np.vstack(sample_desc)
    y_train =  np.asarray(train_data)[:,1].astype('int')
    
# train svm in different reg-terms(C):
models =[]
for c in regTerms :
    with Timer('Fit SVM with C='+str(c)):
        model = svm.SVC(kernel = 'linear',C=c)
        model.fit(X_train, y_train)
        models.append(model)

## TEST :
with Timer('do dense-sift on all test data'):
    sifts, n_kp= calcSifts(test_data, dsift_step_size)

with Timer('calc BOW hist of all test data'):
    sample_desc = [] 
    for sample in sifts:
        sampleHist = calcBOWDescriptor(sample, codebook )
        sample_desc.append(sampleHist)
                
    X_test = np.vstack(sample_desc)
    y_test =  np.asarray(test_data)[:,1].astype('int')

auc = [];
plt.figure(0)
colors = cm.rainbow(np.linspace(0, 1, len(regTerms)))
for modelIdx in range(len(models)):
    c = colors[modelIdx]
    model = models[modelIdx]    
    y_score= model.decision_function(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)#, pos_label=1)    
    auc.append(metrics.auc(fpr,tpr))
    plt.hold('on')
    plt.plot(fpr,tpr, color=c, label='C='+str(regTerms[modelIdx])+' auc='+str(auc[modelIdx])), plt.xlabel('FPR'), plt.ylabel('TPR')

plt.hold('off')
plt.title('ROC Curve')
plt.legend()
plt.figure(1)
plt.plot(range(len(regTerms)), auc), plt.xlabel('C'), plt.ylabel('AUC'),plt.title('AUC vs. C val')
plt.xticks(range(len(regTerms)),regTerms)

