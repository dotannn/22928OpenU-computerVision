# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 00:51:17 2016

@author: Dotan
"""

import time
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import glob
from sklearn.cluster import KMeans
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn

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
    for sample in samples :
        gray= sample[0]
    
        # extract key points (dense)
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
            for x in range(0, gray.shape[1], step_size)]

        if (debug):
            plt.imshow(cv2.drawKeypoints(gray,kp))
            plt.show();
    
        kp2, dense_sift = sift.compute(gray,kp)
        train_sifts.append(dense_sift)
    
    # stack all train sifts in numpy matrix and return
    return np.vstack(train_sifts), len(kp)

    
def train(train_samples, dsift_step_size, n_clusters,regTerm ):
    with Timer('do dense-sift on all train data'):
        # do dense-sift on all train data 
        sifts, n_kp= calcSifts(train_samples, dsift_step_size)
        
    # perform kmeans on all samples
    with Timer('Fit kmeans for vector-quantization'):
        codebook = KMeans(n_clusters= n_clusters, verbose=1, tol=1, n_init=1)
        codebook.fit(sifts)

    # create centers histogram for each image :
    with Timer('calc bow-descriptors'):
        sample_desc = [] 
        for sampleIdx in range(0, len(sifts),n_kp ):
            features = sifts[sampleIdx:sampleIdx + n_kp]
            sampleHist = calcBOWDescriptor(features, codebook )
            sample_desc.append(sampleHist)
                
    X = np.vstack(sample_desc)
    y =  np.asarray(train_samples)[:,1]
    
    # train svm :
    with Timer('Fit SVM'):
        clf = svm.LinearSVC(C=regTerm)
        clf.fit(X, y.astype('int'))
    return clf, codebook
    
def load_data( basedir , class_a, class_b, train_test_ratio):
    # load and split data 
    files = glob.glob(basedir + '/*.jpg')
    classes = []
    imgs = []
    for filename in files :
        # load image and convert to gray    
        img = plt.imread(filename)
        gray= cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
        imgs.append(gray)
        if filename.startswith(basedir +'/' +class_a):
            classes.append(1)
        elif filename.startswith(basedir +'/' +class_b):
            classes.append(2)

    # attach samples to label
    dataset = zip(imgs, classes)
    
    # shuffle for random train and test
    dataset = sklearn.utils.shuffle(dataset)
    
    # return train and test :
    n_train = int(len(dataset) * train_test_ratio)
    return dataset[:n_train], dataset[n_train:] 

def detect(samples, codebook,  model, dsift_step_size ):
    samplesDesc =[]
    for sample in samples:
        gray= sample[0][0]
        # extract key points (dense)
        features = calcSifts([gray],dsift_step_size)[0]
        desc = calcBOWDescriptor(features, codebook)
        samplesDesc.append(desc)
    
    y= model.predict(samplesDesc)
    
    return y
    
    
# params 
basedir = 'data'
class_a = 'opencountry'
class_b = 'tallbuilding'
train_test_ratio = 0.8
dsift_step_size = 15
debug = False
n_clusters = 100
regTerm = 1

train_data, test_data = load_data(basedir,class_a, class_b, train_test_ratio)

# do dense-sift on all train data 
with Timer('do dense-sift on all train data'):
    sifts, n_kp= calcSifts(train_data, dsift_step_size)
        
# perform kmeans on all samples
with Timer('Fit kmeans for vector-quantization'):
    codebook = KMeans(n_clusters= n_clusters, verbose=1, tol=1, n_init=1)
    codebook.fit(sifts)

# create centers histogram for each image :
with Timer('calc bow-descriptors of'):
    sample_desc = [] 
    for sampleIdx in range(0, len(sifts),n_kp ):
        features = sifts[sampleIdx:sampleIdx + n_kp]
        sampleHist = calcBOWDescriptor(features, codebook )
        sample_desc.append(sampleHist)
                
    X_train = np.vstack(sample_desc)
    y_train =  np.asarray(train_data)[:,1].astype('int')
    
# train svm :
with Timer('Fit SVM with C='+str(regTerm)):
    model = svm.LinearSVC(C=regTerm)
    model.fit(X_train, y_train)


with Timer('Detect'):
    detect(train_data, codebook, model, dsift_step_size)