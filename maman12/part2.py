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
from scipy import interp

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
    hist, bins = np.histogram(label,codebook.n_clusters,[1,codebook.n_clusters+1])
    return hist

def calcRocAvarage(tpr_list,fpr_list):
    n = len(tpr_list)
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_list[i] for i in range(n)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += interp(all_fpr, fpr_list[i], tpr_list[i])

    # Finally average it and compute AUC
    mean_tpr /= n
    mean_auc =  metrics.auc(all_fpr, mean_tpr)

    return all_fpr, mean_tpr, mean_auc
    
def calcSifts(samples, step_size):
    sift = cv2.xfeatures2d.SIFT_create()
    sifts = []
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
        sifts.append( (dense_sift, sample[1]) )
    
    # stack all train sifts in numpy matrix and return
    return sifts, len(kp)
    
def load_data( basedir , class_a, class_b):
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
    return dataset
    
def shuffle_and_split( dataset, train_test_ratio ):
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
train_data, test_data = shuffle_and_split( load_data(basedir,class_a, class_b),train_test_ratio)

## TRAIN 
# do dense-sift on all train data 
with Timer('do dense-sift on all train data'):
    sifts_with_labels, n_kp= calcSifts(train_data, dsift_step_size)
    sifts, labels = zip(*sifts_with_labels)  
    
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
    y_train =  labels#np.asarray(train_data)[:,1].astype('int')
    
# train svm in different reg-terms(C):
models =[]
for c in regTerms :
    with Timer('Fit SVM with C='+str(c)):
        model = svm.SVC(kernel = 'linear',C=c)
        model.fit(X_train, y_train)
        models.append(model)

## TEST :
with Timer('do dense-sift on all test data'):
    sifts_with_labels, n_kp= calcSifts(test_data, dsift_step_size)

    sifts, labels = zip(*sifts_with_labels)  
with Timer('calc BOW hist of all test data'):
    sample_desc = [] 
    for sample in sifts:
        sampleHist = calcBOWDescriptor(sample, codebook )
        sample_desc.append(sampleHist)
                
    X_test = np.vstack(sample_desc)
    y_test =  labels 

auc = []
accuracy = []
fpr_list = []
tpr_list = []

n_models = len(models)

plt.figure(0)
# create colors, notice number of colors is models + 1 for the mean curve
colors = cm.rainbow(np.linspace(0, 1, n_models+1))
with Timer('calc ROC for each model'):
    for modelIdx in range(n_models):
        c = colors[modelIdx]
        model = models[modelIdx]    
        y_score= model.decision_function(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)#, pos_label=1) 
        fpr_list.append(fpr)
        tpr_list.append(tpr)   
        auc.append(metrics.auc(fpr,tpr))
        y_predict= model.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test,y_predict))
        plt.hold('on')
        plt.plot(fpr,tpr, color=c, label='C='+str(regTerms[modelIdx])+' (auc = {0:0.4f})'
                   ''.format(auc[modelIdx])), plt.xlabel('FPR'), plt.ylabel('TPR')
    
    all_fpr, mean_tpr, mean_auc = calcRocAvarage(tpr_list, fpr_list)
    
    plt.plot(all_fpr, mean_tpr,
             label='macro-average ROC curve (auc = {0:0.4f})'
                   ''.format(mean_auc),
             color=colors[-1], linestyle=':', linewidth=4)
    plt.hold('off')
    plt.title('ROC Curve'), plt.xlabel('FPR'), plt.ylabel('TPR')
    plt.legend(loc="lower right")
    plt.show()

with Timer('Compare accuracy between model (as a function of C)'):
    plt.figure(1)
    #plt.plot(range(len(regTerms)), auc, label='auc'), plt.xlabel('C')
    #plt.hold('on')
    plt.plot(range(len(regTerms)), accuracy, label='accuracy'), plt.xlabel('C')
    plt.xticks(range(len(regTerms)),regTerms)
    plt.legend()
    plt.show()

# check several splits of train and test data (not a good reuse of code): 
print 'check several splits of train and test data:' 
# set ratios to test
ratios = [0.8, 0.6, 0.4, 0.2]
#colors
colors = cm.rainbow(np.linspace(0, 1, len(ratios)+1))
# C now is fixed to the best we find in prev run
C = 0.001

# load data:
with Timer('load the data again'):
    dataset = load_data(basedir,class_a, class_b)
# do dense-sift on all train data 
with Timer('do dense-sift on all data (train and test)'):
    sifts_with_labels, n_kp= calcSifts(dataset, dsift_step_size)

print 'run on all ratios:'
fpr_list =[]
tpr_list= []
accuracy_list =[]

plt.figure(2)

for ratioIdx in range(len(ratios)):   
    ratio = ratios[ratioIdx]
    with Timer('split data to train and test'):
        train_data, test_data = shuffle_and_split(sifts_with_labels, ratio)
        sifts, y_train= zip(*train_data)          
    
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
        
    with Timer('Fit SVM with C='+str(C)):
        model = svm.SVC(kernel = 'linear',C=C)
        model.fit(X_train, y_train)
    
    sifts, y_test= zip(*test_data)          
    
    with Timer('calc BOW hist of all test data'):
        sample_desc = [] 
        for sample in sifts:
            sampleHist = calcBOWDescriptor(sample, codebook )
            sample_desc.append(sampleHist)
                
        X_test = np.vstack(sample_desc)
    
    with Timer('predict and calc ROC'):
        y_score= model.decision_function(X_test)
        y_predict = model.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        auc = metrics.auc(fpr,tpr)
        accuracy = metrics.accuracy_score(y_test, y_predict)
        plt.plot(fpr,tpr, label='Train:'+str(int(ratio*100))+'% (auc = {0:0.4f})'
                   ''.format(auc), color=colors[ratioIdx])    
        plt.hold('on')
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        accuracy_list.append(accuracy)
    
all_fpr, mean_tpr, mean_auc= calcRocAvarage(tpr_list, fpr_list)
plt.plot(all_fpr, mean_tpr, label='macro-average ROC curve (auc = {0:0.4f})'
                   ''.format(mean_auc),
             color=colors[-1], linestyle=':', linewidth=4)
plt.hold('off')
plt.title('ROC Curve'), plt.xlabel('FPR'), plt.ylabel('TPR'), plt.ylim([0,1]), plt.xlim([0,0.4])
plt.legend(loc="lower right")

percents = [i * 100 for i in ratios] 
with Timer('Compare accuracy between model (as a function of train/test data ratio)'):
    plt.figure(3)
    #plt.plot(range(len(regTerms)), auc, label='auc'), plt.xlabel('C')
    #plt.hold('on')
    
    plt.plot(range(len(ratios)), accuracy_list, label='accuracy'), plt.xlabel('Train %')
    plt.xticks(range(len(ratios)),percents)
    plt.legend()
    plt.show()
