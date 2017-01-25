#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:17:04 2017

@author: dotan
"""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
import cv2

def load_data( basedir, n_train ):
    train =[]
    test = []

    labels = os.listdir(basedir)
    for labelIdx in range(len(labels)):
        labelFiles = glob.glob(basedir  + labels[labelIdx] + '/*.pgm')
        labelFiles.sort()
        
        loadedFiles = [plt.imread(file) for file in labelFiles]
        trainLabel = zip(loadedFiles[:n_train], [int(labelIdx)]*n_train)
        testLabel = zip(loadedFiles[n_train:], np.int16([labelIdx]*len(loadedFiles[n_train:])))
        train.append(trainLabel)
        test.append(testLabel)
        
    
    return np.vstack(train), np.vstack(test), labels


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


        
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    

# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
#    x_min, x_max = np.min(X, 0), np.max(X, 0)
#    X = (X - x_min) / (x_max - x_min)
#
#    plt.figure()
#    ax = plt.subplot(111)
#    for i in range(X.shape[0]):
#        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
#                 color=plt.cm.Set1(y[i] / 10.),
#                 fontdict={'weight': 'bold', 'size': 9})
#
#    if hasattr(offsetbox, 'AnnotationBbox'):
#        # only print thumbnails with matplotlib > 1.0
#        shown_images = np.array([[1., 1.]])  # just something big
#        for i in range(digits.data.shape[0]):
#            dist = np.sum((X[i] - shown_images) ** 2, 1)
#            if np.min(dist) < 4e-3:
#                # don't show points that are too close
#                continue
#            shown_images = np.r_[shown_images, [X[i]]]
#            imagebox = offsetbox.AnnotationBbox(
#                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
#                X[i])
#            ax.add_artist(imagebox)
#    plt.xticks([]), plt.yticks([])
#    if title is not None:
        plt.title(title)


def plot_illumination_normalization(imgs, facerec):
    
    
    plt.figure()
    for idx in range(6):
        imgIdx = idx*10
        tantrigs = facerec.TanTriggs(imgs[imgIdx])
        homo = facerec.homomorphic_filtering(imgs[imgIdx])
        plt.subplot(3,6,idx+1), plt.imshow(imgs[imgIdx],'gray'), plt.axis('off')
        plt.subplot(3,6,6+idx+1), plt.imshow(tantrigs,'gray'), plt.axis('off')
        plt.subplot(3,6,12+idx+1), plt.imshow(homo,'gray'), plt.axis('off')
        

    
def plot_alignment_stages(img, facerec):
    
    bb= facerec.detect_face(img)

    imgFace = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    imgFace = cv2.rectangle(imgFace,(bb[0],bb[1]), (bb[0]+bb[2], bb[1]+bb[3]),(255,0,0),2)
    
    lmrks = facerec.detect_landmark(img,bb)
    
    
    rotate = facerec.noramlize_alignment_rotate(img,bb)
    
    affine = facerec.noramlize_alignment_affine(img,bb)
    smaller = min(affine.shape[0], affine.shape[1])
    affine = affine[:smaller,:smaller]
    
    plt.figure()
    plt.subplot(1,5,1), plt.imshow(img,'gray'), plt.title('input image'), plt.axis('off')
    plt.subplot(1,5,2), plt.imshow(imgFace), plt.title('face detected'), plt.axis('off')
    plt.subplot(1,5,3), plt.imshow(imgFace), plt.title('face-landmarks detected'), 
    plt.hold('on'), plt.scatter(lmrks[:,0], lmrks[:,1], color='b'), plt.axis('off')
    plt.hold('off')
    
    plt.subplot(1,5,4), plt.imshow(rotate,'gray'), plt.title('rotate method'), plt.axis('off')
    plt.subplot(1,5,5), plt.imshow(affine,'gray'), plt.title('affine method'), plt.axis('off')
    
