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
from skimage import feature
from sklearn import manifold
from scipy import interp
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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
    plt.figure()
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
    plt.show()
    
    

# Scale and visualize the embedding vectors
def plot_tsne_before_after(before, Xafter, y):
    
    Xbefore=[]
    for img in before:
        Xbefore.append(cv2.resize(img,(0,0),fx=0.25,fy=0.25).flatten())
 
    Xbefore = np.vstack(Xbefore)

    tsneBefore =manifold.TSNE(n_components=2, init='pca', random_state=0)
    tsneAfter = manifold.TSNE(n_components=2, init='pca', random_state=0)
    
    
    XtsneBefore =  tsneBefore.fit_transform(Xbefore)
    XtsneAfter =  tsneAfter.fit_transform(Xafter)
    
    plt.figure()    

    plt.subplot(1,2,1)    
    x_min, x_max = np.min(XtsneBefore, 0), np.max(XtsneBefore, 0)
    XtsneBefore = (XtsneBefore - x_min) / (x_max - x_min)
    
    for i in range(XtsneBefore.shape[0]):
        plt.text(XtsneBefore[i, 0], XtsneBefore[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 28.),
                 fontdict={'weight': 'bold', 'size': 9})
        
    plt.title('t-SNE on input')
    
    plt.subplot(1,2,2)    
    x_min, x_max = np.min(XtsneAfter, 0), np.max(XtsneAfter, 0)
    XtsneAfter = (XtsneAfter - x_min) / (x_max - x_min)
    
    for i in range(XtsneAfter.shape[0]):
        plt.text(XtsneAfter[i, 0], XtsneAfter[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 28.),
                 fontdict={'weight': 'bold', 'size': 9})
        
    plt.title('t-SNE after Transform')
    
    plt.show()



def plot_illumination_normalization(imgs, facerec): 
    plt.figure()
    for idx in range(6):
        imgIdx = idx*10
        tantrigs = facerec.TanTriggs(imgs[imgIdx])
        homo = facerec.homomorphic_filtering(imgs[imgIdx])
    
        plt.subplot(3,6,idx+1), plt.imshow(imgs[imgIdx],'gray'), plt.axis('off')
        plt.subplot(3,6,6+idx+1), plt.imshow(tantrigs,'gray'), plt.axis('off')
        plt.subplot(3,6,12+idx+1), plt.imshow(homo,'gray'), plt.axis('off')
    plt.show()
        

def plot_different_poses( train_imgs, test_imgs, labelIdx ):
    
    pose0 = train_imgs[65*labelIdx]
    pose1 = test_imgs[520*labelIdx]
    pose2 = test_imgs[520*labelIdx+65*2]
    pose3 = test_imgs[520*labelIdx+65*4]
    pose4 = test_imgs[520*labelIdx+65*6]

    plt.figure(), plt.title('different poses')
    plt.subplot(1,5,1), plt.imshow(pose0,'gray'), plt.axis('off')
    plt.subplot(1,5,2), plt.imshow(pose1,'gray'), plt.axis('off')
    plt.subplot(1,5,3), plt.imshow(pose2,'gray'), plt.axis('off')
    plt.subplot(1,5,4), plt.imshow(pose3,'gray'), plt.axis('off')
    plt.subplot(1,5,5), plt.imshow(pose4,'gray'), plt.axis('off')
    plt.show()
    

def plot_different_illuminations( train_imgs,labelIdx ):
    plt.figure()
    for idx in range(5):
        imgIdx = idx* int(65/5)
        plt.subplot(1,5,idx+1), plt.imshow(train_imgs[labelIdx*65 + imgIdx], 'gray'), plt.axis('off')
        
    plt.show()
            
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
    
    plt.show()
    

    
def plot_grid_on_image(img, grid_x, grid_y):
    imgGrid = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    dx = int(img.shape[1]/grid_x)
    dy = int(img.shape[0]/grid_y)
    
    # Custom (rgb) grid color
    grid_color = [0,0,255]

    # Modify the image to include the grid
    imgGrid[:,::dy,:] = grid_color
    imgGrid[::dx,:,:] = grid_color

    plt.figure()
    plt.imshow(imgGrid), plt.axis('off'), plt.title('grid:'+str(grid_y) + 'x'+str(grid_x))

    
def plot_all_descriptors(img, facerec):
    lbp = feature.local_binary_pattern(img,facerec.S,facerec.radius)
    ror = feature.local_binary_pattern(img, facerec.S,facerec.radius, method='ror')
        
    tp = facerec.tplbp(img)
    (ltpl,ltph) = facerec.ltp(img)
    
    plt.figure()
    plt.subplot(2,3,1), plt.imshow(img,'gray'), plt.title('original'), plt.axis('off')
    plt.subplot(2,3,2), plt.imshow(lbp,'gray'), plt.title('lbp'), plt.axis('off')
    plt.subplot(2,3,3), plt.imshow(ror,'gray'), plt.title('rotation-inv lbp'), plt.axis('off')
    plt.subplot(2,3,4), plt.imshow(tp,'gray'), plt.title('Three-Patch LBP'), plt.axis('off')
    plt.subplot(2,3,5), plt.imshow(ltpl,'gray'), plt.title('Ltp low'), plt.axis('off')
    plt.subplot(2,3,6), plt.imshow(ltph,'gray'), plt.title('Ltp high'), plt.axis('off')
    plt.show()
    
    
def plot_ROC(yscore, ytest):
    
    labels = np.unique(ytest)
    ytestbin = label_binarize(ytest,labels)
    n_classes = len(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ytestbin[:, i], yscore[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
        
    lw = 2
    ## Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(ytestbin.ravel(), yscore.ravel(), drop_intermediate=False)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.hold('on')
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cm.rainbow(np.linspace(0, 1, n_classes+2))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.01, 1.35])
    plt.ylim([0.1, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Face Recognition')
    plt.legend(loc="lower right")
    plt.show()