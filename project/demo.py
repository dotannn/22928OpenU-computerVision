#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:36:33 2017

@author: dotan
"""
#import standard libraries
from sklearn import metrics
import matplotlib.pyplot as plt
import os

#import my libraries
import utils; reload(utils)
import faceRecognition; reload(faceRecognition)

# params :
basedir = '/media/dotan/Data/datasets/proj/'
n_train = 65
do_plot = True
save_model = True
# load data
with utils.Timer('Loading data...'):
    train_data, test_data, labelnames = utils.load_data(basedir + 'ExtendedYaleB/',n_train)

#split
train_imgs, ytrain = zip(*train_data)
test_imgs, ytest = zip(*test_data)

# plot for report:
if do_plot :
    utils.plot_different_poses(train_imgs,test_imgs,1)
    utils.plot_different_illuminations(train_imgs,7)
    utils.plot_grid_on_image(train_imgs[94],8,8)

#load recognizer
with utils.Timer('Init/load recognizer...'):
    if os.path.isfile('model'):  # check for the model in the same directory
        facerec = faceRecognition.FaceRecotnition.load('model')
    else:
        facerec = faceRecognition.FaceRecotnition() # init from scratch
        # train :     
        with utils.Timer('Train model...'):
            facerec.train(train_imgs, ytrain)
        if save_model:
            with utils.Timer('Save model...'):
                facerec.save('model')


## plot for report
if do_plot:
    utils.plot_alignment_stages(train_imgs[854],facerec)
    utils.plot_illumination_normalization(train_imgs[1560:1560+65],facerec)
    utils.plot_all_descriptors(train_imgs[0],facerec)

# show train t-SNE - REMOVE commet to compute t-SNE

#utils.plot_tsne_before_after(train_imgs, facerec.dictionary_,ytrain)

#predict :
with utils.Timer('Predict test...'):
    ypredict = facerec.predict(test_imgs)

## show results:

# accuracy :
print  facerec.get_config() +'--ACCURACY:' + str(metrics.accuracy_score(ytest,ypredict))

# classification report
print metrics.classification_report(ytest,ypredict)

# plot confusion:
if do_plot:
    conf_mat = metrics.confusion_matrix(ytest,ypredict)
    utils.plot_confusion_matrix(conf_mat,labelnames,cmap=plt.cm.coolwarm,normalize=False)

    #plot ROC
    with utils.Timer('Predict probability test...'):
        yscore = facerec.predict_proba(test_imgs)
                                   
    utils.plot_ROC(yscore, ytest)



