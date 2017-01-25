#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:36:33 2017

@author: dotan
"""


# TODO - remove when finish
import telegram

TOKEN = '313645318:AAFDuySwzhlaD7EUhJ885bB1seta4-D11qo'
bot = telegram.Bot(token=TOKEN)
chat_id = 199392648


#import standard libraries
import numpy as np
from sklearn import metrics
from sklearn import manifold
import matplotlib.pyplot as plt
import pandas_ml as pdml

#import my libraries
import utils; reload(utils)
import faceRecognition; reload(faceRecognition)


# params :
basedir = '/media/dotan/Data/datasets/proj/'
n_train = 65

# load data
with utils.Timer('Loading data...'):
    train_data, test_data, labelnames = utils.load_data(basedir + 'ExtendedYaleB/',n_train)

#split
train_imgs, ytrain = zip(*train_data)
test_imgs, ytest = zip(*test_data)

#load recognizer
with utils.Timer('Init recognizer...'):
    facerec = faceRecognition.FaceRecotnition( pn_method='rotate', crop=(60,60))

## plot for report
utils.plot_alignment_stages(train_imgs[854],facerec)

utils.plot_illumination_normalization(train_imgs[1560:1560+65],facerec)

# train :     
with utils.Timer('Train model...'):
    facerec.train(train_imgs, ytrain)

# TODO show cropped/ cropped aligned

# show train t-SNE
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#X_tsne = tsne.fit_transform(facerec.dictionary_)
#
#plot_embedding(X_tsne,"t-SNE embedding of train data")

#plt.show()

#predict :
with utils.Timer('Predict test...'):
    ypredict = facerec.predict(test_imgs)

## show results:

# accuracy :
print metrics.accuracy_score(ytest,ypredict)

text= facerec.get_config() +'--ACCURACY:' + str(metrics.accuracy_score(ytest,ypredict))
bot.sendMessage(chat_id=chat_id, text=text)

# classification report
print metrics.classification_report(ytest,ypredict)


conf_mat = metrics.confusion_matrix(ytest,ypredict)
utils.plot_confusion_matrix(conf_mat,labelnames,cmap=plt.cm.coolwarm)






