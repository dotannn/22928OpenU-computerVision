#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:05:05 2017

@author: dotan
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 22:17:56 2017

@author: dotan
"""

import cPickle
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import metrics
import csv
import random
import telegram

TOKEN = '313645318:AAFDuySwzhlaD7EUhJ885bB1seta4-D11qo'
bot = telegram.Bot(token=TOKEN)
chat_id = 199392648


import cropper; reload(cropper)
import facesclassifier; reload(facesclassifier)
import facepreprocess; reload(facepreprocess)
import facerepresentation; reload(facerepresentation)
import faceoperator; reload(faceoperator)
import frontalizer; reload(frontalizer)
import dimreduction; reload(dimreduction)


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


Croptype = ['nocrop']#, '100','120' ,  '125x160']
Extractdata = ['noextra']#, 'onlyflip', 'fullextra']
Align = [ 'noalign']#], 'hassner','rotation', 'affine', '']
Preprocess = [  'nopreprocess']#, 'tantriggs']
Operator = ['lbp'], #'eldp', , 'tplbp','nooperator']#, 'celdp']
Representation = [ 'spatialhist_u8','spatialhist_u4', 'spatialhist8','flatten','spatialhist12','spatialhist_u12']
DimReduct = ['lda100', 'lda50', 'lda200', 'pca100', 'pca50', 'pca200']
Classifier = ['rbfsvm', 'chi2' 'l2']#, '3nnl2']#, 'linearsvmC1', 'linearsvmC01', 'linearsvmC001','rbfsvmC1', 'rbfsvmC01', 'rbfsvmC001']

basedir = '/media/dotan/Data/datasets/proj/'

conf = ''

n_train = 65
train_data, test_data, labelnames = load_data(basedir + 'ExtendedYaleB/',n_train)

results = dict()
for croptype in Croptype:
    conf = croptype
    print conf
    if os.path.isfile(basedir + '/'+ conf) :
        train_cropped, test_cropped = cPickle.load(open(basedir + '/'+ conf,'rb'))
        print 'cropped loaded'
    else :
        if croptype=='nocrop' : 
            train_cropped = train_data
            test_cropped = test_data
            
           # cPickle.dump( (train_cropped,test_cropped), open(basedir + '/'+conf,'wb'))
            pass
        else:
            xidx= croptype.find('x')
            if xidx == -1:
                res = (int(croptype),int(croptype))
            else :
                res = (int(croptype[:xidx]),int(croptype[xidx+1:]))
            
            crop = cropper.Cropper(res)
            
            train_imgs, ytrain = zip(*train_data)
            test_imgs, ytest = zip(*test_data)
            cropped_train_imgs = crop.crop(train_imgs)
            cropped_test_imgs = crop.crop(test_imgs)
            
            train_cropped = zip( cropped_train_imgs, ytrain)
            test_cropped = zip( cropped_test_imgs, ytest)
            
            cPickle.dump( (train_cropped,test_cropped), open(basedir + '/'+conf,'wb'))
        
    for extractdata in Extractdata:
        conf = croptype + '_' + extractdata
        print conf
        if os.path.isfile(basedir + '/'+ conf) :
            train_extract, test_extract= cPickle.load(open(basedir + '/'+ conf,'rb'))
        else :
            if extractdata=='noextra' : 
                train_extract = train_cropped
                test_extract = test_cropped
                print train_extract[0][0].shape
              #  cPickle.dump( (train_extract,test_extract), open(basedir + '/'+conf,'wb'))
            elif extractdata=='onlyflip':
                continue
            elif extractdata=='fullextra':
                continue
            else:
                continue
        for align in Align:
            conf = croptype + '_' + extractdata + '_' + align
            print conf
            if os.path.isfile(basedir + '/'+ conf) :
                train_align, test_align= cPickle.load(open(basedir + '/'+ conf,'rb'))
            else :  
                if align == 'noalign':
                    train_align=train_extract
                    test_align = test_extract
                  #  cPickle.dump( (train_align,test_align), open(basedir + '/'+conf,'wb'))
                else :
                    frontalizer = frontalizer.FaceFrontalizer(align)
                    train_imgs, ytrain = zip(*train_extract)
                    test_imgs, ytest = zip(*test_extract)
                    aligned_train_imgs = frontalizer.frontalize(train_imgs)
                    aligned_test_imgs = frontalizer.frontalize(test_imgs)
                    train_align =zip( aligned_train_imgs, ytrain)
                    test_align =zip( aligned_test_imgs, ytest)
                    cPickle.dump( (train_align,test_align), open(basedir + '/'+conf,'wb'))
                        
            for prep in Preprocess:
                conf = croptype + '_' + extractdata + '_' + align+ '_' + prep
                print conf
                if os.path.isfile(basedir + '/'+ conf) :
                    train_preprocess, test_preprocess= cPickle.load(open(basedir + '/'+ conf,'rb'))
                else :  
                    if prep == 'nopreprocess':
                        train_preprocess= train_align
                        test_preprocess = test_align
                      #  cPickle.dump( (train_preprocess,test_preprocess), open(basedir + '/'+conf,'wb'))
                    else :
                        preprocessor = facepreprocess.FacePreprocessor(prep)
                        train_imgs, ytrain = zip(*train_align)
                        test_imgs, ytest = zip(*test_align)
                        preprocessed_train_imgs = preprocessor.preprocess(train_imgs)
                        preprocessed_test_imgs = preprocessor.preprocess(test_imgs)
                        train_preprocess =zip( preprocessed_train_imgs, ytrain)
                        test_preprocess =zip( preprocessed_test_imgs, ytest)
                        cPickle.dump( (train_preprocess,test_preprocess), open(basedir + '/'+conf,'wb'))
         
                for op in Operator:
                    conf = croptype + '_' + extractdata + '_' + align+ '_' + prep + '_' +op
                    print conf
                    if os.path.isfile(basedir + '/'+ conf) :
                        train_operator, test_operator= cPickle.load(open(basedir + '/'+ conf,'rb'))
                    else :
                        if op == 'nooperator':
                            train_operator = train_preprocess
                            test_operator = test_preprocess
                        #    cPickle.dump( (train_operator,test_operator), open(basedir + '/'+conf,'wb'))
                        else :
                            operate = faceoperator.FaceOperator(op)
                            train_imgs, ytrain = zip(*train_preprocess)
                            test_imgs, ytest = zip(*test_preprocess)
                            operate_train_imgs = operate.transform(train_imgs)
                            operate_test_imgs = operate.transform(test_imgs)
                            train_operator =zip( operate_train_imgs, ytrain)
                            test_operator =zip( operate_test_imgs, ytest)
                            cPickle.dump( (train_operator,test_operator), open(basedir + '/'+conf,'wb'))
                            
                    for represent in Representation:
                        conf = croptype + '_' + extractdata + '_' + align+ '_' + prep+ '_' +op +'_' + represent
                        print conf
                        if os.path.isfile(basedir + '/'+ conf) :
                            train_represent, test_represent= cPickle.load(open(basedir + '/'+ conf,'rb'))
                        else :
                            representor = facerepresentation.FaceRepresentation(represent)
                            train_imgs, ytrain = zip(*train_operator)
                            random.shuffle(test_operator)
                            test_imgs, ytest = zip(*test_operator[:int(len(test_operator)/3)])
                            representor.train(train_imgs, ytrain)
                            
                            train_represent_imgs = representor.represent(train_imgs)
                            test_represent_imgs = representor.represent(test_imgs)
                            train_represent =zip( train_represent_imgs, ytrain)
                            test_represent =zip( test_represent_imgs, ytest)
                            cPickle.dump( (train_represent,test_represent), open(basedir + '/'+conf,'wb'))
                        
                        for dims_reduction in DimReduct:
                            conf = croptype + '_' + extractdata + '_' + align+ '_' + prep+ '_' +op+'_' + represent+ '_' + dims_reduction
                            
                            reductor = dimreduction.DimReduction(dims_reduction)
                            
                            train_imgs, ytrain = zip(*train_represent)
                            test_imgs, ytest = zip(*test_represent)

                            reductor.train(train_imgs,ytrain)
                            train_reduct_imgs = reductor.transform(train_imgs)
                            test_reduct_imgs = reductor.transform(test_imgs)
                            for classifier_type in Classifier:
                                conf = croptype + '_' + extractdata + '_' + align+ '_' + prep+ '_' +op+'_' + represent+ '_' + dims_reduction + '_' + classifier_type
                                print conf
                                # skip irrlevant combination
    #                            if represent == 'flatten' and op == 'nooperator' :
    #                                continue
    #                            if classifier_type == 'chi2' and represent.find('hist')==-1:
    #                                continue
    #                            if classifier_type != 'chi2' and  represent.find('hist') !=-1:
    #                                continue
    #                        
                                model = facesclassifier.FacesClassifier(classifier_type)
                                train_represent_imgs, ytrain = zip(*train_represent)
                                test_represent_imgs, ytest = zip(*test_represent)
                                model.train(train_represent_imgs, ytrain)
                                ypredict = model.predict(test_represent_imgs)
                                results[conf] = metrics.accuracy_score(ytest, ypredict)
                                text= conf + " : " + str(results[conf])
                                bot.sendMessage(chat_id=chat_id, text=text)
    #                            with open('results.csv', 'a') as f:
    #                                writer = csv.writer(f)
    #                                writer.writerow([conf, results[conf]])
    #                            
                                