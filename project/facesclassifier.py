#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:42:26 2017

@author: dotan
"""
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
import progressbar
from sklearn import svm
from sklearn.model_selection import GridSearchCV

class FacesClassifier:
    def __init__(self, classifier_type):
        self.ctype = classifier_type
    
    
    
    def chi2_distance(self,histA, refs):
        eps = 1e-10
        
        d = np.zeros((len(refs),1))
        
        
        for idx in range(len(refs)):
            histB = refs[idx]
            # compute the chi-squared distance
            d[idx] = 0.5 * np.sum([((histA - histB) ** 2) / (histA + histB + eps)])
        return d
    
    def train(self, X, y):
        self._X = X
        self._y = y
        if self.ctype == '3nnl2':
            self.knn= KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
            self.knn.fit(X,y)
        elif self.ctype == 'linearsvmC1':
            self.svm = OneVsRestClassifier(svm.SVC(kernel='linear', C=1))
            self.svm.fit(X,y)

            
        elif self.ctype == 'linearsvmC01':
            self.svm = OneVsRestClassifier(svm.SVC(kernel='linear', C=0.1))
            self.svm.fit(X,y)
            
        elif self.ctype == 'linearsvmC001':
            self.svm = OneVsRestClassifier(svm.SVC(kernel='linear', C=0.01))
            self.svm.fit(X,y)
            
        elif self.ctype == 'rbfsvm':
            param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
            self.svm = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
            
            self.svm = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1))
            self.svm.fit(X,y)
            
    
        
        
    def predict(self, Xtest):
        print 'predict test'
        if self.ctype == 'l2':
            ypredict = np.zeros((len(Xtest)),'int')
            dists= euclidean_distances(np.vstack(Xtest),np.vstack(self._X))
            bar = progressbar.ProgressBar()
            for idx in bar(range(len(Xtest))):
                dist = dists[idx]
                yIdx = np.argmin(dist)
                ypredict[idx] = self._y[yIdx]
            
            return ypredict
            
        elif self.ctype == 'chi2':
             ypredict = np.zeros((len(Xtest),1))
             bar = progressbar.ProgressBar()
             for idx in bar(range(len(Xtest))):
                 dist = self.chi2_distance(Xtest[idx], self._X )
                 yIdx = np.argmin(dist)
                 ypredict[idx] = self._y[yIdx]

             return ypredict
             
        elif self.ctype == '3nnl2':
            return self.knn.predict(Xtest)
            pass
        elif self.ctype.find('svm') != -1:
            return self.svm.predict(Xtest)
        