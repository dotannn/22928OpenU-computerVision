#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:13:40 2017

@author: dotan
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

DimReduct = ['lda100', 'lda50', 'lda200', 'pca100', 'pca50', 'pca200']

class DimReduction:
    def __init__(self, method):
        self.method = method
        
        if self.method =='lda100':
            self.lda = LinearDiscriminantAnalysis(n_component=100)
        elif self.method =='lda50':
            self.lda = LinearDiscriminantAnalysis(n_component=50)
        elif self.method =='lda200':
            self.lda = LinearDiscriminantAnalysis(n_component=200)

        elif self.method =='pca100':
            self.pca =PCA(n_components=100)
        elif self.method =='pca50':
            self.pca =PCA(n_components=50)
        elif self.method =='pca200':
            self.pca =PCA(n_components=200)
            
    def train(self,X,y):
        if self.method.find('lda')!= -1:
            self.lda.fit(X,y)
        elif self.method.find('pca')!= -1:
            self.pca.fit(X,y)
        
    def transform(self, X):
        if self.method.find('lda') != -1 :
            return self.lda.transform(X)
        elif self.method.find('pca') != -1 :
            return self.pca.transform(X)
            
        
    
        
    