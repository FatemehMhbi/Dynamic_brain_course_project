# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:40:59 2020

@author: Fatemeh
"""

from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import os


def LSVC_(data, label, n_features):
    """returns the features selected based on linearSVC,
    data and label should be numpy array"""
    lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(data, label)
    coef = np.squeeze(np.sum(np.square(np.array(lsvc.coef_)), axis=0))
    coefidx = np.argsort(coef)
    fidx = coefidx[-n_features:]
    print(fidx)
    return fidx
    

if __name__ == '__main__':
    df = pd.read_csv('CNAs_data.csv') 
    df = df.set_index(df.columns[0])
    label = df['label'].to_numpy()
    print(label)
    df = df.drop(['label'], axis=1).to_numpy()
    #LSVC
    n_features = 500
    lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(df, label)
    coef = np.squeeze(np.sum(np.square(np.array(lsvc.coef_)), axis=0))
    coefidx = np.argsort(coef)
    fidx = coefidx[-n_features:]
    df = df[:,fidx]
    df = pd.DataFrame(df)
    df['label'] = list(pd.factorize(label)[0])
    print(df.shape)
    df.to_csv("CNAs_data_LSVC.csv")