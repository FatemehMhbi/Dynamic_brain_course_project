# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:26:36 2020

"""

# build a simple network
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def PCA_rbf(X, n_componenets):
    print("running PCA...")
    transformer = KernelPCA(n_components=n_componenets, gamma=0.0018, kernel='rbf')
    X_transformed = transformer.fit_transform(X)
    print(X_transformed)
    return pd.DataFrame(X_transformed)
    
    
def RandomForest_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    RandomForest_ = RandomForestClassifier(random_state=0)
    RandomForest_.fit(X_train, y_train)
    y_pred = RandomForest_.predict(X_test)
    print("random forest accuracy: ", RandomForest_.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print("precision: ", precision_score(y_test, y_pred, pos_label=1, average='micro'))
    print("recall: ", recall_score(y_test, y_pred, pos_label=1, average='micro'))
    
    
def logistic_reg_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    logisticRegr = LogisticRegression(penalty="l2", C=0.01, solver='lbfgs', multi_class='multinomial')
    logisticRegr.fit(X_train, y_train)
    y_pred = logisticRegr.predict(X_test)
    print("Logistic regression accuracy: ", logisticRegr.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print("precision: ", precision_score(y_test, y_pred, pos_label=1, average='micro'))
    print("recall: ", recall_score(y_test, y_pred, pos_label=1, average='micro'))
    
    
if __name__ == '__main__':

#    binary_pred = pd.read_csv('_variants_data.csv') 
#    binary_pred = binary_pred.set_index(binary_pred.columns[0])
    df = pd.read_csv('extracted_variants_data_based_centers_40.csv') 
    df = df.set_index(df.columns[0])
    label = df['label'].to_numpy()
    print(label)
    df = df.drop(['label'], axis=1)

    logistic_reg_classification(df, label)
    RandomForest_classification(df, label)

