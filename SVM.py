# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:37:54 2020

@author: Fatemeh
"""

from scipy.io import loadmat
import os, glob
import numpy as np
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC, SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Simple_classifications import RandomForest_classification, logistic_reg_classification
from LSVC import LSVC_
from sklearn.utils import shuffle
sns.set()


def plot(x, y, c):
    plt.scatter(x, y, c=c, cmap='viridis')
    plt.savefig('plot_tsne.png')
    

def knn_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    knn = KNeighborsClassifier(n_neighbors=8, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    plt.scatter(
        X_test['x'],
        X_test['y'],
        c=y_pred, 
        cmap = 'Dark2'
    )
    print(confusion_matrix(y_test, y_pred))
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("precision: ", precision_score(y_test, y_pred, pos_label=1, average='micro'))
    print("recall: ", recall_score(y_test, y_pred, pos_label=1, average='micro'))


def kmeans(df):
    print("running kmeans...")
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
    y_kmeans = kmeans.predict(df)
    print(type(y_kmeans))
    sns.scatterplot(x = 'x', y = 'y', data = df)
    plot(df['x'], df['y'], y_kmeans)
    return df


def PCA(X, n_componenets, label):
    print("running PCA...")
    pca = decomposition.PCA(n_components=n_componenets)
    pca.fit(X)
    X_reduced = pca.transform(X)
    print(X_reduced)
#    df = pd.DataFrame()
#    df['x'] = X_reduced[:,0]
#    df['y'] = X_reduced[:,1]
#    sns.scatterplot(x = 'x', y = 'y', data = df)
#    try:
#        plot(df['x'], df['y'], label)
#    except:
#        plot(df['x'], df['y'], label.reshape(label.shape[0]))
    return X_reduced


def T_SNE(mat, label):
    print("running TSNE...")
    TSNE_result = TSNE(learning_rate = 50).fit_transform(mat)
    df = pd.DataFrame()
    df['x'] = TSNE_result[:,0]
    df['y'] = TSNE_result[:,1]
    sns.scatterplot(x = 'x', y = 'y', data = df)
    plot(df['x'], df['y'], label)
    return df


def remove_rows_sp_val(df, col_name, val):
    df_index = df[ df[col_name] ==  val].index
    return df.drop(df_index , inplace=True)


def svm_on_tsne(dataset, labels):
    dataset, labels = shuffle(dataset, labels, random_state=0)
    sum_ = 0
    for i in range(len(dataset.index)):
        data = dataset.drop(dataset.index[i])
        label =  np.delete(labels, i)
        data_index_lsvc = LSVC_(data, label, 100)
        
        data = dataset[dataset.columns[list(data_index_lsvc)]]
        data = T_SNE(data)
        X_test = data.iloc[i]
        data = data.drop(data.index[i])
#        X_test = X_test[list(data_index_lsvc)]
        X_test = X_test.to_frame().T
        y_test = labels[i]
        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(data, label)
#        y_pred = svclassifier.predict(X_test)
#        print(confusion_matrix(y_test,y_pred))
        sum_ = sum_ + svclassifier.score(X_test, y_test)
    print(sum_)
    

if __name__ == '__main__':    
    label_file = pd.read_csv("COBRE_diagnosis.csv")
    label_file = label_file.iloc[:157]
    """remove BP and SZA patients out of data"""
    bp_sza_ixs = [0, 27, 30, 53, 67, 84, 86, 93, 96, 97, 129, 147, 148]
    ixs = list(set(label_file.index) - set(bp_sza_ixs))
    label_file = label_file.loc[ixs]
    labels = pd.DataFrame()
    labels['Diagnosis'] = label_file['Diagnosis(SZ:1, HC:2, BP:0, SZA:-1)'].values
    labels = labels.replace(1, 0).replace(2, 1).to_numpy()
    data = pd.read_csv("COBRE_flattened_Data.csv")
    data = data.set_index(data.columns[0])
    data = data.loc[ixs]
    data['Age'] = label_file['Age'].values
    data['Sex'] = label_file['Sex'].values
    svm_on_tsne(data, labels)

        