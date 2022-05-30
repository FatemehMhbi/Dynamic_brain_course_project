# Dynamic_brain_course_project
Project title: fMRI classification using t-SNE for diagnosis of schizophrenia

In this project, I will be applying k-nearest neighbor algorithm on extracted features from rs-fMRI data. 
The goal is to achieve higher accuracy than current methods in schizophrenia vs healthy control classification. 

First connectivity networks are created by calculating the correlation between ROIs. 
Then we will have 10 networks (from different frequencies). These 10 networks are merged in one fusion network using similarity networks fusion algorithm. 
This network represent the feature space for the classification probelm. 
Before classification we reduce the dimensionality of this feature space using PCA. We also apply t-SNE which could seperate healthy from control perfectly.
As we know t-SNE separates dissimilar points while keep the similar points closer. 

