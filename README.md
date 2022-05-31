# Dynamic_brain_course_project
Project title: fMRI classification using t-SNE for diagnosis of schizophrenia

In this project, I will be applying different machine learning models on extracted features from rs-fMRI data. 
The goal is to achieve higher accuracy than current methods in schizophrenia vs healthy control classification. 

First connectivity networks are created by calculating the correlation between ROIs. 
Then we will have 10 networks (from different frequencies). These 10 networks are merged in one fusion network using similarity networks fusion algorithm. 
This network represent the feature space for the classification probelm. 
Before classification we reduce the dimensionality of this feature space.

- Apply linear SVC as dimensionality reducion and select 100 features.
- Use t-SNE on selected features and reduce the dimensionality to 2D.
- Using SVM as the classification method, we obtained 0.63 accuracy.
- Evaluation: LOOCV.

