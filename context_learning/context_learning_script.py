import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

n_classes = 4

import generate_features

f = generate_features.featureExtractor('./split_data_yolo_cleaned/split_train_clean')
train_data_list = []
n_files = len(f.files)
i = 0
for file in f.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f.subImageCounts(file, False):
        l = pair[0]+[int(pair[1])]
        train_data_list.append(l)
data_train = np.array(train_data_list)

f = generate_features.featureExtractor('./split_data_yolo_cleaned/split_val_clean_balanced')
val_data_list = []
n_files = len(f.files)
i = 0
for file in f.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f.subImageCounts(file, False):
        l = pair[0]+[int(pair[1])]
        val_data_list.append(l)
data_val = np.array(val_data_list)
    


# Process data 
X_train = data_train[:, :-1]
Y_train = data_train[:, -1]
X_val = data_val[:, :-1]
Y_val = data_val[:, -1]



# Standard sklearn classifiers: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", #"Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for name, clf in zip(names, classifiers):
    print("Fitting classifier", name, "...")
    clf.fit(X_train, Y_train)
    print("Fitted")
    print("Score: ", clf.score(X_val, Y_val))
    
    
    
"""
Results obtained:

Fitting classifier Nearest Neighbors ...
Fitted
Score:  0.965687053217
Fitting classifier Linear SVM ...
Fitted
Score:  0.918347895155
Fitting classifier RBF SVM ...
Fitted
Score:  0.973153296267
Fitting classifier Decision Tree ...
Fitted
Score:  0.928196981732
Fitting classifier Random Forest ...
Fitted
Score:  0.930738681493
Fitting classifier Neural Net ...
Fitted
Score:  0.931850675139
Fitting classifier AdaBoost ...
Fitted
Score:  0.916441620334
Fitting classifier Naive Bayes ...
Fitted
Score:  0.841620333598
Fitting classifier QDA ...
Fitted
Score:  0.841302621128
    
"""    
    
    
    
    
    
    
    
    
    
    
    
