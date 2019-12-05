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
data_list = []
n_files = len(f.files)
i = 0
for file in f.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f.subImageCounts(file, False):
        l = pair[0]+[int(pair[1])]
        data_list.append(l)
data = np.array(data_list)
    


# Process data 

n_data = data.shape[0]
n_classes = data.shape[1]-1
n_val = n_data // 10
n_test = n_val
n_train = n_data - n_val - n_test

np.random.shuffle(data)

X_train = data[:n_train, :-1]
Y_train = data[:n_train, -1]
X_val = data[n_train:n_train+n_val, :-1]
Y_val = data[n_train:n_train+n_val, -1]
X_test = data[n_train+n_val:, :-1]
Y_test = data[n_train+n_val:, -1]

#print(data)


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