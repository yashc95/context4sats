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

## TO REPLACE WITH THE DATA
n = 1000
n_classes = 4
X = np.random.randint(0, 10, (n, n_classes))
Y = np.random.randint(0, n_classes, (n, 1))
print(Y)
## TO REPLACE WITH THE DATA


# Process data 

n_data, n_classes = X.shape
n_val = n_data // 10
n_test = n_val
n_train = n_data - n_val - n_test

data = np.zeros((n_data, n_classes + 1))
data[:, :n_classes] = X
data[:, -1] = Y.reshape(data[:, -1].shape)
np.random.shuffle(data)

X_train = data[:n_train, :-1]
Y_train = data[:n_train, -1]
X_val = data[n_train:n_train+n_val, :-1]
Y_val = data[n_train:n_train+n_val, -1]
X_test = data[n_train+n_val:, :-1]
Y_test = data[n_train+n_val:, -1]

print(data)


# Standard sklearn classifiers: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
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

