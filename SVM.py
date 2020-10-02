'''

Practice Exercise: Support Vector Machines in Python
Watts Dietrich
Oct 2 2020

The goal of this exercise is to practice building a support vector machine using sklearn.
It uses a breast cancer dataset built in to the sklearn library.
A SVM is trained on 30 features and used to predict whether a tumor is benign or malignant with >90% accuracy.
An 80/20 train/test split was used.
The program also prints the predicted and actual values of the test set for comparison.

'''

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print feature and target names from dataset
print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

# split into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.2)
#print(x_train, y_train)

# string labels to substitute for numeric target values
classes = ['malignant', 'benign']

# create and fit new SVM
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# get predictions and score accuracy
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

for x in range(len(y_pred)):
    print("Predicted: ", classes[y_pred[x]], "Actual: ", classes[y_test[x]])
