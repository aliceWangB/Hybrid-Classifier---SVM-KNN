import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names=colnames)

X = irisdata.drop('Class', axis=1)
Y = irisdata['Class']

X = X.values
Y = Y.values

X_train = np.zeros(shape=(120, 4))
for i in range(0, 40):
    X_train[i, :] = X[i, :]
for i in range(50, 90):
    X_train[i-10, :] = X[i, :]
for i in range(100, 140):
    X_train[i-20, :] = X[i, :]

Y_train = []
for i in range(0, 40):
    Y_train.append(Y[i])
for i in range(50, 90):
    Y_train.append(Y[i])
for i in range(100, 140):
    Y_train.append(Y[i])

X_test = np.zeros(shape=(30, 4))
for i in range(40, 50):
    X_test[i-40, :] = X[i, :]
for i in range(90, 100):
    X_test[i-80, :] = X[i, :]
for i in range(140, 150):
    X_test[i-120, :] = X[i, :]

Y_test = []
for i in range(40, 50):
    Y_test.append(Y[i])
for i in range(90, 100):
    Y_test.append(Y[i])
for i in range(140, 150):
    Y_test.append(Y[i])

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, Y_train)

Y_pred = svclassifier.predict(X_test)

print(confusion_matrix(Y_test, Y_pred))
