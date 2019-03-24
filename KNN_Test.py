import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
colnames = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
data = pd.read_csv(url, names=colnames)

X = data.drop('1', axis=1)
X = X.drop('7', axis=1)
X = X.drop('11', axis=1)
Y = data['11']

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.20)

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(trainX, trainY)

Y_pred = classifier.predict(testX)

print(confusion_matrix(testY, Y_pred))
