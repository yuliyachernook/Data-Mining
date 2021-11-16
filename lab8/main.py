from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # for splitting the data into train and test samples
from sklearn.metrics import classification_report  # for model evaluation metrics
import plotly.express as px  # for data visualization
import plotly.graph_objects as go  # for data visualization
from sklearn import metrics

df = pd.read_csv('cardio_train.csv', ';')

y = df.cardio
X = df.drop('cardio', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Linear
lin = SVC(kernel='linear', probability=True, C=1, gamma='scale')
print('Linear')
clf1 = lin.fit(X_train, y_train)

print('Evaluation on test data')
score_te = lin.score(X_test, y_test)
print('Accuracy Score: ', score_te)

print('Evaluation on training data')
score_tr = lin.score(X_train, y_train)
print('Accuracy Score: ', score_tr)

y_pred = lin.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print()

#Poly
pol = SVC(kernel='poly', probability=True, C=1, gamma='scale')
print('Poly')
clf2 = pol.fit(X_train, y_train)

print('Evaluation on test data')
score_te = pol.score(X_test, y_test)
print('Accuracy Score: ', score_te)

print('Evaluation on training data')
score_tr = pol.score(X_train, y_train)
print('Accuracy Score: ', score_tr)

y_pred = pol.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print()

#RBF
rbf = SVC(kernel='rbf', probability=True, C=1, gamma='scale')
print('RBF')
clf3 = rbf.fit(X_train, y_train)

print('Evaluation on test data')
score_te = rbf.score(X_test, y_test)
print('Accuracy Score: ', score_te)

print('Evaluation on training data')
score_tr = rbf.score(X_train, y_train)
print('Accuracy Score: ', score_tr)

y_pred = rbf.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
