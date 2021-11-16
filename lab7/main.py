import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


data = pd.read_csv("waterQuality1.csv")

y = data.is_safe  # Формируем вектор меток У
X = data.drop("is_safe", axis=1)  # Формируем массив данных X


# split X and y into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3)


# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter=150)
# fit the model with data
logreg.fit(X_train, y_train)
# качество модели
print("Logreg score: ", logreg.score(X, y))


# calculate the predicted values
y_pred=logreg.predict(X_test)
print("y_pred: ", y_pred)

# Вычислим confusion matrix (матрицу ошибок)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", cnf_matrix)
print()

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print()

# ROC - график зависимости истинно положительных результатов от ложноположительных.
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# Оценка AUC=1 идеальный классификатор, а 0,5 бесполезный

# Улучшаем модель
# С - сила регуляризации, по умолчанию -1.0
model = LogisticRegression(C=10.0, max_iter=150, random_state=0)
model.fit(X, y)
print("Logreg score: ", model.score(X, y))

y_pred = model.predict(X_test)
print("y_pred: ", y_pred)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", cnf_matrix)
print()

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(X_test)[::, 1]
y_pred_proba2 = model.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
fpr2, tpr2, _ = metrics.roc_curve(y_test,  y_pred_proba2)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc2 = metrics.roc_auc_score(y_test, y_pred_proba2)
plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
plt.plot(fpr2, tpr2, label="data 2, auc="+str(auc2))
plt.legend(loc=4)
plt.show()