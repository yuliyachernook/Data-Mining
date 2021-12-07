import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_csv("portugese-students.csv")
data.head()
data.info()

data["school"]=data["school"].map({"GP": 1, "MS": 2})
data["sex"]=data["sex"].map({"F": 1, "M": 2})
data["address"]=data["address"].map({"U": 1, "R": 2})
data["famsize"]=data["famsize"].map({"LE3": 1, "GT3": 5})
data["Pstatus"]=data["Pstatus"].map({"T": 1, "A": 2})
data["Mjob"]=data["Mjob"].map({"teacher": 1, "health": 2, "services": 3, "at_home": 4, "other": 5})
data["Fjob"]=data["Fjob"].map({"teacher": 1, "health": 2, "services": 3, "at_home": 4, "other": 5})
data["reason"]=data["reason"].map({"home": 1, "reputation": 2, "course": 3, "other": 4})
data["guardian"]=data["guardian"].map({"mother": 1, "father": 2, "other": 3})
data["schoolsup"]=data["schoolsup"].map({"yes": 1, "no": 0})
data["famsup"]=data["famsup"].map({"yes": 1, "no": 0})
data["paid"]=data["paid"].map({"yes": 1, "no": 0})
data["activities"]=data["activities"].map({"yes": 1, "no": 0})
data["nursery"]=data["nursery"].map({"yes": 1, "no": 0})
data["higher"]=data["higher"].map({"yes": True, "no": False})
data["internet"]=data["internet"].map({"yes": 1, "no": 0})
data["romantic"]=data["romantic"].map({"yes": 1, "no": 0})

data.head()
data["higher"] = data["higher"].astype("int")

# вектор меток У и массив данных Х
y = data.higher
X = data.drop("higher", axis=1)

print("\Shape of X:", X.shape)
print("\Shape of y:", y.shape)

print("Random forest:")
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                    stratify=y,
                                                    random_state=42)

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)
print("Time spent on model training: %s seconds" % (time.time() - start_time))
print('Accuracy: {:.5f}'.format(accuracy_score(y_test,
                                               preds.argmax(axis=1))))

print("Principal Component Analysis (PCA), n_components=2:")
pca = decomposition.PCA(n_components=2)
X_centered = X - X.mean(axis=0)
pca.fit(X_centered)
X_pca = pca.transform(X_centered)
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=.3,
                                                    stratify=y,
                                                    random_state=42)

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)
print("Time spent on model training: %s seconds" % (time.time() - start_time))
print('Accuracy: {:.5f}'.format(accuracy_score(y_test,
                                               preds.argmax(axis=1))))

pca = decomposition.PCA().fit(X_centered)
plt.figure(figsize=(10, 7))

plt.plot(np.cumsum(pca.explained_variance_ratio_), color='b', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 63)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axhline(0.9, c='r')
plt.show()

print("Principal Component Analysis (PCA), n_components=6:")
pca = decomposition.PCA(n_components=7)
X_centered = X - X.mean(axis=0)
pca.fit(X_centered)
X_pca = pca.transform(X_centered)
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=.3,
                                                    stratify=y,
                                                    random_state=42)

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)
print("Time spent on model training: %s seconds" % (time.time() - start_time))
print('Accuracy: {:.5f}'.format(accuracy_score(y_test,
                                               preds.argmax(axis=1))))