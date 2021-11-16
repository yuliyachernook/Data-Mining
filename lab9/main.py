import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_csv('waterQuality1.csv', ',')
print(data)
data.info()

# Формируем вектор меток У и массив данных Х
y = data["is_safe"].astype(int)
X = data.drop("is_safe", axis=1)
print(f'Размерность массива данных Х: {X.shape}')
print(f'Размерность вектора меток у: {y.shape}')

print(f'\nРазделение набора на тестовую и обучающую выборки')
X_train, X_valid, y_train, y_valid = train_test_split(   # по умолчанию 75% и 25%
    X, y, test_size=0.3, random_state=11)
print(f'X train: {X_train.shape}')
print(f'X test: {X_valid.shape}')
print(f'y train: {y_train.shape}')
print(f'y test: {y_valid.shape}')


first_tree = DecisionTreeClassifier(random_state=11)
#оценка модели с помощью кросс-валидации
scores = cross_val_score(first_tree, X_train, y_train, cv=5)
print(np.mean(scores))

first_knn = KNeighborsClassifier()
scorekNN = cross_val_score(first_knn, X_train, y_train, cv=5)
print("Метод К ближайших соседей")
print(np.mean(scorekNN))

tree_params = {"max_depth": np.arange(1, 11), "max_features": [0.5, 0.7, 1]}
tree_grid = GridSearchCV(first_tree, tree_params, cv=5, n_jobs=-1)
tree_grid.fit(X_train, y_train)
print(f'Лучший коэффициент tree: {tree_grid.best_score_}')
print(f'Лучшие параметры tree: {tree_grid.best_params_}')

knn_params = {"n_neighbors": range(5, 30, 5)}
knn_grid = GridSearchCV(first_knn, knn_params, cv=5)
knn_grid.fit(X_train, y_train)
print(f'Лучший коэффициент kNN: {knn_grid.best_score_}')
print(f'Лучшие параметры kNN: {knn_grid.best_params_}')

estimator = tree_grid.best_estimator_
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=80)
tree.plot_tree(estimator, feature_names=X.columns, max_depth=3, filled=True, fontsize=6)
plt.show()
fig.savefig('tree.png')