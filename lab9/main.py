import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_csv('waterQuality1.csv', ',')
print(data)
data.info()

# Формируем вектор меток У и массив данных Х
y = data["is_safe"].astype(int)
X = data.drop("is_safe", axis=1)
print("Shape of original dataset:", data.shape)

print(f'\nРазделение набора на тестовую и обучающую выборки')
x_train, x_test, y_train, y_test = train_test_split(  # по умолчанию 75% и 25%
    X, y, test_size=0.3, random_state=11)
print(f'Shape of input - training set: {x_train.shape}')
print(f'shape of input - testing set: {x_test.shape}')
print(f'Shape of output - training set: {y_train.shape}')
print(f'Shape of output - testing set: {y_test.shape}')

print("-------------------------------------------")
print("Decision Tree")
first_tree = DecisionTreeClassifier(random_state=11)
# оценка модели с помощью кросс-валидации
scores = cross_val_score(first_tree, x_train, y_train, cv=5)
print(f'Mean: {np.mean(scores)}')

tree_params = {"max_depth": np.arange(1, 11), "max_features": [0.5, 0.7, 1]}
tree_grid = GridSearchCV(first_tree, tree_params, cv=5, n_jobs=-1)
tree_grid.fit(x_train, y_train)
print(f'Best score: {tree_grid.best_score_}')
print(f'Best params: {tree_grid.best_params_}')

print("-------------------------------------------")
print("Метод К ближайших соседей")
first_knn = KNeighborsClassifier()
scorekNN = cross_val_score(first_knn, x_train, y_train, cv=5)
print(f'Mean: {np.mean(scorekNN)}')

knn_params = {"n_neighbors": range(5, 30, 5)}
knn_grid = GridSearchCV(first_knn, knn_params, cv=5)
knn_grid.fit(x_train, y_train)
print(f'Best score: {knn_grid.best_score_}')
print(f'Best params:  {knn_grid.best_params_}')

print("-------------------------------------------")
print("Logistic regression")
logreg = LogisticRegression(max_iter=150)
logreg.fit(x_train, y_train)
print("Logreg score: ", logreg.score(X, y))

y_pred = logreg.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("-------------------------------------------")
print("Random forest")
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=make_scorer(mean_squared_error))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Train score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score")

    plt.legend(loc="best")
    return plt


plot_learning_curve(first_tree, "Decision tree curve", x_train, y_train, cv=5)
plt.show()

plot_learning_curve(first_knn, "K-neighbours curve", x_train, y_train, cv=5)
plt.show()

plot_learning_curve(logreg, "Logistic regression curve", x_train, y_train, cv=5)
plt.show()

plot_learning_curve(clf, "Random forest curve", x_train, y_train, cv=5)
plt.show()
