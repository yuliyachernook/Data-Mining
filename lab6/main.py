import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import patsy as pt
import sklearn.linear_model as lm

data = pd.read_csv("winequality-red.csv")
print(data)
data.head()

sns.heatmap(data, center=0, cmap='gist_ncar')
plt.show()

sns.heatmap(data.corr(), annot=True)

scatter_matrix(data)
plt.show()

X = data[['density']].values
y = data['fixed acidity'].values
reg = LinearRegression()
reg.fit(X, y)

#Визуализация
plt.scatter(X, y)
plt.plot(X, reg.predict(X), linewidth=2)
plt.show()

print('Determ coeff:', reg.score(X, y, sample_weight=None))

X2 = data[['density', 'volatile acidity', 'citric acid', 'residual sugar', 'pH']].values
reg2 = LinearRegression()
reg2.fit(X2, y)
print('Determ coeff:', r2_score(y, reg2.predict(X2)))


