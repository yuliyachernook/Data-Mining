import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

data = 'column1,column2,column3\n1,15,36\n2,34,68\n3,55.2,\n4,,46.7\n5,64,5\n6,,\n7,23,87'
df = pd.read_csv(StringIO(data))
plt.hist(df["column3"])
plt.show()
print(df)

print("\nIs null:\n", pd.isnull(df))
print("\nInfo:\n", df.info())
print("\nIs null count:\n", df.isnull().sum())

print(df)
print("\nDropna, axis = 1:")
df = df.dropna(axis=1, how=any, thresh=4)
print(df)
print("\nDropna, axis = 2:")
df = df.dropna(axis=0, how=any, thresh=2)
print(df)

print("\nFillna, mean:")
print(df.fillna(df.mean()))

print("\nFillna, forward fill:")
print(df.fillna(value=None, method="ffill"))

print("\nFillna, mode:")

for column in df.columns:
    df[column] = df[column].fillna(df[column].value_counts().idxmax())
print(df)

plt.hist(df["column3"])
plt.show()
