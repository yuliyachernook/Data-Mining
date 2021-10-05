import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('mobile.csv')
print(data)

plt.hist(data['Apple'])

median = data['Apple'].median()
mean = data['Apple'].mean()
print('median: ', median)
print('mean: ', mean)

plt.xlabel('Apple (%)')
plt.ylabel('Count')

plt.axvline(median, color='r', linestyle='--')
plt.axvline(mean, color='y', linestyle='--')
plt.show()

plt.boxplot(data['Apple'], showmeans=True)
plt.show()

print(data['Apple'].describe())
