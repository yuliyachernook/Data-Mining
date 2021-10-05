import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()

array1 = np.array([[1, 2, 3], [3, 5, 6]])
print('array1: ', array1)

array2 = np.array([[9, 8, 7], [6, 5, 4]])
print('array2: ', array2)

arrayAdd = array1 + array2
print('array1 + array2: ', arrayAdd)

arraySub = array1 - array2
print('array1 - array2: ', arraySub)

arrayMult = array1 * array2
print('array1 * array2: ', arrayMult)

arrayRand = np.random.randint(0, 20, 20)
print('random array: ', arrayRand)

arrayReshape = arrayRand.reshape(4, 5)
print('array 4x5: ', arrayReshape)

arraySplit = np.array_split(arrayReshape, 2)
print('split array: ', arraySplit)
print(arraySplit[0])

arraySearch = np.argwhere(arraySplit[0] == 5)
print('indexes of elements with required value: ', arraySearch)
print('count of elements with required value: ', len(arraySearch))

min = np.min(arraySplit[1])
print('min: ', min)

max = np.max(arraySplit[1])
print('max: ', max)

average = np.average(arraySplit[1])
print('average: ', average)

series = pd.Series(arrayRand)
print('series from numpy: ', series)
print('size: ', series.size)
print('series[2]: ', series[2])
print('series[0:3]: ', series[0:3])
series[2] = 5
print('series[series > 10]: ', series[series > 10])
print('series / 2: ', series/2)
print('series.value_counts(): ', series.value_counts())

dict = {'0': 5, '1': 4, '2': 6}
series2 = pd.Series(dict)
print('series from dict: ', series2)

series3 = pd.Series(np.random.randint(0, 10, 20))
print('series2 from numpy: ', series3)

seriesAdd = series + series3
print('seriesAdd: ', seriesAdd)

seriesSub = series - series3
print('seriesSub: ', seriesSub)

seriesMult = series * series3
print('seriesMult: ', seriesMult)

s = {"price": pd.Series([2, 6, 4], ['v1', 'v2', 'v3']),
     "count": pd.Series([14, 2, 3], ['v1', 'v2', 'v3'])}
dataframe = pd.DataFrame(s)
print('dataframe: \n', dataframe)
print('index: ', dataframe.index)
print('columns: ', dataframe.columns)
print('avg price: ', dataframe.price.mean())
print('size: ', dataframe.size)
print('values: ', dataframe.values)
print('dataframe[0:2]: ', dataframe[0:2])
print('dataframe T: ', dataframe.T)
print('dataframe min: ', dataframe.idxmin())
print('dataframe max: ', dataframe.idxmax())
print('dataframe add: ', dataframe.add(dataframe))
print('dataframe sub: ', dataframe.sub(dataframe))
print('dataframe div: ', dataframe.div(dataframe))
print('dataframe mul: ', dataframe.mul(dataframe))

n = {"price": np.array([6, 5, 12]),
      "count": np.array([20, 5, 4])}
dataframe2 = pd.DataFrame(n, ['v1', 'v2', 'v3'])
print('dataframe: \n', dataframe2)
print('index: ', dataframe2.index)
print('columns: ', dataframe2.columns)

d = [{"price": 4, "count": 2}, {"price": 6, "count": 7}]
dataframe3 = pd.DataFrame(d)
print('dataframe: \n', dataframe3)
print('index: ', dataframe3.index)
print('columns: ', dataframe3.columns)

year = [1950, 1975, 2000, 2018]
population = [2.12, 3.681, 5.312, 6.981]

plt.plot(year, population)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population')
plt.show()

plt.scatter(year, population)
plt.show()

values = [0, 1.2, 1.3, 1.9, 4.3, 2.5, 2.7, 4.3, 1.3, 3.9]
plt.hist(values, 4)
plt.show()

names = 'Tom', 'Richard', 'Harry', 'Jill', 'Meredith', 'George'
speed = [8, 7, 12, 4, 3, 2]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red', 'blue']
explode = (0.1, 0, 0, 0, 0, 0)
plt.pie(speed, explode, names, colors, '%1.1f%%', True, 140)
plt.axis('equal')
plt.show()