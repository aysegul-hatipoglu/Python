# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 12:57:57 2022

@author: aysegulhatipoglu
"""

from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt

sales = pd.read_csv('maaslar.csv')
level = sales.iloc[:,1:2].values
sale = sales.iloc[:,-1:].values



dtRegressor = DecisionTreeRegressor(random_state=0)
dtRegressor.fit(level,sale)

plt.scatter(level, sale, color='red')
plt.plot(level, dtRegressor.predict(level), color='blue')

z = level + 0.5
k = level - 0.4

#Aşağıda Görüleceği üzere yine farklı değer de gelse ilgili aralığın ortalamasına düşüyor.
plt.plot(level, dtRegressor.predict(z), color='green')
plt.plot(level, dtRegressor.predict(k), color='yellow')
print(dtRegressor.predict([[11]]))
print(dtRegressor.predict([[6.6]]))
