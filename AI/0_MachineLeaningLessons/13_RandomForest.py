# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:08:43 2022

@author: aysegulhatipoglu
"""

#Random Forest Decision Treeden daha başarılı sonuçlar verecektir.
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

sales = pd.read_csv('maaslar.csv')
level = sales.iloc[:,1:2].values
sale = sales.iloc[:,-1:].values


rfRegressor = RandomForestRegressor(random_state=0, n_estimators=10)#n_estimators:kaç tane decisiontree çizeceğimizdir.
rfRegressor.fit(level,sale.ravel())

print(rfRegressor.predict([[6.6]]))

plt.scatter(level,sale,color='red')

z = level + 0.5
k = level - 0.4

plt.plot(level, rfRegressor.predict(level),color='blue')
plt.plot(level, rfRegressor.predict(z),color='green')
plt.plot(level, rfRegressor.predict(k),color='yellow')