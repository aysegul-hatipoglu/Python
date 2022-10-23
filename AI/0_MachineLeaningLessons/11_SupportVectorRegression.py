# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 23:03:46 2022

@author: aysegulhatipoglu
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVR

salary = pd.read_csv('maaslar.csv')


x = salary.iloc[:,1:2]
xDF = x.values
y = salary.iloc[:,2:]
yDF = y.values


sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(xDF)
sc2 = StandardScaler()
y_olcekli = sc1.fit_transform(yDF)

svr_reg = SVR(kernel = 'rbf') #Gaussian radial basis fonk
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli,color='red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli),color='blue')
plt.show()#Eğer plt.show yapmazsak burdan sonra yapacağımız tüm işlemleri bunun üzerine çizer :)





