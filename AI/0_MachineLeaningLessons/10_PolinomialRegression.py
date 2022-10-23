# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:36:14 2022

@author: aysegulhatipoglu
"""

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

salary = pd.read_csv('maaslar.csv')


x = salary.iloc[:,1:2]
xDF = x.values
y = salary.iloc[:,2:]
yDF = y.values



#Değerlerimizi polinomial ifade edip sonra lineer regresyona sokuyoruz
poly_reg = PolynomialFeatures(degree = 4)#4.dereceden polinom
x_poly = poly_reg.fit_transform(xDF)

lin_reg = LinearRegression()
lin_reg.fit(x_poly,yDF)#Linear regresyonu fitledik ama veriler lineear değil
plt.scatter(xDF,yDF,color='red')
plt.plot(xDF,lin_reg.predict(x_poly),color='blue')
plt.show()


print(lin_reg.predict(poly_reg.fit_transform([[6.6]])))



