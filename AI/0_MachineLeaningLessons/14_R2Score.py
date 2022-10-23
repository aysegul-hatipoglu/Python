# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:15:07 2022

@author: aysegulhatipoglu
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pandas as pd


sales = pd.read_csv('maaslar.csv')
level = sales.iloc[:,1:2].values
sale = sales.iloc[:,-1:].values

#Random Forest Regression
rfRegressor = RandomForestRegressor(random_state=0, n_estimators=10)#n_estimators:kaç tane decisiontree çizeceğimizdir.
rfRegressor.fit(level,sale.ravel())

#DecisionTree Regression
dtRegressor = DecisionTreeRegressor(random_state=0)
dtRegressor.fit(level,sale)

#Polynomial Regression
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(level)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,sale)


#Linear Regression
lin_reg2 = LinearRegression()
lin_reg2.fit(level,sale)


#SVR
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(level)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(sale)
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)




print('Random Forest r2 değeri')
#Gerçek ve tahmin değerleri arasındaki bağlantıyı bulur, 1e yaklaşması bizim için iyi birşey
print(r2_score(sale,rfRegressor.predict(level)))

print('---------------------------------------')
print('DecisionTree r2 değeri')#aynı değeri vereceğinden 1 çıkar. Detayı bilmesek en iyi yöntem bu diye düşünerek hataya düşeriz
print(r2_score(sale,dtRegressor.predict(level)))

print('---------------------------------------')
print('Polynomial Regression r2 değeri')
print(r2_score(sale,lin_reg.predict(x_poly)))

print('---------------------------------------')
print('Linear Regression r2 değeri')
print(r2_score(sale,lin_reg2.predict(level)))

print('---------------------------------------')
print('SVR r2 değeri')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

