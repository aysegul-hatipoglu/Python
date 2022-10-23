# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:28:21 2022

@author: aysegulhatipoglu
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('sales.csv')


months = data[['Aylar']]
sales = data[['Satislar']]

#Veriler eğitim ve test verisi olarak bölünmeli
x_train,x_test,y_train,y_test = train_test_split(months,sales,test_size=0.33,random_state=0)

#Verilerin ölçeklenmesi
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test =sc.fit_transform(x_test)

#Lineer regresyon modeli inşası
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#lr.fit(X_train,Y_train)//#ölçeklendirilmiş hali

lr.fit(x_train,y_train)#ölçeklendirilmemiş hali

tahmin = lr.predict(X_test)

#Verileri yukarıda sıralamadan kullandığımız için bu haliyle grefik biraz karmaşık gözüküyor
#plt.plot(x_train,y_train)

#Verileri indexe göre çizdirip sıralayalım
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test)) #Lineer regresyon grafiğini çizer
plt.title('Aylara göre satış')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
