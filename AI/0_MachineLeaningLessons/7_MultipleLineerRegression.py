# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:41:06 2022

@author: aysegulhatipoglu
"""


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

values = pd.read_csv('values.csv')

height_weight_age = values.iloc[:,1:4].values

#Ülkeleri Sayısal hale getirip, sonrasında tek veri haline getiriyoruz.
country = values.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(values.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()

#Cinsiyeti Sayısal hale getirip, sonrasında tek veri haline getiriyoruz.
gender = values.iloc[:,-1:].values
le = preprocessing.LabelEncoder()
gender[:,-1] = le.fit_transform(values.iloc[:,-1])

ohe = preprocessing.OneHotEncoder()
gender = ohe.fit_transform(gender).toarray()


country = pd.DataFrame(data=country, index=range(22), columns=['fr','tr','us'])
height_weight_age = pd.DataFrame(data=height_weight_age, index=range(22), columns=['boy','kilo','yas'])

# 0:1 - :1 :ikiside aynıdır. 0dan 1e kadar olan kısmı al demek
gender = pd.DataFrame(data=gender[:,:1], index=range(22), columns=['cinsiyet'])


#Şimdi tüm dataframeleri birleştirelim
inputData = pd.concat([country,height_weight_age],axis=1)
allDatas = pd.concat([country,height_weight_age,gender],axis=1)

x_train,x_test,y_train,y_test = train_test_split(inputData,gender,test_size=0.33,random_state=0)



regressor = LinearRegression()
regressor.fit(x_train,y_train)#x_train'den y_train'i öğren diyoruz. 6boyutlu veri kullanılıyor

y_pred = regressor.predict(x_test)#Test verisini kullanarak tahmin ettiriyoruz.



#Şimdide boy kolonunu tahmin etmeye çalışalım. Bunun için veriyi uygun hale getirmeliyiz.
height = allDatas.iloc[:,3:4].values #values dersek dizi demezsek DF olur
left_height = allDatas.iloc[:,:3]
rigth_height = allDatas.iloc[:,4:]

#Boyun sol ve sağındaki değerleri birleştirip eğitim verisi oluştup boyu tahmin etmeye çalışacağız
data = pd.concat([left_height,rigth_height],axis=1)

x_train,x_test,y_train,y_test = train_test_split(data,height,test_size=0.33,random_state=0)
r2=LinearRegression()
r2.fit(x_train, y_train)
pred2 = r2.predict(x_test)



