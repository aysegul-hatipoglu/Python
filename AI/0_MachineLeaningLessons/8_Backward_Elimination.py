# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 12:51:08 2022

@author: aysegulhatipoglu
"""

#Bu kütüphane kullanılarak model ve modeldeki değişkenlerin başarısı ile ilgili bir sistem kurulabilir
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import preprocessing


values = pd.read_csv('values.csv')

height_weight_age = values.iloc[:,1:4].values
country = values.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(values.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
gender = values.iloc[:,-1:].values
le = preprocessing.LabelEncoder()
gender[:,-1] = le.fit_transform(values.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
gender = ohe.fit_transform(gender).toarray()
country = pd.DataFrame(data=country, index=range(22), columns=['fr','tr','us'])
height_weight_age = pd.DataFrame(data=height_weight_age, index=range(22), columns=['boy','kilo','yas'])
gender = pd.DataFrame(data=gender[:,:1], index=range(22), columns=['cinsiyet'])
allDatas = pd.concat([country,height_weight_age,gender],axis=1)
height = allDatas.iloc[:,3:4].values
left_height = allDatas.iloc[:,:3]
rigth_height = allDatas.iloc[:,4:]
height = allDatas.iloc[:,3:4].values
data = pd.concat([left_height,rigth_height],axis=1)

#Bir dizi oluşturup dizeye tüm değişkenleri başlangıçtaatayacağız.
#Ve daha az atkileyenleri eleyerek gideceğiz
#22 tane 1den oluşan bir dizi oluşturduk. Çünkü Formülde bulunan sabit değeri eklemek gerekiyordu.
X = np.append(arr = np.ones((22,1)).astype(int), values=data, axis=1)
X_list = data.iloc[:,[0,1,2,3,4,5]].values
X_list = np.array(X_list,dtype=float)

#istatistiksel değerleri çıkarmamıza yarıyor. Bağımlı değişken ve analiz yapmak istediğimiz bağımsız değişkenlerden oluşan diziyi veriyoruz.
model = sm.OLS(height, X_list).fit()
#Amacımız en yüksek p değerine sahip olanı elemek p ne kadar düşükse bizim için o kadar iyidir.
print(model.summary())



X_list = data.iloc[:,[0,1,2,3,5]].values
X_list = np.array(X_list,dtype=float)
model = sm.OLS(height, X_list).fit()
print(model.summary())



X_list = data.iloc[:,[0,1,2,3]].values
X_list = np.array(X_list,dtype=float)
model = sm.OLS(height, X_list).fit()
print(model.summary())












