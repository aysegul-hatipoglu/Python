# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:39:26 2022

@author: aysegulhatipoglu
"""

#x:bağımsız değişkenler
#y:bağımlı değişkenler

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('missing_value.csv')


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
age = data.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])


country = data.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(data.iloc[:,0]) #sayısal hale çevirdik

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray() #Tek veri haline getirdik

gender = data.iloc[:,-1]

countryResult = pd.DataFrame(data=country, index=range(22),columns = ['fr','tr','us'])
ageResult = pd.DataFrame(data=age, index=range(22),columns = ['boy','kilo','yas'])
genderResult = pd.DataFrame(data=gender, index=range(22),columns = ['cinsiyet'])

result = pd.concat([countryResult,ageResult], axis=1) #Birleştirme işlemi, axis:1 yatay bileştirme/axis=0 dikey birleştirme


#test_size:ne kadarı test için bölünecek geri kalanları train için
#random_state:verinin ne kadar rassal bölüneceği
x_train,x_test,y_train,y_test = train_test_split(result,genderResult,test_size=0.33, random_state=0)


#Ölçekleme yaparak verileri yakın aralıklara getiriyoruz. Birbirlerine göre ölçeklenmiş oldu
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

