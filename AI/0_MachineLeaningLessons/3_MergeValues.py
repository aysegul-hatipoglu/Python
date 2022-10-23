# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:15:11 2022

@author: aysegulhatipoglu
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing #scikit-learn
from sklearn.impute import SimpleImputer

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
result = pd.concat([result, genderResult], axis=1)

