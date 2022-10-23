# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:58:05 2022

@author: aysegulhatipoglu
"""

import pandas as pd
from sklearn import preprocessing #scikit-learn

data = pd.read_csv('missing_value.csv')

country = data.iloc[:,0:1].values

le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(data.iloc[:,0]) #sayısal hale çevirdik

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray() #Tek veri haline getirdik