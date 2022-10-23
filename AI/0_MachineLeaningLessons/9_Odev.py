# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 18:05:42 2022

@author: aysegulhatipoglu
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np

tennisData = pd.read_csv('odev_tenis.csv')

#Daha önceki yöntemlerdeki gibi tek tek encode etmeyip tüm datayı encode ederiz
tennisDataEnc = tennisData.apply(preprocessing.LabelEncoder().fit_transform)


outlook = tennisDataEnc.iloc[:,0:1].values
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
outlookDF = pd.DataFrame(data=outlook,index=range(14),columns=['overcast','rainy','sunny'])


temperature = tennisData.iloc[:,1:2].values
temperature = pd.DataFrame(data=temperature,index=range(14),columns=['Sıcaklık'])



windy_play =  tennisDataEnc.iloc[:,3:].values
windy_playDF = pd.DataFrame(data=windy_play,index=range(14),columns=['Rüzgar','Play'])


df1 = pd.concat([outlookDF,temperature],axis=1)
dataFrameInputs = pd.concat([df1,windy_playDF],axis=1)



humidity= tennisData.iloc[:,2].values
humidity = pd.DataFrame(data=humidity,index=range(14), columns=['Nem'])


# Eğitim ve test verilerini ayıralım
x_train,x_test,y_train,y_test = train_test_split(dataFrameInputs,humidity, train_size=0.33,random_state=0)

#Eğitim
lr = LinearRegression()
lr.fit(x_train, y_train)
tahmin = lr.predict(x_test)

