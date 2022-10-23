# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 20:16:02 2022

@author: aysegulhatipoglu
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

values = pd.read_csv('values.csv')


height_weight_age = values.iloc[:,1:4].values#bağımsız değişken
gender = values.iloc[:,4:].values #bağımlı değişken


x_train,x_test,y_train,y_test = train_test_split(height_weight_age,gender,test_size=0.33,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logRegressor = LogisticRegression(random_state=0)
logRegressor.fit(X_train,y_train)

y_pred = logRegressor.predict(X_test)

confusionMatrix = confusion_matrix(y_test,y_pred)
#Bu örnek için 8 örnekte 1 başarı var ama burada o önemli değil confusionMatrix'i anlamak
#outlierları çıkarırsak(çocuk değerleri) daha başarılı sonuç elde ederiz