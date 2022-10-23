# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:46:46 2022

@author: aysegulhatipoglu
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

values = pd.read_csv('values.csv')


height_weight_age = values.iloc[:,1:4].values#bağımsız değişken
gender = values.iloc[:,4:].values #bağımlı değişken


x_train,x_test,y_train,y_test = train_test_split(height_weight_age,gender,test_size=0.33,random_state=0)

sc = StandardScaler()

#fit:eğitme, transform:o eğitimi kullanma/uygulama
#fit_transform() ile transform() farkı
#fit_transform.öğrenip ve transform ediyor. transform.
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logRegressor = LogisticRegression(random_state=0)
logRegressor.fit(X_train,y_train)

y_pred = logRegressor.predict(X_test)
print(y_pred)
print(y_test)