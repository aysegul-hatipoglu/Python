# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 01:13:05 2022

@author: aysegulhatipoglu
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


values = pd.read_csv('values.csv')

height_weight_age = values.iloc[:,1:4].values
gender = values.iloc[:,4:].values
x_train,x_test,y_train,y_test = train_test_split(height_weight_age,gender,test_size=0.33,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

randomForestClassifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
randomForestClassifier.fit(X_train,y_train)

y_pred = randomForestClassifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)