# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:06:37 2022

@author: aysegulhatipoglu
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

values = pd.read_csv('values.csv')

height_weight_age = values.iloc[:,1:4].values
gender = values.iloc[:,4:].values
x_train,x_test,y_train,y_test = train_test_split(height_weight_age,gender,test_size=0.33,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# svc = SVC(kernel='linear')
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
confusionMatrix = confusion_matrix(y_test,y_pred)