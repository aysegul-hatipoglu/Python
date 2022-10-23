# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 00:38:27 2022

@author: aysegulhatipoglu
"""


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

values = pd.read_csv('values.csv')

height_weight_age = values.iloc[:,1:4].values
gender = values.iloc[:,4:].values
x_train,x_test,y_train,y_test = train_test_split(height_weight_age,gender,test_size=0.33,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

decisionTreeClassifier = DecisionTreeClassifier(criterion='entropy')#default entropy:'gini'
decisionTreeClassifier.fit(X_train,y_train)
y_pred = decisionTreeClassifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('DTC')
print(cm)


