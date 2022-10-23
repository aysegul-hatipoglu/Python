# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:59:15 2022

@author: aysegulhatipoglu
"""

import pandas as pd
import re
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

stop = nltk.download('stopword')
ps = PorterStemmer()

comments = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines='skip')
derlem = []

#Preprocessing
for i in range(comments[comments.columns[0]].count()):
    comment = re.sub('[^a-zA-Z]', ' ', comments['Review'][i])
    comment = comment.lower()
    comment = comment.split()#listeye çevirir.
    newComment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    newComment = ' '.join(newComment)
    derlem.append(newComment)

#Feature Extraction-öz nitelik çıkarımı / Bag of Words(BOW)
cv = CountVectorizer(max_features=2000) #En fazla kullanılan 2000 kelimeyi al dedik
X = cv.fit_transform(derlem).toarray() #Kelime vektörünü oluşturduk #bağımsız değişken
y = comments.iloc[:,1:].values#bağımlı

y[np.isnan(y)] = 0


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm) #%72.5 accuracy
