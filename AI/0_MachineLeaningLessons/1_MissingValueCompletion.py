# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:07:45 2022

@author: aysegulhatipoglu
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 


data = pd.read_csv('missing_value.csv')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Age = data.iloc[:,1:4].values

imputer = imputer.fit(Age[:,1:4])
Age[:,1:4] = imputer.transform(Age[:,1:4])
print(Age)