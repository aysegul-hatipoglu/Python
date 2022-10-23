# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:53:32 2022

@author: aysegulhatipoglu
"""

import pandas as pd

data = pd.read_csv('missing_value.csv')


boy_kilo = data[['boy','kilo']]
print(boy_kilo)
