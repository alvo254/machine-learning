#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 03:24:17 2019

@author: alvo
"""
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv('/home/alvo/Desktop/Salary.csv')


X = data.iloc[:, 1:2].values
y = data.iloc[:, 2:].values



# splitting the dataset
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# print(X_train)

# feature scaling

"""sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting the model
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X, y)

# predicting the new results
y_pred = reg.predict(np.array([[6.5]]))
print(y_pred)

# visualizing the model

plt.scatter(X, y, color='red')
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, reg.predict(X_grid), color='blue')
plt.title('Candle')
plt.xlabel('position label')
plt.ylabel('Sal')

plt.show()
