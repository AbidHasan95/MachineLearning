#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 00:00:53 2018

@author: sohail
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_x = LabelEncoder()
X[:,3] = label_x.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoid dummy variable trap
X = X[:,1:]

#split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#Fitting Mutiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting results
y_pred = regressor.predict(X_test)

#Building the Optimal model using Backward Elimination
import statsmodels.formula.api as sm
#axis =1 for vertical 0 for horizontal
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()

