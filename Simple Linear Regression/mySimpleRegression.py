#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:18:17 2018

@author: sohail
"""


#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=1/3, random_state=0)


#Fitting Simple Linear Regression to the training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict test set Salaries
y_pred = regressor.predict(X_test)

#Visualise the data and results - train set
plt.figure()
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs experience (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Visualise - test set
plt.figure()
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs experience (Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()