# -*- coding: utf-8 -*-
"""
Author: Elizabeta Budini
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from init import *
from util import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def model_tree(frame):
    print("\n\n************************************************************")
    print("MODEL 1 Decision tree regression")
    y=frame.iloc[:,-1] #target
    x=independentFeatures(frame) #features
    print("features=", x.columns)
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
    
    #apply regression model
    regressor= DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    l_reg = regressor.score(X_test, y_test)
    
    #show results of regression model
    print("score: ", l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: ",regression_error)

def model_SVR(frame):
    y=frame.iloc[:,-1] #target
    x=independentFeatures(frame) #features
    print("\n\n************************************************************")
    print("MODEL 2 SVRegression")
    print("features=", x.columns)

     
    #fitting the SVR to the dataset
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(x, y)
    
    y_pred = regressor.predict(x)
    l_reg = regressor.score(x, y)
    
    #show results of regression model
    print("score: ", l_reg)
    regression_error = mean_absolute_error(y, y_pred)
    print("regression_error: ",regression_error)
    