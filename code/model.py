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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def model_tree(frame):
    print("\n\n************************************************************")
    print("MODEL Decision tree regression")
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
    
def model_forest(frame):
    print("\n\n************************************************************")
    print("MODEL Random forest regression")
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
    
def model_forest_hyp(frame):
    print("\n\n************************************************************")
    print("MODEL Random forest regression hyp")
    y=frame.iloc[:,-1] #target
    x=independentFeatures(frame) #features
    print("features=", x.columns)
    
    n_test=[100]
    params_dict={'n_estimators':n_test, 'n_jobs':[-1], 'max_features': ["auto", "sqrt", "log2"]}
    
    regressor=GridSearchCV(estimator=RandomForestRegressor(), param_grid=params_dict, scoring='r2')
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
    
    #apply regression model
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
    print("MODEL SVRegression")
    print("features=", x.columns)
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
   
    
    regressor= SVR()
    
    #apply regression model
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    l_reg = regressor.score(X_test, y_test)
    
    #show results of regression model
    print("score: ", l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: ",regression_error)

def model_SVR_all(frame):
    y=frame.iloc[:,-1] #target
    x=frame.iloc[:,0:-1] #features
    print("\n\n************************************************************")
    print("MODEL 4 SVRegression with all features")
    print("features=", x.columns)

    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
   
    regressor= SVR()
    
    #apply regression model
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    l_reg = regressor.score(X_test, y_test)
    
    #show results of regression model
    print("score: ", l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: ",regression_error)