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
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import cross_val_score
import time
from sklearn.model_selection import RandomizedSearchCV



def model_tree(frame):
    # """
    # ML model applying Decision Tree Regression
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """
    frame.iloc[:,:-1] = StandardScaler().fit_transform(frame.iloc[:,:-1])

    print("\n\n************************************************************")
    print("MODEL Decision tree regression")
    y=frame.iloc[:,-1] #target
    x=filterFeatures(frame) #features
    print("features=", x.columns)
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
    
    
    #apply regression model
    model_tree= DecisionTreeRegressor()
    t0 = time.time()
    model_tree.fit(X_train, y_train)
    execution = time.time() - t0
    y_pred = model_tree.predict(X_test)
    l_reg = model_tree.score(X_test, y_test)
    
    #show results of regression model
    print("score: %.4f" % l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: %.4f" % regression_error)
    return l_reg, regression_error, execution
    
def model_forest(frame):
    # """
    # ML model applying Random Forest Regression to filtered features
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """
    frame.iloc[:,:-1] = StandardScaler().fit_transform(frame.iloc[:,:-1])

    print("\n\n************************************************************")
    print("MODEL Random forest regression")
    y=frame.iloc[:,-1] #target
    x=filterFeatures(frame) #features
    print("features=", x.columns)
    
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
    
    #apply regression model
    model_forest= RandomForestRegressor()
    t0 = time.time()
    model_forest.fit(X_train, y_train)
    execution = time.time() - t0
    y_pred = model_forest.predict(X_test)
    l_reg = model_forest.score(X_test, y_test)
    
    #show results of regression model
    print("score model_forest: %.4f" % l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: %.4f" % regression_error)
    return l_reg, regression_error, execution
    
def model_forest_cross(frame):
    # """
    # ML model applying Random Forest Regression to filtered features
    # using cross validation
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """
    print("\n\n************************************************************")
    print("MODEL Random forest regression with cross validation")
    y=frame.iloc[:,-1] #target
    X=filterFeatures(frame) #features
    print("features=", X.columns)
    #apply regression model
    model_forest= RandomForestRegressor()
    t0 = time.time()
    model_forest.fit(X, y)
    execution = time.time() - t0
    y_pred=model_forest.predict(X)
    scores= cross_val_score(model_forest, X, y, cv=10)
    #show results of regression model
    print("cross validation scores {}".format(scores))
    print("Average cross validation score: {:.4f}".format(scores.mean()))
    regression_error = mean_absolute_error(y, y_pred)
    print("regression_error: %.4f" % regression_error)
    return scores, regression_error, execution
  

def model_SVR(frame):
    # """
    # ML model applying Support Vector Regression to all features
    # using GridSearch to tune the model hyperparameters
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """
    frame.iloc[:,:-1] = StandardScaler().fit_transform(frame.iloc[:,:-1])
    y=frame.iloc[:,-1] #target
    #x=frame.iloc[:, :-2]
    x=filterFeatures(frame) #features
    print("\n\n************************************************************")
    print("MODEL SV Regression")
    print("features=", x.columns)

    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
   
    # model_SVR = GridSearchCV(SVR(gamma=0.1),
    #                param_grid={"C": [1e0, 1e1, 1e2, 1e3],
    #                            "gamma": np.logspace(-2, 2, 5)})
    
    # model_SVR = RandomizedSearchCV(SVR(gamma=0.1),
    #                param_distributions={"C": [1e0, 1e1, 1e2, 1e3],
    #                            "gamma": np.logspace(-2, 2, 5)})
    
    model_SVR= GridSearchCV(SVR(gamma=0.1),
                    param_grid={"gamma": np.logspace(-2, 2, 5)})
    
    #apply regression model
    t0 = time.time()
    model_SVR.fit(X_train, y_train)
    execution = time.time() - t0
    
    y_pred = model_SVR.predict(X_test)
    l_reg = model_SVR.score(X_test, y_test)
    
    #show results of regression model
    print("score: %.4f" % l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: %.4f" % regression_error)
    print("Execution time: ", execution)
    return l_reg, regression_error, execution

    
    
def model_linear(frame):
    # """
    # ML model applying Linear Regression to linearly correlated features
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    # """
    print("\n\n************************************************************")
    print("MODEL Linear regression 1")
    
    #print("Correlation with target variable")
    
    correlation=frame.corr()[allV].sort_values(ascending=False)
    #print(correlation)

    y=frame.iloc[:,-1] #target
    x=frame.iloc[:, -7:-2] #linearly correlated features (vehicle types)
    print("features=", x.columns)
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
    
    #apply regression model
    model_linear= LinearRegression(copy_X=True)
    t0 = time.time()
    model_linear.fit(X_train, y_train)
    execution = time.time() - t0
    y_pred = model_linear.predict(X_test)
    l_reg = model_linear.score(X_test, y_test)
    
    #show results of regression model
    print("score linear: %.4f" % l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: %.4f" % regression_error)
    return l_reg, regression_error, execution


def model_linear2(frame):
    # """
    # ML model applying Linear Regression to all features
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """
    y2=frame.iloc[:,-1] #target
    x2=frame.iloc[:, :-7]
    print("\n\n************************************************************")
    print("MODEL Linear regression 2")
    print("features=", x2.columns)
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=validation_size)
    
    #apply regression model
    model_linear2= LinearRegression(copy_X=True)
    t0 = time.time()
    model_linear2.fit(X_train, y_train)
    execution = time.time() - t0
    y_pred = model_linear2.predict(X_test)
    l_reg = model_linear2.score(X_test, y_test)
    
    #show results of regression model
    print("score linear2: %.4f" % l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: %.4f" % regression_error)
    return l_reg, regression_error, execution
    
    

def model_linear3(frame):
    # """
    # ML model applying Linear Regression to all features using RANSAC.
    # The outliers influence significantly linear regression, RANSAC will
    # select only inliers when fitting the model.
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """
    print("\n\n************************************************************")
    print("MODEL Linear regression ransac")
    
    y=frame.iloc[:,-1] #target
    x=frame.iloc[:, :-7]
    print("features=", x.columns)
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
    
    
    ransac = RANSACRegressor(LinearRegression(), 
                             max_trials=100, 
                             min_samples=50, 
                             loss='absolute_loss', 
                             residual_threshold=5.0, 
                             random_state=0)
    t0 = time.time()
    ransac.fit(X_train, y_train)
    execution = time.time() - t0
    y_pred = ransac.predict(X_test)
    l_reg = ransac.score(X_test, y_test)
    
    #show results of regression model
    print("score linear ransac: %.4f" % l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: %.4f" % regression_error)
    return l_reg, regression_error, execution

def model_forest_synthetic(frame):
    # """
    # ML model applying Random Forest Regression to the synthetic traffic 
    # dataset passed as input 
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """

    print("\n\n************************************************************")
    print("MODEL Random forest regression with synthetic features")
    y=frame.loc[:,allV] #target
    x=frame.loc[:,["min", "max", "std", "SMA_5", year, roadCat]]
    print("features=", x.columns)
    
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
    
    #apply regression model
    model_forest= RandomForestRegressor()
    t0 = time.time()
    model_forest.fit(X_train, y_train)
    execution = time.time() - t0
    y_pred = model_forest.predict(X_test)
    l_reg = model_forest.score(X_test, y_test)
    
    #show results of regression model
    print("score model_forest: %.4f" % l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: %.4f" % regression_error)
    return l_reg, regression_error, execution

from bayes_opt.util import Colours

def model_forest_hyp(frame):
    # """
    # ML model applying Random Forest Regression with GridSearchCV to tune 
    # the model hyperparameters
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """
    frame.iloc[:,:-1] = StandardScaler().fit_transform(frame.iloc[:,:-1])

    
    print("\n\n************************************************************")
    print("MODEL Random forest regression hyp")
    
    y=frame.iloc[:,-1] #target
    x=filterFeatures(frame) #features
    print("features=", x.columns)
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
    
    n_test=[100]
    params_dict={'n_estimators':n_test, 'n_jobs':[-1], 'max_features':["auto", "sqrt", "log2"]}
    
    model_forest_hyp=GridSearchCV(estimator=RandomForestRegressor(), param_grid=params_dict, scoring='r2')

    #apply regression model
    t0 = time.time()
    search = model_forest_hyp.fit(X_train, y_train)
    execution = time.time() - t0
    print(search.best_params_)
    y_pred = model_forest_hyp.predict(X_test)
    l_reg = model_forest_hyp.score(X_test, y_test)
    
    #show results of regression model
    print("score model_forest_hyp: %.4f" % l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: %.4f" % regression_error)
    print("Execution time: ", execution)
    return l_reg, regression_error, execution
    


def model_forest_hyp_random(frame):
    # """
    # ML model applying Random Forest Regression with RandomizedSearchCV
    # to tune the model hyperparameters
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """
    frame.iloc[:,:-1] = StandardScaler().fit_transform(frame.iloc[:,:-1])

    
    print("\n\n************************************************************")
    print("MODEL Random forest regression hyp randomCV")
    
    y=frame.iloc[:,-1] #target
    x=filterFeatures(frame) #features
    print("features=", x.columns)
    
    #split train and test set
    validation_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
    
    n_test=[100]
    params_dict={'n_estimators':n_test, 'n_jobs':[-1], 'max_features':["auto", "sqrt", "log2"]}
    
    model_forest_hyp=RandomizedSearchCV(estimator=RandomForestRegressor(), 
          param_distributions=params_dict, scoring='r2')

    #apply regression model
    t0 = time.time()
    search = model_forest_hyp.fit(X_train, y_train)
    execution = time.time() - t0
    print(search.best_params_)
    y_pred = model_forest_hyp.predict(X_test)
    l_reg = model_forest_hyp.score(X_test, y_test)
    
    #show results of regression model
    print("score model_forest_hyp: %.4f" % l_reg)
    regression_error = mean_absolute_error(y_test, y_pred)
    print("regression_error: %.4f" % regression_error)
    print("Execution time: ", execution)
    return l_reg, regression_error, execution

def bayesian_random_forest(frame):
    # """
    # ML model applying Random Forest Regression with Bayesian optimization
    # to tune the model hyperparameters
    
    # input: Pandas dataframe to use in modelling
    # output: (score, error, execution time)
    
    # """
    # using https://github.com/fmfn/BayesianOptimization
    frame.iloc[:,:-1] = StandardScaler().fit_transform(frame.iloc[:,:-1])
    print("\n\n************************************************************")
    print("MODEL Random forest regression Bayesian opt")
    
    y=frame.iloc[:,-1] #target
    x=filterFeatures(frame) #features

    print(Colours.green("--- Optimizing Random Forest ---"))
    
    t0 = time.time()
    optimize_rfr(x, y)
    execution = time.time() - t0
    print("Execution time: ", execution)
    