# -*- coding: utf-8 -*-
"""
Author: Elizabeta Budini
"""
#VMT is calculated by multiplying the amount of daily traffic on a roadway segment by
#the length of the segment, then summing all the segments’ VMT to give you a total for
#the geographical area of concern. 

import pandas as pd
from init import *
import numpy as np
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor as RFR
from bayes_opt import BayesianOptimization

def add_synthetic_features(frame):
    nFrame=frame.groupby([year, roadCat]).sum().sort_values([roadCat,year], ascending=True).reset_index()
    categories = nFrame[roadCat].unique()
    
    newFrame=pd.DataFrame(columns=nFrame.columns)
    frames=[]
    
    for cat in categories:
        categorySlice=nFrame[nFrame[roadCat] == cat]
        window=categorySlice.iloc[:,11].rolling(window=5)
        categorySlice['SMA_5'] = window.mean()
        categorySlice['min'] = window.min()
        categorySlice['max'] = window.max()
        categorySlice['std'] = window.std()
 
        frames.append(categorySlice)
   
    newFrame = pd.concat(frames)
    newFrame.fillna( method ='bfill', inplace = True)
    newFrame=newFrame.drop([regionID], axis=1)
    return newFrame

def filterFeatures(frame):
    # """
    # Filters out features that are highly correlated with each other
    
    # input: Pandas dataframe to filter
    # output: filtered dataframe
    
    # """
    corr_matrix = frame.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with high correlation
    to_drop = [column for column in upper.columns if any(upper[column] > 0.50)]
    x=frame.drop(frame[to_drop], axis=1) #features
    return x


def rfr_cv(n_estimators, max_features, data, targets):
# using https://github.com/fmfn/BayesianOptimization
    estimator = RFR(
        n_estimators=n_estimators,
        max_features=max_features,
    )
    cval = cross_val_score(estimator, data, targets,
                           scoring='r2', cv=4)
    return cval.mean()

def optimize_rfr(data, targets):
    #using https://github.com/fmfn/BayesianOptimization
    
    # Apply Bayesian Optimization to Random Forest parameters.
    
    
    # """
    def rfr_crossval(n_estimators, max_features):
        return rfr_cv(
            n_estimators=int(n_estimators),
            max_features=max_features,
            data=data,
            targets=targets,
        )
    optimizer = BayesianOptimization(
        f=rfr_crossval,
        pbounds={
            "n_estimators": (10, 100),
            "max_features": (0.1, 0.999),
        },
        random_state=1234,
    )
    optimizer.maximize(n_iter=1)

    print("Final result:", optimizer.max)
