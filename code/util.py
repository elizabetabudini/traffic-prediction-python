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

def movingAverage(frame):
    nFrame=frame.groupby([year, roadCat]).sum().sort_values([roadCat,year], ascending=True).reset_index()
    categories = nFrame[roadCat].unique()
    
    newFrame=pd.DataFrame(columns=nFrame.columns)
    frames=[]
    
    for cat in categories:
        categorySlice=nFrame[nFrame[roadCat] == cat]
        categorySlice['SMA_5'] = categorySlice.iloc[:,11].rolling(window=5).mean()
 
        frames.append(categorySlice)
   
    newFrame = pd.concat(frames)
    newFrame['SMA_5'].fillna( method ='bfill', inplace = True)
    newFrame=newFrame.drop([regionID], axis=1)
    return newFrame

def independentFeatures(frame):
    #the independent variables need to be uncorrelated with each other. 
    #Correlation with output variable
    corr_matrix = frame.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with high correlation
    to_drop = [column for column in upper.columns if any(upper[column] > 0.50)]
    
    
    x=frame.drop(frame[to_drop], axis=1) #features
    #print("to drop= ", to_drop)
    #print("features= ", x.columns)
    #print("target= ", y)
    
    
    return x