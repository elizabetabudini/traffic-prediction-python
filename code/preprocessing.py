# -*- coding: utf-8 -*-
"""
Author: Elizabeta Budini
"""

from init import *
from util import *

def data_preprocessing(frame):
    # """
    # Checks if dataset has null values.
    # Checks if some link length is equal to 0 and removes those rows
    # Describes the distribution of unique values for some columns
    
    # """
    missing_value=frame.isna()
#    print(missing_value)
    
    #exclude roads with 0 km length
    frame=frame[frame[linkKM] != 0]
    describe=frame.describe()
    #print(describe)
        
    print("\n\n************************************************************")
    print("UNIQUE DISTRIBUTION OF FEATURES")
    
    #find distribution of unique values for region column
    numRegions = len(frame[regionID].unique())
    print("Total number of regions: ", numRegions)
    
    #find distribution of unique values for category column
    numCategories = len(frame[roadCat].unique())
    print("Total number of road categories: ", numCategories)
    
    return frame
    
