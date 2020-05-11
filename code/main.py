# -*- coding: utf-8 -*-
"""
Author: Elizabeta Budini
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# import initialization file
from init import *

# import regresion model functions
from model import *

#import visualization functions
from visualization import *

#import utility functions
from util import *

#import preprocessing functions
from preprocessing import *


# import dataset
# https://data.gov.uk/dataset/208c0e7b-353f-4e2d-8b7a-1a7118467acc/gb-road-traffic-counts
frame = pd.read_csv(file)

##
##         PRE-PROCESSING
##

# data understanding, visualisation & pre-processing
frame=data_preprocessing(frame)
frame=frame.drop("ons_code", axis=1)
#visualize in billion vehicle miles
frame.iloc[:, -7:] /= 1e9
describe=frame.describe()
print(describe)

#print("categorical data:",frame.select_dtypes(include=['object']).copy())
#print("null data:",frame.isnull().sum()

# creating instance of one-hot-encoder
#enc = OneHotEncoder(handle_unknown='ignore')
#enc_df = pd.DataFrame(enc.fit_transform(frame[[regionName]]).toarray())
# merge with main df frame on key values
#frame = frame.join(enc_df)
#frame=frame.dropna()

##
##         VISUALIZATION
##

visualize_by_year(frame)
visualize_regional_stackbar(frame)
visualize_vehicle_type(frame)
visualize_2018(frame)
visualize_distribution(frame)

#regionID and region name have the same information
frame=frame.drop([regionName], axis=1)

##
##         REGRESSION MODELS
##

#decision tree regression
model_tree(frame)

#random forest regression
model_forest(frame)

#random foreste regression with cross validation
model_forest_cross(frame)

#linear regression
model_linear(frame)

#generate synthetic feature and forest regression
frameSynthetic= add_synthetic_features(frame)
frameSynthetic=frameSynthetic.drop([cars,motorcycles,coaches,lorries,vans, pedalCycles, linkKM], axis=1)
model_forest_synthetic(frameSynthetic)

##
##         MODELS COMPARISON
##

# the model functions return (accuracy, error, exec time)
# which are assigned to 3 variables a, b, c
a=model_forest_hyp(frame)
b=model_SVR(frame)
c=model_linear2(frame)

models=pd.DataFrame({
    "Model name":["Random forest", "SVR", "Linear regression"],
    "Accuracy": [a[0], b[0], c[0]],
    "Error":[a[1], b[1], c[1]],
    "Execution time":[a[2], b[2], c[2]]
    })

print(models)
    
