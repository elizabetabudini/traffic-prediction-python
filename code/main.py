# -*- coding: utf-8 -*-
"""
Author: Elizabeta Budini
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
# import initialization file
from init import *
from model import *
from visualization import *
from util import *
from preprocessing import *
import pandas as pd

# import dataset
# https://data.gov.uk/dataset/208c0e7b-353f-4e2d-8b7a-1a7118467acc/gb-road-traffic-counts
frame = pd.read_csv(file)

# data understanding, visualisation & pre-processing
frame=data_preprocessing(frame)
frame=frame.drop("ons_code", axis=1)
#visualize in billion vehicle miles
frame.iloc[:, -7:] /= 1e9
describe=frame.describe()
print(describe)



visualize_histogram_by_year(frame)
visualize_regional_stackbar(frame)
visualize_vehicle_type(frame)
visualize_2018(frame)
visualize_distribution(frame)

#print("categorical data:",frame.select_dtypes(include=['object']).copy())
#print("null data:",frame.isnull().sum()
# creating instance of one-hot-encoder
#enc = OneHotEncoder(handle_unknown='ignore')
#enc_df = pd.DataFrame(enc.fit_transform(frame[[regionName]]).toarray())
# merge with main df frame on key values
#frame = frame.join(enc_df)
#frame=frame.dropna()


#regionID and region name have the same information
frame=frame.drop([regionName], axis=1)
frame.iloc[:,:] = StandardScaler().fit_transform(frame.iloc[:,:])


#applying regression models
model_tree(frame)
model_SVR(frame)
model_forest(frame)
#model_forest_hyp(frame)
model_linear(frame)

#frameSMA= movingAverage(frame)
#visualize_averageHistogram(frameSMA, frame)
#fit_model2(frame)
