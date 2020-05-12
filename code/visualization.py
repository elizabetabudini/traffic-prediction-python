# -*- coding: utf-8 -*-
"""
Author: Elizabeta Budini
"""
from util import *
import pandas as pd
import numpy as np
from init import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
from prettytable import PrettyTable
from tabulate import tabulate

def visualize_by_year(frame):
    
    #Adapted from : https://python-graph-gallery.com/123-highlight-a-line-in-line-plot/
    
    # """
    # Plots a multiple line plot for the vehicles mile travled 
    # by selected vehicle type over the years.
    # Five different lines will be plotted, one for each vehicle type
    
    # x = years (1993-2018)
    # y = billion of vehicle miles traveled
    
    # """
    plt.figure(1)
    nFrame=frame.copy()
    nFrame.iloc[:,-7:] = MinMaxScaler().fit_transform(nFrame.iloc[:,-7:])

    nFrame=nFrame.groupby(year).sum().reset_index()

    print("\n\n************************************************************")
    print("GROUP DISTRIBUTION OF TRAFFIC BY YEAR")
    print(nFrame)
    # Make a data frame
    df=pd.DataFrame({'x': nFrame[year], 
                     'vans': nFrame[vans],
                     'cars_and_taxis': nFrame[cars],
                     
                     
                     'two_wheeled_motor_vehicles': nFrame[motorcycles], 
                     'lorries': nFrame[lorries],
                     'buses_and_coaches': nFrame[coaches]
                     
                      })
     
    # style
    plt.style.use('seaborn-darkgrid')
    #determine which type of vehicle has the min number across all types in 1993
    baseCol = df.idxmin(axis=1) #Get Column name
    base = df[baseCol[0]].min() #get first year min value
    
    #Then for each other type, get the different between 
    #the number in 1993 for this type and the base, 
    #then subtract the result from each number in all years.
    for column in df.drop('x', axis=1):
        diff = df[column].iloc[0]-base
        df[column] -= diff
     
     
    # get colormap instance
    colormap = plt.get_cmap('Set1')
    # multiple line plot
    num=0
    for column in df.drop('x', axis=1):
        num+=1
        plt.plot(df['x'], df[column], marker='', linewidth=2, alpha=0.9, label=column, color=colormap(num))
    
    # Add legend
    plt.legend(loc=2, ncol=1)
     
    # Add titles
    plt.title("Index of vehicle miles by vehicle type, 1993 - 2018", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Year")
    plt.ylabel("Billion vehicle miles")
    plt.savefig('img/by_year.png', bbox_inches='tight')
    
def visualize_regional_stackbar(frame):
    # """
    # Plots a vertical stack bar plot for types of vehicle used in 
    # different regions in 2018
    
    # x = regions (11)
    # y = billion of vehicle miles traveled
    
    # """
     plt.figure(2)
     #regional traffic in 2018
     nFrame= frame[frame["year"]==2018]
     
     #group by region name
     nFrame=nFrame.groupby(regionName).sum().sort_values(by=[allV],ascending=True).reset_index()
     sum_column = nFrame[coaches] + nFrame[motorcycles]
     nFrame["other_vehicles"] = sum_column
     
     print("\n\n************************************************************")
     print("TRAFFIC BY REGION IN 2018")
     print(nFrame)
     
     nFrame.plot(regionName,[vans,lorries,cars, "other_vehicles"],kind = 'bar', stacked=True)
     plt.title("Traffic by vehicle types in each region in 2018", loc='left', fontsize=12, fontweight=0, color='orange')
     plt.xlabel("Region")
     plt.ylabel("Billion vehicle miles")
     plt.savefig('img/regional.png', bbox_inches='tight')
     
def visualize_vehicle_type(frame):
    # """
    # Plots a horizontal bar plot to compare traffic composition in
    # 1993 and 2018. It shows how different types of vehicles contribute
    # to the total traffic.
    
    # x = billion of vehicle miles traveled
    # y = years (1993 and 2018)
    
    # """
    plt.figure(3)
    #group by region name
    nFrame=frame.groupby(year).sum().sort_values(by=[allV],ascending=True).reset_index()
 
    nFrame= nFrame[(nFrame[year]==2018) | (nFrame[year]==1993)]
    sum_column = nFrame[coaches] + nFrame[motorcycles]
    nFrame["other_vehicles"] = sum_column
    
    print("\n\n************************************************************")
    print("VEHICLE TYPE COMPARISON")
    print(nFrame)
    
    nFrame.plot(year,[vans,lorries,cars, "other_vehicles"],kind = 'barh', stacked=True, figsize=[9,3])
    plt.title("Traffic by vehicle types comparison in 1993 and 2018", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.ylabel("Year")
    plt.xlabel("Billion vehicle miles")
    plt.legend(loc=4)
    plt.savefig('img/type.png', bbox_inches='tight')
     
def visualize_2018(frame):
   # """
   #  Prints in the console a pretty table containing total 
   #  amount of vehicle miles travelled for each vehicle in 2018
    
   #  """
   nFrame=frame.copy()
   nFrame= nFrame[nFrame[year]==2018]
   nFrame=nFrame.groupby(year).sum().reset_index()
   df=nFrame.iloc[:, -7:].transpose()
   df=df.sort_values(by=[0],ascending=False)

   print(tabulate(df, headers=['Vehicle type', "Billion vehicle miles"], tablefmt='psql'))

def visualize_distribution(frame):
    # """
    # Plots pairwise relationships in the dataset to find
    # distribution patterns and outliers 
    
    # """
    
    plt.figure(8)
    cols = [allV, cars, vans, motorcycles]
    sns.pairplot(frame[cols], height=2.5)
    plt.tight_layout() 
    plt.show()
    
    plt.figure(9)
    cols = [allV, regionID, linkKM, roadCat]
    sns.pairplot(frame[cols], height=3)
    plt.tight_layout() 
    plt.show()


    
    



