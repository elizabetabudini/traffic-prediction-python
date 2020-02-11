# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:24:38 2020

@author: Utente
"""

#https://www.youtube.com/watch?v=9lMwjk8jE48

from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn import datasets
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import pandas as pd
from sklearn.model_selection import train_test_split

# import some data to play with
iris = pd.read_csv("C:/Users/Utente/Google Drive/University/DataScience/iris.csv")
#print (iris.head())
#print (iris.describe())
#print (iris.corr())

#target è la classe (quindi le 3 varietà di fiore che nel nostro file
#sono salvate come 0,1,2)
features = iris.iloc[:,1:5] 
target = iris.iloc[:,0]
#print(features)
#print(target)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 2)

clf = tree.DecisionTreeClassifier()

clf = clf.fit( features_train, target_train)

with open("iris.dot", "w") as f:
    f = tree.export_graphviz(clf, out_file=f)
    
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_pdf("iris.pdf")

clf.predict(iris.iloc[:1,:])
array([0])