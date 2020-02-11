# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:24:38 2020

@author: Utente
"""

from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn import datasets
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# import some data to play with
iris = datasets.load_iris()

clf = tree.DecisionTreeClassifier()

clf = clf.fit( iris.data, iris.target)

with open("iris.dot", "w") as f:
    f = tree.export_graphviz(clf, out_file=f)
    
    dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_pdf("iris.pdf")