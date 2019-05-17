# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot
import numpy as np
import preprocessing
from math import sqrt


#1 = bots, 0 = legit
#training data
df = preprocessing.import_data()
train_bots = df[1].values[:12000,3:].astype(int)
train_legit = df[2].values[:12000,3:].astype(int)

feature_list = df[3][3:]

X_train = np.vstack((train_bots,train_legit))

#testing data
test_bots = df[1].values[15000:16000,3:].astype(int)
test_legit = df[2].values[15000:16000,3:].astype(int)

X_test = np.vstack((test_bots,test_legit))

#training labels
train_bots_label = np.ones((train_bots.shape[0],1))
train_legit_label = np.zeros((train_legit.shape[0],1))

Y_train = np.vstack((train_bots_label,train_legit_label))
Y_train = Y_train.ravel(order='C')

#testing labels
test_bots_label = np.ones((test_bots.shape[0],1))
test_legit_label = np.zeros((test_legit.shape[0],1))

Y_test = np.vstack((test_bots_label,test_legit_label))
Y_test = Y_test.ravel(order='C')
a = list()
#random forest algorithm using sklearn
def random_forest(n_trees, max_depth, n_features):
    rf = RandomForestClassifier(n_estimators = n_trees, max_depth = max_depth, max_features = n_features, bootstrap = True)
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_test)
    save_tree(rf)
    return predictions

#calculate accuracy for predictions
def accuracy(predictions):
#    for i in range(len(predictions)):
#        if predictions[i] < 0.5:
#            predictions[i] = 0.0
#        else:
#            predictions[i] = 1.0
    accuracy = 0.0
    for i in range(len(predictions)):
        if predictions[i] == Y_test[i]:
            accuracy += 1.0
    accuracy = accuracy / len(predictions) * 100
    print("Accuracy: ",accuracy, "%")
    return
#reference for visualisation
#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

#Visualise a tree
def save_tree(rf):
    tree = rf.estimators_[1]
    export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')
    return


max_depth = 10
n_features = int(sqrt(len(X_test[0])))

for n_trees in [2,5,10,50,100,500,1000]:
    print("Trees: ",n_trees)
    predictions = random_forest(n_trees, max_depth, n_features)
    accuracy(predictions)

