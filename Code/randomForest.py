# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from random import seed
from random import randrange
import math

iris = datasets.load_iris()

X = iris.data
Y = iris.target



def random_forest(X,Y,n_trees):
    #for i in range(n_trees):
        #create bagging(bootstrapping) from original dataset
        #build trees
        #predictions
    return predictions
    
    

#create a sample with replacement for bagging
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

#calculate the gini value
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini



#find where to split the data
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}


#split the data with a specific attribute(index)
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right



#build the tree and return the root
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root



#splitting children and stuff
#(recursive stuff)
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
#	if depth >= max_depth:
#		node['left'], node['right'] = to_terminal(left), to_terminal(right)
#		return
	#left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	#right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)


#terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

#prediction stuff
#def predict(node, row):
#	if row[node['index']] < node['value']:
#		if isinstance(node['left'], dict):
#			return predict(node['left'], row)
#		else:
#			return node['left']
#	else:
#		if isinstance(node['right'], dict):
#			return predict(node['right'], row)
#		else:
#			return node['right']
        
        
        
        
