# -*- coding: utf-8 -*-

from sklearn import datasets
from random import seed
from random import randrange
from math import sqrt
import numpy as np
import preprocessing


iris = datasets.load_iris()

X = iris.data
Y = iris.target
dataset = []
#for i in range(len(X)):
#    dataset.append(np.hstack((X[i],Y[i])))

#cleaning up data(picking out useful rows)
df = preprocessing.import_data()
bots = df[1].values[:2000,3:].astype(int)
legit = df[2].values[:2000,3:].astype(int)


#adding label, bots=1 legit=0
bots = np.hstack((bots,np.ones((bots.shape[0],1))))
legit = np.hstack((legit,np.zeros((legit.shape[0],1))))

dataset = np.vstack((bots,legit))

test_bots = df[1].values[2000:3000,3:].astype(int)
test_legit = df[2].values[2000:3000,3:].astype(int)

test_data = np.vstack((test_bots, test_legit))
test_label = np.vstack((np.ones((test_bots.shape[0],1)), np.zeros((test_legit.shape[0],1))))

#References:
#https://www.python-course.eu/Random_Forests.php
#https://machinelearningmastery.com/implement-random-forest-scratch-python/

#split a dataset into k folds
def cv(dataset,folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset)/folds
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

#calculate accuracy
def accuracy_calc(actual, predicted):
    acc = 0.0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            acc += 1.0
    return acc / len(actual) * 100.0

#evaluate the algorithm with cv
def evaluate(dataset, algorithm, folds, *args):
    folds = cv(dataset,folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_calc(actual, predicted)
        scores.append(accuracy)
    return scores


#split the data into two based on a threshold(value)
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left,right

#calculate the gini index
def get_gini(groups, classes):
    instances = float(sum([len(group) for group in groups]))
    gini= 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p*p
        gini += (1.0 - score) * (size / instances)
    return gini

#pick the best split 
def best_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999,999,999,None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = get_gini(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

#create a terminal node value
def terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

#create child splits for a node or make a terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    #check for no splits
    if not left or not right:
        node['left'] = node['right'] = terminal(left + right)
        return
    #check if max depth
    if depth >= max_depth:
        node['left'], node['right'] = terminal(left), terminal(right)
        return
    #go down left child
    if len(left) <= min_size:
        node['left'] = terminal(left)
    else:
        node['left'] = best_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    #go down right child
    if len(right) <= min_size:
        node['right'] = terminal(right)
    else:
        node['right'] = best_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)
        

#build a tree
def build_tree(train, max_depth, min_size, n_features):
    root = best_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


#make a prediction using the tree and nodes
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


#create a subsample for the data (the bagging/bootstrapping process)
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

#make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

#random forest algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        #print(i)
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    testing_predictions = [bagging_predict(trees,row) for row in test_data]
    print('Backup testing:', accuracy_calc(test_label, testing_predictions))
    return predictions

#seed
seed(1314)

dataset = np.random.permutation(dataset)


#print results and stuff
n_folds = 2
max_depth  = 10
min_size = 1
sample_size = 1
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [2,5,10]:
    scores = evaluate(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean accuracy: %.3f%%' %(sum(scores)/float(len(scores))))


