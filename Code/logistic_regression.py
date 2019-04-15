# -*- coding: utf-8 -*-

import numpy as np
import preprocessing

df = preprocessing.import_data()
bots = df[1].values[:100,3:].astype(int)
legit = df[2].values[:100,3:].astype(int)


for i in range(bots.shape[0]-1,0,-1):
    for j in range(bots.shape[1]):
        if bots[i][j] == 0:
            bots = np.delete(bots,i,0)

for i in range(legit.shape[0]-1,0,-1):
    for j in range(legit.shape[1]):
        if legit[i][j] == 0:
            legit = np.delete(legit,i,0)
            

X = np.vstack((bots,legit))
X = np.concatenate((np.ones((len(X),1)),X), axis=1)

Y = np.concatenate((np.ones((bots.shape[0],1)),np.zeros((legit.shape[0],1))))


#main function
def LR(x,y,alpha):
    theta = np.zeros(X.shape[1])
    loss = []  #used to store the value of the cost function after each iteration
    while True:
        theta = grad_descent(theta,x,y,alpha)
        loss.append(cost(theta,x,y))
        if len(loss) > 1: #make sure there are at least 2 values in list
            if abs(loss[-1] - loss[-2]) < 1.0e-5: #checking for change in loss
                print("Theta: ",theta)
                print("Accuracy: ",accuracy(theta,x,y))
                return loss
                
    
#returns the value entered run through the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#cost function
def cost(theta,x,y):
    cost = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        cost[i] = np.dot(-y[i],np.log(hypothesis(theta,x,i))) - (np.dot((1 - y[i]),np.log(1 - hypothesis(theta,x,i))))
        #cost[i] = (-y[i]*np.log(hypothesis(theta,x,i))) - ((1-y[i]) * np.log(1-hypothesis(theta,x,i))) # works the same
    return cost.mean()

#hypothesis function, returns a probablity based on sigmoid function
def hypothesis(theta,x,i):
    h = 0
    for j in range(len(theta)):
        h += x[i][j] * theta[j]
    return sigmoid(h)

#calculating the new values for theta using gradient descent
def grad_descent(theta,x,y,alpha):
    for j in range(len(theta)):
        sum = np.zeros(x.shape[0])
        for m in range(len(sum)):
            sum[m] = (hypothesis(theta,x,m)- y[m]) * x[m][j]
        sum = sum.mean()
        theta[j] = theta[j] - (alpha * sum)
    return theta

#calculate the accuracy of the model
def accuracy(theta,x,y):
    pred = np.zeros(x.shape[0])
    correct = np.zeros(x.shape[0])
    for i in range(len(pred)):
        if hypothesis(theta,x,i) < 0.5:
            pred[i] = 0
        else:
            pred[i] = 1
    for i in range(len(correct)):
        if pred[i] == y[i]:
            correct[i] = 1
    correct = list(correct)
    return (correct.count(1) / x.shape[0]) * 100




loss = LR(X,Y,1)
print(len(loss))
loss










