# -*- coding: utf-8 -*-

import numpy as np
import preprocessing
from sklearn.preprocessing import StandardScaler
from keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import datasets



#iris = datasets.load_iris()
#
#X = iris.data
#Y = iris.target
#dataset = []
#for i in range(len(X)):
#    dataset.append(np.hstack((X[i],Y[i])))
#dataset = np.random.permutation(dataset)
#x_train = dataset[:100,:4]
#y_train = dataset[:100,-1]
#
#x_test = dataset[100:120,:4]
#y_test = dataset[100:120,:4]
#
#x_val = dataset[120:,:4]
#y_val = dataset[120:,-1]
#
#y_true = dataset[:,-1]


#############PREPROCESSING##############
df = preprocessing.import_data()
train_bots = df[1].values[:10000,3:].astype(int)
train_legit = df[2].values[:10000,3:].astype(int)

x_train = np.vstack((train_bots,train_legit))


test_bots = df[1].values[12000:16000,3:].astype(int)
test_legit = df[2].values[12000:16000,3:].astype(int)

x_test = np.vstack((test_bots,test_legit))


train_bots_label = np.ones((train_bots.shape[0],1))
train_legit_label = np.zeros((train_legit.shape[0],1))

y_train = np.vstack((train_bots_label,train_legit_label))


test_bots_label = np.ones((test_bots.shape[0],1))
test_legit_label = np.zeros((test_legit.shape[0],1))

y_test = np.vstack((test_bots_label,test_legit_label))
y_true = y_test

val_bots = df[1].values[10000:12000,3:].astype(int)
val_legit = df[2].values[10000:12000,3:].astype(int)

x_val = np.vstack((val_bots, val_legit))

val_bots_label = np.ones((val_bots.shape[0],1))
val_legit_label = np.zeros((val_legit.shape[0],1))

y_val = np.vstack((val_bots_label,val_legit_label))



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
x_val = sc.fit_transform(x_val)

######create model######
model = models.Sequential()
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(2, activation = 'sigmoid'))


model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

epochs = 100
batch_size = 16

history = model.fit(x_train,
                    y_train,
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_data = (x_val, y_val))


###predictions###
y_pred = model.predict(x_test)

for i in range(len(y_pred)):
    if(y_pred[i][0] < y_pred[i][1]):
        y_pred[i] = 1
    else:
        y_pred[i] = 0
y_pred = y_pred[:,0]

acc = 0.0
for i in range(len(y_pred)):
    if y_pred[i] == y_true[i]:
        acc += 1.0
acc = acc / len(y_pred) * 100
print('Epochs run: ', epochs)
print('Batch size: ', batch_size)
print('Test data size: ', len(y_pred))
print('Accuracy on test data: ', acc, '%')

def comparison_plot(x, 
                    y_A, style_A, label_A, 
                    y_B, style_B, label_B, 
                    title, x_label, y_label):
    
    plt.clf()
    plt.plot(x, y_A, style_A, label = label_A)
    plt.plot(x, y_B, style_B, label = label_B)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']
comparison_plot(range(1, len(loss) + 1),
                loss, 'bo', 'Training',
                val_loss, 'b', 'Validation',
                'Training and validation loss',
                'Epochs',
                'Loss')

acc = history.history['acc']
val_acc = history.history['val_acc']
comparison_plot(range(1, len(loss) + 1),
                acc, 'bo', 'Training',
                val_acc, 'b', 'Validation',
                'Training and validation accuracy',
                'Epochs',
                'Accuracy')

print("Highest validation accuracy:", val_acc[np.argmax(val_acc)], "Epochs:", np.argmax(val_acc))
print("Lowest validation loss:", val_loss[np.argmin(val_loss)], "Epochs:", np.argmin(val_loss))

