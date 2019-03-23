# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:04:25 2019

@author: srjcp
"""

##Importing Keras Library functions
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import data_loading
import networks

data_name='nopad'
data_type='GT'
net_type='normal'
net_name='CNN1D'
#Load Data
X, y= data_loading.get_1Ddata(data_name,data_type)
# Encoding labels
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y) 
## Spliting training and testing data randomly with 20% being the test data
X_train,X_test,y_train,y_test =train_test_split(X, y, test_size = 0.2, random_state = 0)
batch_size=50
no_epochs=1


model=networks.get_1Dmodel(net_type)
model.summary()
model_weights= net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_weights.h5"
callbacks = networks.get_callbacks(weights_name = model_weights, patience_lr=10)
Classifier=model.fit(X_train, y_train, batch_size = batch_size, epochs =no_epochs, verbose=2, validation_split=0.1, callbacks=callbacks)
model.save_weights(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"end_weights.h5")

##Checking model accuracy on test and training data
Test_accuracy=model.evaluate(X_test,y_test,batch_size=64,verbose=2)
print('Test set Accuracy is {}'.format(Test_accuracy))
Train_accuracy=model.evaluate(X_train,y_train,batch_size=64,verbose=2)
print ("Train set Accuracy is {}".format(Train_accuracy))

# summarize history for accuracy
plt.figure(1)
plt.plot(Classifier.history['acc'])
plt.plot(Classifier.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
Accuracy_fig = plt.gcf()
#plt.draw()
Accuracy_fig.savefig('Epoch Accuracy Figure('+net_name+"_"+data_name+"_"+net_type+"_"+data_type+').png')

# summarize history for loss
plt.figure(2)
plt.plot(Classifier.history['loss'])
plt.plot(Classifier.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
Loss_fig = plt.gcf()
#plt.draw()
Loss_fig.savefig('Epoch Loss Figure('+net_name+"_"+data_name+"_"+net_type+"_"+data_type+').png')