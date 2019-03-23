# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:09:18 2019

@author: srjcp
"""
import networks
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import data_loading
from keras.models import load_model
import numpy as np
import pickle

data_name='nopad'
data_type='GT'
net_type='normal'
net_name='CNN1D'
#Load Data
X, y= data_loading.get_1Ddata(data_name,data_type)
# Encoding categorical data i.e. labels
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y) 

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))

cvscores_test = []
cvscores_train = []
test_index=[]
wrong_index=[]
ypred_index=[]
batch_size=128
no_epochs=100
###Checking if imported network is correct or not
if net_name=='CNN1D':
    model=networks.get_1Dmodel(net_type)
    model.summary()
elif net_name=='CNN2D':
    model=networks.get_2Dmodel(net_type)
    model.summary()

for i,(train_idx,val_idx) in enumerate(folds):
    X_train_CV=X[train_idx]
    y_train_CV=y[train_idx]
    X_val_CV=X[val_idx]
    y_val_CV=y[val_idx]
    test_index.append(val_idx)
    model_weights = net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_fold" + str(i) + "_weights.h5"
    callbacks = networks.get_callbacks(name_weights = model_weights, patience_lr=10)
    if net_name=='CNN1D':
        model=networks.get_1Dmodel(net_type)
    elif net_name=='CNN2D':
        model=networks.get_2Dmodel(net_type)
    model.fit(X_train_CV, y_train_CV, batch_size = batch_size, epochs =no_epochs, verbose=2, validation_data=(X_val_CV,y_val_CV), callbacks=callbacks)
    model=load_model(model_weights) 
    #evaluate model
    scores_test = model.evaluate(X_val_CV, y_val_CV, verbose=2)
    #evaluate model
    scores_train = model.evaluate(X_train_CV, y_train_CV, verbose=2)
    cvscores_test.append(scores_test[1] * 100)
    cvscores_train.append(scores_train[1] * 100)
    y_pred = model.predict(X_val_CV, batch_size=batch_size)
    Y_pred = np.argmax(y_pred, axis=1) 
    ypred_index.append(Y_pred)
    Wrong= np.not_equal(Y_pred,y_val_CV)
    Wrong_index=np.argwhere(Wrong)
    wrong_index.append(Wrong_index)
    ##Confusion Matrix
#    CM_test=confusion_matrix(y_val_CV,Y_pred)
#    y_train_pred = model.predict(X_train_CV,batch_size=batch_size)
#    Y_train_pred = np.argmax(y_train_pred, axis=1)
#    CM_train=confusion_matrix(y_train_CV,Y_train_pred)
#   # IA_test,IA_train=wave_accuracy(CM_test,CM_train)
#    #indivisual_accuracy_test.append(IA_test)
#   # indivisual_accuracy_train.append(IA_train) 
#    CM_Train.append(CM_train)
#    CM_Test.append(CM_test)
    

#with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_CM_Train.txt", "wb") as fp:
#    pickle.dump(CM_Train, fp)
#with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_CM_Test.txt", "wb") as fp:
#    pickle.dump(CM_Test, fp)
#with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_IA_Test.txt", "wb") as fp:
#    pickle.dump(indivisual_accuracy_test, fp)    
#with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_IA_Train.txt", "wb") as fp:
#    pickle.dump(indivisual_accuracy_train, fp)    
with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_wrong_index.txt", "wb") as fp:
    pickle.dump(wrong_index, fp)    
with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_Test_index.txt", "wb") as fp:
    pickle.dump(test_index, fp)    
with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_ypred_index.txt", "wb") as fp:
    pickle.dump(ypred_index, fp)    

mean_test = np.mean(cvscores_test)
variance_test = np.std(cvscores_test)

mean_train = np.mean(cvscores_train)
variance_train = np.std(cvscores_train)
#print "Mean Test Accuracy is", mean_test
#print "Test variance is", variance_test
#
#print "Mean Train Accuracy is", mean_train
#print "Train variance is", variance_train

test_accuracies=np.asarray(cvscores_test)
train_accuracies=np.asarray(cvscores_train)
with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_Test_accuracies.txt", "wb") as fp:
    pickle.dump(test_accuracies, fp)    
with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_Train_accuracies.txt", "wb") as fp:
    pickle.dump(train_accuracies, fp) 
#print "Train Accuracies are", train_accuracies
#print "Test Accuracies are", test_accuracies