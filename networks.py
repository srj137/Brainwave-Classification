# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:19:42 2019

@author: srjcp
"""

from keras.layers import Conv1D
from keras.layers import MaxPooling1D,GlobalAveragePooling1D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ReduceLROnPlateau
#from keras.utils import multi_gpu_model
from keras.regularizers import l2
from keras import optimizers
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

def get_1Dmodel(net_type):
    ##Defining parameters 
    l2_lambda=0.01
    initializer=initializers.he_uniform()
    initializer1=initializers.glorot_uniform()
    dropout_rate=0.1
    
    visible2=Input(shape=(2000,1), name='Input2')
    batch_norm10=BatchNormalization()(visible2)
     # First bloack of convolutional and pooling layer
    conv1d11=Conv1D(8,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_11')(batch_norm10)
    batch_norm11=BatchNormalization()(conv1d11)
    activation11=Activation('elu')(batch_norm11)
    dropout11=Dropout(rate= dropout_rate,name='dropout11')(activation11)
    conv1d12=Conv1D(8,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_12')(dropout11)
    batch_norm12=BatchNormalization()(conv1d12)
    activation12=Activation('elu')(batch_norm12)
    dropout12=Dropout(rate= dropout_rate,name='dropout12')(activation12)
    pool12=MaxPooling1D(pool_size =2, strides=2,name='pool1d_1')(dropout12)
     
    # Adding a second bloack convolutional and pooling layer
    conv1d21=Conv1D(8,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_21')(pool12)
    batch_norm21=BatchNormalization()(conv1d21)
    activation21=Activation('elu')(batch_norm21)
    dropout21=Dropout(rate= dropout_rate,name='dropout21')(activation21)
    conv1d22=Conv1D(8,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_22')(dropout21)
    batch_norm22=BatchNormalization()(conv1d22)
    activation22=Activation('elu')(batch_norm22)
    dropout22=Dropout(rate= dropout_rate,name='dropout22')(activation22)
    pool22=MaxPooling1D(pool_size =2, strides=2,name='pool1d_2')(dropout22)
     
    # Adding a third block of convolutional and pooling layer
    conv1d31=Conv1D(8,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_31')(pool22)
    batch_norm31=BatchNormalization()(conv1d31)
    activation31=Activation('elu')(batch_norm31)
    dropout31=Dropout(rate= dropout_rate,name='dropout31')(activation31)
    conv1d32=Conv1D(8,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_32')(dropout31)
    batch_norm32=BatchNormalization()(conv1d32)
    activation32=Activation('elu')(batch_norm32)
    dropout32=Dropout(rate= dropout_rate,name='dropout32')(activation32)
    conv1d33=Conv1D(8,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_33')(dropout32)
    batch_norm33=BatchNormalization()(conv1d33)
    activation33=Activation('elu')(batch_norm33)
    dropout33=Dropout(rate= dropout_rate,name='dropout33')(activation33)
    pool33=MaxPooling1D(pool_size =2, strides=2,name='pool1d_3')(dropout33)
    
    # Adding a fourth convolutional and pooling layer
    conv1d41=Conv1D(8,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_41')(pool33)
    batch_norm41=BatchNormalization()(conv1d41)
    activation41=Activation('elu')(batch_norm41)
    dropout41=Dropout(rate= dropout_rate,name='dropout41')(activation41)
    conv1d42=Conv1D(8,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_42')(dropout41)
    batch_norm42=BatchNormalization()(conv1d42)
    activation42=Activation('elu')(batch_norm42)
    dropout42=Dropout(rate= dropout_rate,name='dropout42')(activation42)
    conv1d43=Conv1D(16,(3),padding='same', kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv1d_43')(dropout42)
    batch_norm43=BatchNormalization()(conv1d43)
    activation43=Activation('elu')(batch_norm43)
    dropout43=Dropout(rate= dropout_rate,name='dropout43')(activation43)
 #    Choosing Global Average Pooling Or normal Max Pooling approach   
    if net_type=='GAP':
        pool43= GlobalAveragePooling1D()(dropout43)
        FC1 = Dense(units=16,kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='FC1')(pool43)
    elif net_type=='normal':
        pool43=MaxPooling1D(pool_size =2, strides=2,name='pool1d_4')(dropout43)
        flat = Flatten()(pool43)
        ###fully connected layer
        FC1 = Dense(units=16,kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='FC1')(flat)
    
    batch_norm=BatchNormalization()(FC1)
    activation1=Activation('elu')(batch_norm)
    dropout1= Dropout(rate=0.5,name='dropout1')(activation1)
    output = Dense(units=4,kernel_initializer=initializer1,kernel_regularizer=l2(l2_lambda), activation = 'softmax',use_bias=False,name='output')(dropout1)
    classifier = Model(inputs=visible2, outputs=output)
    ##Compile created model 
    optimizer=optimizers.nadam(lr=0.001)
    classifier.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return classifier

def get_2Dmodel(net_type):
    l2_lambda=0.01
    initializer=initializers.he_uniform()
    initializer1=initializers.glorot_uniform()
    dropout_2d=0.2
   
    visible=Input(shape=(16,2000,1), name='Input')
    batch_norm10=BatchNormalization()(visible)
    # First bloack of convolutional and pooling layer
    conv11 = Conv2D(8, (3,3), padding='same',kernel_initializer=initializer ,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv2d_11')(batch_norm10)
    batch_norm11=BatchNormalization()(conv11)
    activation11=Activation('elu')(batch_norm11)
    dropout11=Dropout(rate= dropout_2d,name='dropout11')(activation11)
    pool11 = MaxPooling2D(pool_size = (2,2), strides=(2,2),name='pool2d_1')(dropout11)
    
    # Adding a second bloack convolutional and pooling layer
    conv21 = Conv2D(16, (3,3), padding='same',kernel_initializer=initializer ,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv2d_21')(pool11)
    batch_norm21=BatchNormalization()(conv21)
    activation21=Activation('elu')(batch_norm21)
    dropout21=Dropout(rate= dropout_2d,name='dropout21')(activation21)
    conv22 = Conv2D(16, (3,3), padding='same', kernel_initializer=initializer ,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv2d_22')(dropout21)
    batch_norm22=BatchNormalization()(conv22)
    activation22=Activation('elu')(batch_norm22)
    dropout22=Dropout(rate= dropout_2d,name='dropout22')(activation22)
    conv23 = Conv2D(16, (3,3), padding='same', kernel_initializer=initializer ,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv2d_23')(dropout22)
    batch_norm23=BatchNormalization()(conv23)
    activation23=Activation('elu')(batch_norm23)
    dropout23=Dropout(rate= dropout_2d,name='dropout23')(activation23)
    pool23 = MaxPooling2D(pool_size = (2,2), strides=(2,2),name='pool2d_2')(dropout23)
    
    # Adding a third block of convolutional and pooling layer
    conv31 = Conv2D(16, (3,3), padding='same',kernel_initializer=initializer ,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv2d_31')(pool23)
    batch_norm31=BatchNormalization()(conv31)
    activation31=Activation('elu')(batch_norm31)
    dropout31=Dropout(rate= dropout_2d,name='dropout31')(activation31)
    conv32 = Conv2D(16, (3,3), padding='same',kernel_initializer=initializer ,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv2d_32')(dropout31)
    batch_norm32=BatchNormalization()(conv32)
    activation32=Activation('elu')(batch_norm32)
    dropout32=Dropout(rate= dropout_2d,name='dropout32')(activation32)
    pool32 = MaxPooling2D(pool_size = (2,2), strides=(2,2),name='pool2d_3')(dropout32)
    
    # Adding a fourth convolutional and pooling layer
    conv41 = Conv2D(16, (3,3), padding='same',kernel_initializer=initializer ,kernel_regularizer=l2(l2_lambda),use_bias=False,name='conv2d_41')(pool32)
    batch_norm41=BatchNormalization()(conv41)
    activation41=Activation('elu')(batch_norm41)
    dropout41=Dropout(rate= dropout_2d,name='dropout41')(activation41)
#    Choosing Global Average Pooling Or normal Max Pooling approach
    if net_type=='GAP':
        pool4= GlobalAveragePooling2D()(dropout41)
        ##Fully Connected Layers
        FC1 = Dense(units=16, kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='FC1')(pool4)
    elif net_type=='normal':
        pool4 = MaxPooling2D(pool_size = (2,2), strides=(2,2),name='pool2d_4')(dropout41)
        #COnnecting CNN
        flat=Flatten()(pool4)
        ##Fully Connected Layers
        FC1 = Dense(units=16, kernel_initializer=initializer,kernel_regularizer=l2(l2_lambda),use_bias=False,name='FC1')(flat)
    
    batch_norm1=BatchNormalization()(FC1)
    activation1=Activation('elu')(batch_norm1)
    dropout1= Dropout(rate=0.5,name='dropout1')(activation1)
    output = Dense(units=4,kernel_initializer=initializer1 ,kernel_regularizer=l2(l2_lambda), activation = 'softmax',use_bias=False,name='output')(dropout1)
    classifier = Model(inputs=visible, outputs=output)
    
    optimizer=optimizers.Nadam(lr=0.001)
    classifier.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return classifier

def get_callbacks(weights_name, patience_lr):
    checkpointer = ModelCheckpoint(weights_name,  verbose=1, save_best_only=True, monitor='val_loss')
    reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_lr)
    return [checkpointer, reduce_lr]

