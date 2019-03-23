# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:48:17 2019

@author: srjcp
"""

from scipy import io
import numpy as np
import h5py 

def get_2Ddata(data_name,data_type):
    ##loading data from a .mat files
    if data_name=='zeropad':
        if data_type=='LN':
            f = h5py.File('LargeNoise3D_NewData.mat','r')   ##Name of a .mat file
        elif data_type=='SN':
            f = h5py.File('SmallNoise3D_NewData.mat','r') ##Name of a .mat file
        elif data_type=='GT':
            f = h5py.File('GroundTruth3D_NewData.mat','r')   ##Name of a .mat file
        data = f.get('H') 
        data = np.array(data)
        data=np.transpose(data)
        data=np.expand_dims(data, axis =3)
        #Loading labels
        Y = io.loadmat('updatedData_labels_equal.mat')
        y=Y['a']
        
    if data_name=='nopad':
        if data_type=='LN':
            f = h5py.File('SameTimeSample_LN_3D1.mat','r')   ##Name of a .mat file
        elif data_type=='SN':
            f = h5py.File('SameTimeSample_SN_3D1.mat','r') ##Name of a .mat file
        elif data_type=='GT':
            f = h5py.File('SameTimeSample_GT_3D1.mat','r')   ##Name of a .mat file
        data = f.get('H') 
        data = np.array(data)
        data=np.transpose(data)
        data=np.expand_dims(data, axis =3)
        #Loading labels
        Y = io.loadmat('SameTimeSample_labels1.mat')
        y=Y['a']
    return data, y

def get_1Ddata(data_name,data_type):
    ##loading data from a .mat files
    if data_name=='zeropad':
        if data_type=='LN':
            X = io.loadmat('NewData_14db_noise_equal.mat') ##Name of a .mat file
        elif data_type=='SN':
            X = io.loadmat('NewData_5db_noise_equal.mat') ##Name of a .mat file
        else:
            X = io.loadmat('NewData_GT_equal.mat') ##Name of a .mat file
        x=X['e'] ##'e' is a variable name stored in .mat file
        x=np.expand_dims(x, axis =2)
        Y = io.loadmat('updatedData_labels_equal.mat')
        y=Y['a']
        
    if data_name=='nopad':
        if data_type=='LN':
            X = io.loadmat('SameTimeSample_LN1.mat') ##Name of a .mat file
        elif data_type=='SN':
            X = io.loadmat('SameTimeSample_SN1.mat') ##Name of a .mat file
        else:
            X = io.loadmat('SameTimeSample_GT1.mat') ##Name of a .mat file
        x=X['e'] ##'e' is a variable name stored in .mat file
        x=np.expand_dims(x, axis =2)
        #Loading labels
        Y = io.loadmat('SameTimeSample_labels1.mat')
        y=Y['a']
    return x, y

def get_metadata(data_name):
    if data_name=='nopad':
        ## Loading Signal Meta-data
        Ampvalues = io.loadmat('SameTimeSample_burstAmp1.mat')
        Ampvalues=Ampvalues['burstAmp']
        Cycle_no = io.loadmat('SameTimeSample_burstCycNum1.mat')
        Cycle_no=Cycle_no['burstCycNum']
        original_freq=io.loadmat('SameTimeSample_burstFreq1.mat')
        original_freq=original_freq['burstFreq']
        
        Y = io.loadmat('SameTimeSample_labels1.mat')
        y=Y['a']
    elif data_name=='zeropad':
        ## Loading Signal Meta-data
        Ampvalues = io.loadmat('NewData_Amplitude_equal.mat')
        Ampvalues=Ampvalues['burstAmp']
        Cycle_no = io.loadmat('NewData_CycNum_equal.mat')
        Cycle_no=Cycle_no['burstCycNum']
        original_freq=io.loadmat('NewData_Frequency_equal.mat')
        original_freq=original_freq['burstFreq']
        
        Y = io.loadmat('updatedData_labels_equal.mat')
        y=Y['a']
    return Ampvalues,Cycle_no,original_freq,y