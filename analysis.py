# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:54:21 2019

@author: srjcp
"""
##Importing libraries
import numpy as np
import pickle
from scipy import io
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import data_loading
from operator import itemgetter
import itertools

def plot_confusion_matrix(cm,classes,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    x_classes=classes+ ['Recall']
    y_classes=classes+ ['Precision']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontname='Times New Roman',fontsize = 16, y=1.03)
    cbar=plt.colorbar(fraction=0.046)
    cbar.ax.tick_params(labelsize=14) 
    tick_marks = np.arange(len(x_classes))
    plt.xticks(tick_marks, x_classes, fontsize = 13,fontname='Times New Roman')
    plt.yticks(tick_marks, y_classes, fontsize = 13,fontname='Times New Roman')
    plt.axhline(y=3.51,color='black')
    plt.axvline(x=3.51,color='black')

    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=13,fontname='Times New Roman')
    plt.ylabel('True label',fontsize = 15,fontname='Times New Roman')
    plt.xlabel('Predicted label',fontsize = 15,labelpad=10,fontname='Times New Roman')
    plt.tight_layout()


net_type='normal'
data_name='nopad'
data_type='LN'
net_name='CNN2D'

Ampvalues,Cycle_no,original_freq,y = data_loading.get_metadata(data_name)
# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

#Loading files which are saved during Cross-Validation program
with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_wrong_index.txt", "rb") as fp:
    wrong_index = pickle.load(fp, encoding='latin1')
with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_Test_index.txt", "rb") as fp:
    test_index = pickle.load(fp, encoding='latin1')
with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_ypred_index.txt", "rb") as fp:
    ypred_index = pickle.load(fp, encoding='latin1')

''' Creating a single long array of missclassified signals (when that signal is in the test set while performing k-fold CV). 
Columns of "wrong_indices" are: Indices of missclassified signal, Model predicted class and original class of a signal. 
One might need to alter this code depending on value of k in k-fold CV program. For eg. for k=3 calculation till ypred3 will suffice.'''

ypred_indices=[]
test_indices=[]
for i in range(0,len(ypred_index)):
    test_indices1=np.asarray(test_index[i])
    test_indices1=test_indices1[np.array(wrong_index[i])]
    test_indices=test_indices+list(test_indices1)
    ypred1=np.asarray(ypred_index[i])
    ypred1=ypred1[np.array(wrong_index[i])]
    ypred_indices =ypred_indices + list(ypred1)

ypred_indices=np.asarray(ypred_indices)
test_indices=np.asarray(test_indices)
wrong=np.concatenate((test_indices,ypred_indices,y[test_indices]),axis=1)
#io.savemat('CNN2D_nopad_normal_LN_wrong_array_forCM.mat',{'wrong':wrong})
cm_wrong = confusion_matrix(labelencoder_X_1.inverse_transform(wrong[:,2]),labelencoder_X_1.inverse_transform(wrong[:,1]));

#Sort wrong indices array as per wave number in original data
wrong_indices=np.asarray(sorted(wrong,key=itemgetter(0), reverse=False))

complete_cm=np.copy(cm_wrong)
####Confusion Matrix for model
for i in range(0,4):
    complete_cm[i,i]=5000-np.sum(cm_wrong[i,:])

###Recall and Precision Matrix
recall_precision_mat=np.zeros((2,5))
#Precision and Recall Calculations (First row recall 2nd row precision)
for i in range(0,4):
#    Alpha,Beta,Gamma,Theta
    recall_precision_mat[0,i]=complete_cm[i,i]/np.sum(complete_cm[i,:])
    recall_precision_mat[1,i]=complete_cm[i,i]/np.sum(complete_cm[:,i])
#Saving recall and precision values
#with open(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_recall_precision.txt", "wb") as fp:
#    pickle.dump(recall_precision_mat, fp)
    
#Adding recall and precision to confusion matrix
recall_precision_mat[0,4]=np.trace(complete_cm)/20000
#recall_precision_mat=recall_precision_mat/100
#Normalizing confusion matrix
complete_cm=complete_cm.astype('float') /complete_cm.sum(axis=1)
complete_cm=np.vstack((complete_cm,recall_precision_mat[1,0:4]))
complete_cm=np.hstack((complete_cm,np.reshape(np.transpose(recall_precision_mat[0,:]),[5,1])))

#Ploting confusion matrix
plt.figure(figsize=(5,5))
plot_confusion_matrix(complete_cm,title='Normalized Confusion Matrix',classes=['Alpha','Beta','Gamma','Theta'])
confusion_mat=plt.gcf()
confusion_mat.savefig(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_confusion_mat.png",bbox_inches='tight')

'''FFT Confusion Matrix'''
fft_cm = io.loadmat('cm_'+data_name+'_'+data_type+'.mat')
fft_cm=fft_cm['Con_Mat']
fft_re_pre = io.loadmat('recall_precision_'+data_name+'_'+data_type+'.mat')
fft_re_pre=fft_re_pre['recall_precision']
fft_recall_precision_mat=np.zeros((2,5))
fft_recall_precision_mat[:,0:4]=fft_re_pre
fft_recall_precision_mat[0,4]=np.trace(fft_cm)/20000
fft_cm=fft_cm.astype('float') /fft_cm.sum(axis=1)
fft_cm=np.vstack((fft_cm,fft_recall_precision_mat[1,0:4]))
fft_cm=np.hstack((fft_cm,np.reshape(np.transpose(fft_recall_precision_mat[0,:]),[5,1])))
plt.figure(figsize=(5,5))
plot_confusion_matrix(fft_cm,title='Normalized Confusion matrix',classes=['Gamma','Alpha','Beta','Theta'])
confusion_mat=plt.gcf()
confusion_mat.savefig('FFT_'+data_name+"_"+net_type+"_"+data_type+"_confusion_mat.png",bbox_inches='tight')
    
## function for creating an array of signals origianl frequency,amplitude and number of cycles 
def wrong_array(wrong_index):
    category_freq=original_freq[wrong_index]
    category_Amp=Ampvalues[wrong_index]
    category_cycle=Cycle_no[wrong_index]
    Wrong_array=np.concatenate((category_freq,category_Amp,category_cycle),axis=1)
    return Wrong_array

## Freq.,amp,cycle array for missclassified signals
Wrong_array=wrong_array(wrong_indices[:,0])
#io.savemat('CNN2D_nopad_normal_LN_wrong_array.mat',{'Wrong_array_ln':Wrong_array})

plt.figure(figsize=(8,6))
#plt.subplot(121)
plt.scatter(Wrong_array[:,0],Wrong_array[:,1],c=wrong_indices[:,1],cmap=plt.cm.get_cmap('jet', 4))
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
cbar=plt.colorbar(ticks=range(4))
cbar.ax.get_yaxis().labelpad = 22
cbar.ax.set_ylabel('Predicted Category', rotation=270,fontsize=18,fontname='Times New Roman')
cbar.ax.tick_params(labelsize=16) 
plt.clim(-0.5,3.5)
plt.xlabel('Frequency (Hz)',fontsize=18,labelpad=10,fontname='Times New Roman')
plt.ylabel('Amplitude',fontsize=18,fontname='Times New Roman')
plt.title('Misclassified Brainwaves',fontsize=22,fontname='Times New Roman')
plt.tight_layout()
scatter_plot=plt.gcf()
scatter_plot.savefig(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_misclass_scatter_plot.png",bbox_inches='tight')



'''CNN and FFT result comparison plots'''
##Loading Matlab results
fft_wrong = io.loadmat('wrong_signal_mat_'+data_name+'_'+data_type+'1.mat')
fft_wrong_signals=fft_wrong['wrong_detected_signals_mat1']

plt.figure(figsize=(20,8))
plt.subplot(121)
plt.scatter(Wrong_array[:,0],Wrong_array[:,1],c=wrong_indices[:,1],cmap=plt.cm.get_cmap('jet', 4))
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
cbar=plt.colorbar(ticks=range(4))
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.set_ylabel('Predicted Category', rotation=270,fontsize=22,fontname='Times New Roman')
cbar.ax.tick_params(labelsize=16) 
#cbar.ax.set_yticklabels(['Alpha','Beta', 'Gamma', 'Theta'])
#clb.ax.set_label('This is a title',rotation=270)
plt.clim(-0.5,3.5)
plt.xlabel('Frequency (Hz)',fontsize=22,labelpad=10,fontname='Times New Roman')
plt.ylabel('Amplitude',fontsize=22,fontname='Times New Roman')
plt.title('(a)',fontsize=24,fontname='Times New Roman')
plt.subplot(122)
plt.scatter(fft_wrong_signals[:,4],fft_wrong_signals[:,2],c=fft_wrong_signals[:,1],cmap=plt.cm.get_cmap('jet', 4))
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
cbar1=plt.colorbar(ticks=range(4))
cbar1.ax.get_yaxis().labelpad = 20
cbar1.ax.set_ylabel('Predicted Category', rotation=270,fontsize=22,fontname='Times New Roman')
cbar1.ax.tick_params(labelsize=16) 
#clb1.ax.set_label('This is a title',rotation=270)
plt.clim(-0.5,3.5)
plt.xlabel('Frequency (Hz)',fontsize=22,labelpad=10,fontname='Times New Roman')
plt.ylabel('Amplitude',fontsize=22,fontname='Times New Roman')
plt.title('(b)',fontsize=24,fontname='Times New Roman')
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar( ticks=['Alpha','Beta','Gamma','Theta'])
plt.tight_layout()
scatter_plot=plt.gcf()
scatter_plot.savefig(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"FFT_misclass_scatter_plot.png")

###FFT and CNN Comparison plots
##fft_wrong_signals=np.transpose(fft_wrong_signals)
#common_wrong_signals=np.intersect1d((fft_wrong_signals-1),wrong_indices[:,0])
#common_wrong_array=wrong_array(common_wrong_signals)
#
#wrongfft_correctcnn=np.setdiff1d((fft_wrong_signals-1),common_wrong_signals)
#wrongfft_correctcnn_array=wrong_array(wrongfft_correctcnn)
#
#wrongcnn_correctfft=np.setdiff1d(wrong_indices[:,0],common_wrong_signals)
#wrongcnn_correctfft_array=wrong_array(wrongcnn_correctfft)
#
#io.savemat('Wrongfft_correctcnn_array_'+net_type+"_"+data_type+'.mat',{'wrongfft_correctcnn_array':wrongfft_correctcnn_array})
#io.savemat('Wrongcnn_correctfft_array_'+net_type+"_"+data_type+'.mat',{'wrongcnn_correctfft_array':wrongcnn_correctfft_array})
#io.savemat('Common_wrong_arrayFFT_'+net_type+"_"+data_type+'.mat',{'common_wrong_array':common_wrong_array})
#plt.subplot(1,2,1)
#plt.scatter
#plt.scatter(common_wrong_array[:,0],common_wrong_array[:,1],c=common_wrong_array[:,2],cmap='gnuplot')
#plt.colorbar(orientation="horizontal")
#plt.xlabel('Frequency (Hz)',fontsize=12)
#plt.ylabel('Amplitude',fontsize=12)
#plt.title('Misclassified Brainwaves',fontsize=14)
#plt.subplot(1,2,2)
#plt.scatter
#plt.scatter(wrongcnn_correctfft_array[:,0],wrongcnn_correctfft_array[:,1],c=wrongcnn_correctfft_array[:,2],cmap='gnuplot')
#plt.colorbar(orientation="horizontal")
#plt.xlabel('Frequency (Hz)',fontsize=12)
#plt.ylabel('Amplitude',fontsize=12)
#plt.title('Misclassified Brainwaves',fontsize=14)
#plt.tight_layout()
#plt.subplot(1,3,3)
#plt.scatter
#plt.scatter(wrongfft_correctcnn_array[:,0],wrongfft_correctcnn_array[:,1],c=wrongfft_correctcnn_array[:,2],cmap='gnuplot')
#plt.colorbar(orientation="horizontal")
#plt.xlabel('Frequency (Hz)',fontsize=12)
#plt.ylabel('Amplitude',fontsize=12)
#plt.title('Misclassified Brainwaves',fontsize=14)
#plt.tight_layout()
#scatter_plot=plt.gcf()
#scatter_plot.savefig(net_name+"_"+data_name+"_"+net_type+"_"+data_type+"_misclass_scatter_plot.png")
