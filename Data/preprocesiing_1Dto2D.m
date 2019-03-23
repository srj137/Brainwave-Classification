% % Use this program for converting 1D data to 2D
clear all;
close all;
load('SameTimeSample_GT1.mat');
filter_order=200;
cutoff_freq=0.01;
F=zeros(16,(filter_order+1));
% Creating 16( 1LP and 15BP) FIR filters  
for i=1:16
    if i==1
       F(i,:)=fir1(filter_order,cutoff_freq,'low'); % lowpass filter with hamming window of length 201
    else
       F(i,:)=fir1(filter_order,[(i-1)*cutoff_freq,cutoff_freq*i],'bandpass'); % bandpass filter with hamming window of length 201
    end
%      figure(i);
%      freqz(F(i,:));
end
% Applying filters to brainwaves
H=zeros(size(e,1),16,size(e,2));
for j=1:size(e,1)
for k=1:16
    H(j,k,:)= filter(F(k,:),1,e(j,:));
end
end

filename='SameTimeSample_GT_3D1';
save(filename,'H','-v7.3');