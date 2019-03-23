clc;close all;clear all
load('NewData_14db_noise_equal.mat');
load('updatedData_labels_equal.mat');
load('NewData_CycNum_equal.mat');
load('NewData_Frequency_equal.mat');
load('NewData_Amplitude_equal.mat');
% load('NewData_5db_noise_equal.mat');
% % load('NewData_GT_equal.mat');
data_name='zeropad'
data_type='LN'

% load('SameTimeSample_LN1.mat');
% load('SameTimeSample_labels1.mat');
% load('SameTimeSample_burstFreq1.mat');
% load('SameTimeSample_burstCycNum1.mat');
% load('SameTimeSample_burstAmp1.mat');
% % load('SameTimeSample_SN1');
% % load('SameTimeSample_GT1');
% data_name='nopad'
% data_type='LN'

%FFT calculation
fprintf('FFT calculating...\n')
Fs=1000; % sampling frequency
nfft=2^12;  % power values have to be change to check frequecy values V/S nfft value 
% other FFT calculation techniques can be used, this code was used, so that
% each signal with its FFT representation can be seen individually
f=(0:nfft/2-1)*Fs/nfft; % frequency scaling
X=fft(e,nfft,2);   % signal amplitude is the variable inwhich time samples are stored
Z=X(:,1:nfft/2); % deleting mirrored frequency values
min_freq_indx = find(f<4,1,'last')+1; % selecting boundary for calculated FFT frequency values 
max_freq_indx = find(f<80,1,'last'); % only values from 4 Hz to 64 Hz are taken into consideration
f1 = f(min_freq_indx:max_freq_indx); % frequency boundary adjustement
[~,max_freq_pos] = max(abs(Z(:,min_freq_indx:max_freq_indx)),[],2);
FFT_freq = f1(max_freq_pos);
clear Z f1 

Max_Min = [8 12;12 30;30 80;4 8];
fprintf('Separating signals on the basis of FFT...\n')
% A = java_array('java.lang.String', length(FFT_freq));
fft_labels=zeros(20000,1);
for count_for=1:length(FFT_freq)
    labels(count_for,:)=blanks(5);
    if (FFT_freq(count_for) > Max_Min(4,1)) && (FFT_freq(count_for) < Max_Min(4,2))
        labels(count_for,:)= 'Theta';
        fft_labels(count_for)= 3;
    elseif (FFT_freq(count_for) > Max_Min(1,1)) && (FFT_freq(count_for) < Max_Min(1,2))
        labels(count_for,:)= 'Alpha';
        fft_labels(count_for)= 0;
    elseif (FFT_freq(count_for) > Max_Min(2,1)) && (FFT_freq(count_for) < Max_Min(2,2))
        labels(count_for,:)= 'Beta ';
        fft_labels(count_for)= 1;
    elseif (FFT_freq(count_for) > Max_Min(3,1)) && (FFT_freq(count_for) < Max_Min(3,2))
        labels(count_for,:)= 'Gamma';
        fft_labels(count_for)= 3;
%     else
%        labels(count_for,:)= 'No Category';
    end
end
FFT_Label = labels;
clear labels

%Calculating the classification ccuracy 
fprintf('Accuracy calculating...\n')
detected_count =0;wrongly_detected_count=0;signals_cmp=zeros();
wrongly_detected_signals = zeros();detected_signals = zeros();
for k_for= 1:length(FFT_freq)
    signals_cmp(k_for)= strcmp(a(k_for),FFT_Label(k_for));
    if signals_cmp(k_for) == 1
        detected_count = detected_count + 1;
        detected_signals(detected_count) = k_for;
    else
        wrongly_detected_count = wrongly_detected_count + 1;
        wrongly_detected_signals(wrongly_detected_count) = k_for;
    end
end
wrong_detected_signals_mat1=zeros(length(wrongly_detected_signals),5);
for i=1:length(wrongly_detected_signals)
    wrong_detected_signals_mat1(i,1)=wrongly_detected_signals(i);
    wrong_detected_signals_mat1(i,2)=fft_labels(wrongly_detected_signals(i));
    wrong_detected_signals_mat1(i,3)=burstAmp(wrongly_detected_signals(i));
    wrong_detected_signals_mat1(i,4)=burstCycNum(wrongly_detected_signals(i));
    wrong_detected_signals_mat1(i,5)=burstFreq(wrongly_detected_signals(i));
end
wrong_signal_mat_filename=strcat('wrong_signal_mat_',data_name,'_',data_type);
save(wrong_signal_mat_filename,'wrong_detected_signals_mat')
clear i_for  j_for  k_for A X signals_cmp detected_count,
clear count_for p_for Fs 
a=cellstr(a);
FFT_Label=cellstr(FFT_Label);
fprintf('creating wrong detected signal info matrix...\n')
wrong_detected_signals_mat = cell(wrongly_detected_count,8);
for i_count = 1:wrongly_detected_count
    wrong_detected_signals_mat(i_count,1) = {i_count};
    wrong_detected_signals_mat(i_count,2) = {wrongly_detected_signals(i_count)};
    wrong_detected_signals_mat(i_count,3) = a(wrongly_detected_signals(i_count));
    wrong_detected_signals_mat(i_count,4) = {burstFreq(wrongly_detected_signals(i_count))};
    wrong_detected_signals_mat(i_count,6) = FFT_Label(wrongly_detected_signals(i_count));
    wrong_detected_signals_mat(i_count,5) = {FFT_freq(wrongly_detected_signals(i_count))};
    wrong_detected_signals_mat(i_count,7) = {burstAmp(wrongly_detected_signals(i_count))};
    wrong_detected_signals_mat(i_count,8) = {burstCycNum(wrongly_detected_signals(i_count))};
end
clear i_count wrongly_detected_count wrongly_detected_signals
fprintf('calculating confusion matrix and classification accuracy...\n')
[Con_Mat, order] = confusionmat(a,FFT_Label); % confusion matrix can be foud under 
% 'Con_Mat' and order of the matrix is under 'order' variable.
Accuracy = (trace(Con_Mat))/ size(e,1)

for i =1:size(Con_Mat,1)
    recall_precision(1,i)=Con_Mat(i,i)/sum(Con_Mat(i,:));
    recall_precision(2,i)=Con_Mat(i,i)/sum(Con_Mat(:,i));
end
recall_precision_filename=strcat('recall_precision_',data_name,'_',data_type);
save(recall_precision_filename,'recall_precision');