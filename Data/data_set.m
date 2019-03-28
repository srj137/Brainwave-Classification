%%% Run this program 6 times with different rng values and store
%%% data,data_noisy,labels,burstCycNum,burstAmp,burstFreq variables.
%%% Combine data from all 6 runs and select first 5000 burst of each class
%%% category to have same number of bursts 

clc; clear; close all;
rng(21);
scalFac = 0.6; % Scaling factor (Increase to increase noise level)
%% Synthetic data
sigDT = 0.001; %time step
chanNum = 15; %number of channels
gToLRatio = 0.5; %ratio of noise that is global versus local
meanBurstAmp = 5; %scale factor for burst strengths
burstRate = 1; %how often bursts occur
sigDur = 20010; %total duration of synthetic signal
burstSeq = round(sort(((rand(burstRate*(sigDur-10),1)*(sigDur-10))+1)/sigDT));
burstTrace = zeros(1,round(sigDur/sigDT));
%% create random walk
[bH, aH]= butter(2,0.5/500,'high'); %% 0.001
% freqz(bH,aH)
randWalkLocal = cumsum(randn(chanNum,size(burstTrace,2)),2);
randWalkGlobal = cumsum(randn(1,size(burstTrace,2)),2);
randWalkGlobal = filtfilt(bH,aH,randWalkGlobal);
for j = 1:chanNum
    randWalkLocal(j,:) = filtfilt(bH,aH,randWalkLocal(j,:));
end
randComp = bsxfun(@plus,randWalkLocal,gToLRatio*randWalkGlobal);
randComp = scalFac*randComp; % Multiply scaling factor as gToL ratio is set% high
%% create ground truth trace
burstCount = 1;
% =================================================
% rand('state',113);
MaxNumBurstEachKind=5000;
MaxNumBurst=MaxNumBurstEachKind*4;
epsilon=0.2; % tolerance between brainwave classes
%Theta=[4 8];
%Alpha=[8 12];
%Beta=[12 30];
%Gamma=[30 80];
 
Theta=[4+epsilon 8-epsilon];
Alpha=[8+epsilon 12-epsilon];
Beta=[12+epsilon 30-epsilon];
Gamma=[30+epsilon 80-epsilon];
 
freqTheta=rand(1,MaxNumBurstEachKind)*diff(Theta)+Theta(1);
freqAlpha=rand(1,MaxNumBurstEachKind)*diff(Alpha)+Alpha(1);
freqBeta=rand(1,MaxNumBurstEachKind)*diff(Beta)+Beta(1);
freqGamma=rand(1,MaxNumBurstEachKind)*diff(Gamma)+Gamma(1);
 
freqAll=[freqTheta,freqAlpha,freqBeta,freqGamma];
rndIdx=rand(MaxNumBurst,1)*(MaxNumBurst)+1;
rndIdx=round(rndIdx);
rndIdx(find(rndIdx>MaxNumBurst))=MaxNumBurst;
rndIdx(find(rndIdx<1))=1;
 
burstFreq=freqAll(rndIdx);
% =================================================
for j = 1:length(burstSeq)
    j/length(burstSeq);
%     burstFreq(j) = 4+(rand(1)*100); % burst frequencies range from 1 to 101 Hz
    burstCycNum(j) = 2+round(rand(1)*10); % number of cycles per burst ranges from 2 to 12
%     burstCycNum(j) = (burstFreq(j)/1000)*2000; %getting number of cycle in 2 seconds
    burstAmp(j) = 2*meanBurstAmp*rand(1); % burst amplitudes range from 0 to 2*meanBurstAmp
    burstPhase(j) = (rand(1)*2*pi)-pi; % phase of oscillation at burst peak goes from 0 to 360 degrees
    tPts = -(burstCycNum(j)/(2*burstFreq(j))):sigDT:(burstCycNum(j)/(2*burstFreq(j)));
    indList = ((1:length(tPts))-floor(length(tPts)/2));
    
    ind_cell{j,1}=indList; % IndList for each burst
    wid(j)=length(indList); % width for each burst
    
    nPDF = normpdf(tPts,0,(burstCycNum(j)/(2*burstFreq(j)))/3);
    sigma(j)=(burstCycNum(j)/(2*burstFreq(j)))/3;
    oscBurst = (nPDF/max(nPDF)).*sin((2*pi*burstFreq(j)*tPts)+burstPhase(j))...
        *burstAmp(j); % burst waveform
    osc_cell{j,1} = oscBurst;% OscBurst for each burst   
    burstTrace(burstSeq(j)+indList) = burstTrace(burstSeq(j)+indList)+oscBurst; %burst formation
    %Get Burst Indexes
    if(mod(wid(j),2)==0)
        sd(j)=(wid(j)/2);
        sInd=(sd(j)-1);
        eInd=sd(j);
    else
        sd(j)=round(wid(j)/2);
        sInd=(sd(j)-2);
        eInd=sd(j);
    end
    Index_ori(j,1)=sInd;
    Index_ori(j,2)=eInd;
    trueindexs(j,1)=burstSeq(j)-Index_ori(j,1); 
    trueindexs(j,2)=burstSeq(j)+Index_ori(j,2);
end
%% combine burst trace with noise
simLFP = bsxfun(@plus,burstTrace,randComp);
% burstSim.Settings.scalingFac = scalFac;
% burstSim.Settings.num=chanNum;
% burstSim.Settings.GToLRatio = gToLRatio;
% burstSim.Settings.Amp = meanBurstAmp;
% burstSim.Settings.BurstRate = burstRate;
% % Simulated local field potentials with oscillatory bursts
% burstSim.Traces.RandComp = randComp; % random walk noise
% burstSim.Traces.BurstTrace = burstTrace; % bursts, no noise
% burstSim.Traces.SimLFP = simLFP; % combined random walk and bursts, the simulated LFP
% % Properties of the bursts that we generated
% burstSim.Bursts.BurstIndices = burstSeq';
% burstSim.Bursts.BurstFreq = burstFreq;
% burstSim.Bursts.BurstCycNum = burstCycNum;
% burstSim.Bursts.BurstAmp = burstAmp;
% burstSim.Bursts.BurstPhase = burstPhase;
% burstSim.Bursts.width = wid;
%% calculate SNR of noise combined burst trace
meansimLFP = mean(simLFP);
ps = sum(sum((burstTrace-mean(mean(burstTrace))).^2));
pn = sum(sum((burstTrace-meansimLFP).^2));
snr = 10*log10(ps/pn);
%% 4 - 80 Hz bursts, using traditional burst detection
% Select only the bursts between 4-80 Hertz for analysis
% selBursts = (burstSim.Bursts.BurstFreq>4)&(burstSim.Bursts.BurstFreq<80);
% trueBursts = burstSim.Bursts.BurstIndices(selBursts);
% pos=find(selBursts);
% %Indexes of the bursts between 4-80Hz
% for i=1:length(pos)
%     ind = pos(i);
%     mBurst = burstSim.Bursts.BurstIndices(ind);
%     wBurst = burstSim.Bursts.width(ind);
%     actualfreq(i,1)=burstSim.Bursts.BurstFreq(ind);
%     actualamplitude(i,1)=burstSim.Bursts.BurstAmp(ind);
%     actualwidth(i,1)=burstSim.Bursts.width(ind);
%     actualphase(i,1)=burstSim.Bursts.BurstPhase(ind);
%     actualcycNum(i,1)=burstSim.Bursts.BurstCycNum(ind);
%     if(mod(wBurst,2)==0)
%         wBurst=(wBurst/2);
%         startInd=(wBurst-1);
%         endInd=wBurst;
%     else
%         wBurst=round(wBurst/2);
%         startInd=(wBurst-2);
%         endInd=wBurst;
%     end
%     Index(i,1)=startInd;
%     Index(i,2)=endInd;
% end
% for u=1:length(trueBursts)
%     midu=trueBursts(u);
%     true_ind(u,1)=midu-Index(u,1);
%     true_ind(u,2)=midu+Index(u,2);
% end
%% Find how many are overlapped bursts
b=1; count_overlap=0;
for n=2:length(trueindexs)-1
    if max(trueindexs(1:(n-1),2))>trueindexs(n,1)|| min(trueindexs(n+1:length(trueindexs),1))<trueindexs(n,2)
        count_overlap=count_overlap+1;
        overlappedBursts(b,1)=n;
        b=b+1;
    end
end
%% seperating groundtruth signals
%z=trueindexs(:,2)-trueindexs(:,1);
%[max_z,max_zind]=max(z);
h=linspace(1,20000,20000);
non_overlappedBursts=setdiff(h,overlappedBursts);
data=zeros(length(trueindexs),max(wid));
for k=1:length(trueindexs)
    a=burstTrace(1,trueindexs(k,1):trueindexs(k,2));
    for l=1:length(a)
        data(k,l)= a(1,l);
    end
end
data=data(:,1:2000);
data=data(non_overlappedBursts,:);
% index=find(wid<=2000);
% e=e(index,:);
%% Seperating Noisy data
% noisyBurstTrace=(sum(simLFP))/chanNum;
data_noisy=zeros(length(trueindexs),max(wid));
for k=1:length(trueindexs)
    a=meansimLFP(1,trueindexs(k,1):trueindexs(k,2));
    for l=1:length(a)
        data_noisy(k,l)= a(1,l);
    end
end
data_noisy=data_noisy(:,1:2000);
data_noisy=data_noisy(non_overlappedBursts,:);
%% Creating labels
% A = java_array('java.lang.String', length(true_ind));
data_count=length(non_overlappedBursts);
Max_Min = [8 12;12 30;30 80;4 8];
Freq=burstFreq';
Freq=Freq(non_overlappedBursts);
for count_for=1:data_count
    labels(count_for,:)=blanks(5);
    if (Freq(count_for) > Max_Min(4,1)) && (Freq(count_for) < Max_Min(4,2))
        labels(count_for,:)= 'Theta';
    elseif (Freq(count_for) > Max_Min(1,1)) && (Freq(count_for) < Max_Min(1,2))
       labels(count_for,:)= 'Alpha';
    elseif (Freq(count_for) > Max_Min(2,1)) && (Freq(count_for) < Max_Min(2,2))
       labels(count_for,:)= 'Beta ';
    elseif (Freq(count_for) > Max_Min(3,1)) && (Freq(count_for) < Max_Min(3,2))
       labels(count_for,:)= 'Gamma';
    end
end
labels=labels;
%% Creating Freq,Amp,cyc Data
burstFreq=burstFreq';
burstFreq=burstFreq(non_overlappedBursts);

burstAmp=burstAmp';
burstAmp=burstAmp(non_overlappedBursts);

burstCycNum=burstCycNum';
burstCycNum=burstCycNum(non_overlappedBursts);

sigma=sigma';
sigma=sigma(non_overlappedBursts);
