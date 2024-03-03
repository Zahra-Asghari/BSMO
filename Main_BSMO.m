%%    Binary Starling Murmuration Optimizer Algorithm to Select Effective Features from Medical Data
%        Mohammad H Nadimi-Shahraki, Zahra Asghari Varzaneh, Hoda Zamani, Seyedali Mirjalili
%        Journal Applied Sciences, Publisher Multidisciplinary Digital Publishing Institute
%        https://doi.org/10.3390/app13010564
%------------------------------------------------------------------------------------------------------------
clear
clc
warning off;
fprintf('==================================================================================\n');
fprintf('   Binary Starling Murmuration Optimizer Algorithm to Select Effective Features from Medical Data\n ')
fprintf('       ------------------------------------------------------------------\n');
%%  Initial parameter values
ID  = 1;                % S1 transfer function
run = 1;
Searchagents = 20;      % The number of search agents
Max_It = 300;           % The maximum number of iterationS
%% Hepatitis data from the UCI machine learning repository
global Training Testing;
CaseName ='hepatitis.data';
Data = load('hepatitis.data');
Dataset = Data(:,1:end-1);
Label = Data(:,end);
Samples = size(Dataset,1);
Features = size(Dataset,2);
D = size(Dataset,2);
lu = [zeros(1, D); ones(1, D)];
rate = 0.70 ;
idx = randperm(Samples)  ;
Training = idx(1:round(rate*Samples)) ; 
Testing = idx(round(rate*Samples)+1:end);
 
%% The BSMO algorithm 
fprintf('       Numbers of Samples = %d, Numbers of Features = %d\n',Samples,Features)
[Convergence,ConvAccuracy,Fbest,Best_pos]=  BSMO(lu,Searchagents,Max_It,D,Dataset,Label);

BSMO_Result.NumFeatures = sum (Best_pos==1);
BSMO_Result.Convergence = Convergence;
BSMO_Result.Fitness = min(Convergence);
BSMO_Result.Accuracy = max(ConvAccuracy);
fprintf('%d th run, BestFit = %7.4f, Accuracy= %7.4f\n', run , Fbest,BSMO_Result.Accuracy)
 
 