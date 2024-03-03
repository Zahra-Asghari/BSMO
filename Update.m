%%    Binary Starling Murmuration Optimizer Algorithm to Select Effective Features from Medical Data
%        Mohammad H Nadimi-Shahraki, Zahra Asghari Varzaneh, Hoda Zamani, Seyedali Mirjalili
%        Journal Applied Sciences, Publisher Multidisciplinary Digital Publishing Institute
%        https://doi.org/10.3390/app13010564
%------------------------------------------------------------------------------------------------------------

function [OutPop,Out,Fit]= Update(Pop,NewOutput,RealPop,RealOutput,D)

NewFit = [NewOutput.Fit]; 
NewAcc = [NewOutput.Accuracy]; 
RealFit = [RealOutput.Fit]; 
RealAcc = [RealOutput.Accuracy];
tmp  = (NewFit<= RealFit);
tmp1 = repmat(tmp',1,D);
OutPop  = tmp1.* Pop + (1-tmp1).*RealPop;
Out.Fit  = tmp.* NewFit + (1-tmp).*RealFit;
Out.Accuracy  = tmp.* NewAcc + (1-tmp).*RealAcc;
Fit = [Out.Fit];
end