%%    Binary Starling Murmuration Optimizer Algorithm to Select Effective Features from Medical Data
%        Mohammad H Nadimi-Shahraki, Zahra Asghari Varzaneh, Hoda Zamani, Seyedali Mirjalili
%        Journal Applied Sciences, Publisher Multidisciplinary Digital Publishing Institute
%        https://doi.org/10.3390/app13010564
%------------------------------------------------------------------------------------------------------------
function Bstep = QTFs(pos,N,dim)

s=1./(1+exp(-2.*pos)); %S1 transfer function
Bstep=rand(N,dim)<s;
end