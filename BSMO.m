%%    Binary Starling Murmuration Optimizer Algorithm to Select Effective Features from Medical Data
%        Mohammad H Nadimi-Shahraki, Zahra Asghari Varzaneh, Hoda Zamani, Seyedali Mirjalili
%        Journal Applied Sciences, Publisher Multidisciplinary Digital Publishing Institute
%        https://doi.org/10.3390/app13010564
%------------------------------------------------------------------------------------------------------------
function[Fitness,Accuracy,FinalFitness,BinaryPos]= BSMO(lu,PopSize,MaxIt,ProblemSize,Dataset,Label)

FlockNum = 5;
% Randomly distribute starlings in the search space
Pop = repmat(lu(1,:),PopSize,1)+rand(PopSize,ProblemSize).*(repmat(lu(2,:)- lu(1,:),PopSize,1));
NewbinaryPos = Pop;
NewbinaryPos(NewbinaryPos>=0.5)=1;
NewbinaryPos(NewbinaryPos<0.5)=0;

for i = 1:size(NewbinaryPos,1)
    output = Knn_Classifier(NewbinaryPos(i,:),Dataset,Label);
    Pnt(i).Fit = output.Fit ;
    Pnt(i).Accuracy = output.Accuracy ;
    Pnt(i).binaryPos = NewbinaryPos(i,:);
    Pnt(i).selected_features= output.selected_features;
end
Accuracy(1) = max([Pnt.Accuracy]);

Val = [Pnt.Fit];
Pop = NewbinaryPos;

%% Starling position
[~,sorted_index] = sort(Val,'ascend');
Pop = Pop(sorted_index,:);
Val = Val(sorted_index);
%% Global best soluation (Starling)
BestVal = min(Val);
BestPos = Pop(1,:); % The global position obtained so far
PosConvergance = Pop(1:20,:);
t = 1;
while  t <= MaxIt
    %% A) Seprating strategy
    %1)  A portion of the starlings is randomly selected from the starling
    %    population to construct a separated population Psep or PopSep
    Sep_rate = log(t+ProblemSize)/(log(MaxIt)*2);    % Eq.(1)
    SepSize  = max(round(Sep_rate * size(Pop,1)), 2);
    SepInd   = (randperm(size(Pop,1),SepSize))';
    PopSep   = Pop(SepInd,:);       % A separated population Psep or PopSep
    %2) Definition 1 (Separation Operator)
    U        = rand(size(PopSep,1),size(PopSep,2));
    % Fsep = normrnd(U,0.1);
    Fsep = QHO(U);
    term_pos = Fsep == -1; Fsep(term_pos) = 0;
    Fsep     = min(Fsep, 1);Fsep = max(Fsep, 0);
    %3) Separating search strategy (diversity)
    Param2Change = randperm(ProblemSize,randi(ProblemSize));
    r1 = randperm (size(PopSep, 1),size(PopSep, 1))';
    Xr_hat = [PosConvergance; PopSep];
    r11 = randperm (size(Xr_hat, 1),size(PopSep, 1))';
    % Eq.(2)
    PopSep(:,Param2Change) = BestPos(1,Param2Change)+ Fsep(:,Param2Change).* (Xr_hat(r11,Param2Change) - Pop(r1,Param2Change));
    NewPop (SepInd,:) = PopSep; 
    %% B) Definition 2 (Dynamic Multi-Flock Construction)
    Ind    = (setdiff(1:PopSize,SepInd))';   %  Diving and whirling subpopulation
    %1) Determin the population of diving and whirling
    [EVal,sorted_index] = sort(Val(Ind), 'ascend');
    EPop = Pop(sorted_index,:);
    %2) Multi flock construction
    Rep_Set = (1:FlockNum)';
    Rep_Pos(:,1:ProblemSize) = EPop(Rep_Set,:); % Representative members
    SubPop_Num  = (FlockNum+1:size(Ind,1))';
    
    FlockPopSize = floor((size(Ind,1) - mod(size(Ind,1),FlockNum))/(FlockNum));%  The number of subpoulation
    Result = cell(1, FlockNum);
    for k = 1:FlockNum
        if k ==  FlockNum
            Result{k} = [FlockNum;SubPop_Num(((k-1)*(FlockPopSize-1)+1):end)];
        else
            Result{k} = [Rep_Set(k);SubPop_Num(((k-1)*(FlockPopSize-1)+1):k*(FlockPopSize-1))];
        end
        F_Mean(k)   = mean(EVal(Result{k}));
    end
    %3) Definition 3 (Flock Quality)
    F_Quality =(sum(F_Mean))./repmat(F_Mean,FlockNum, 1); % Eq.(7)
    F_Quality = F_Quality(1,:);
    
    %% C) Whirling search strategy (exploitation)
    %1) Whirling Population
    [~,FW] = find(F_Quality > mean(F_Quality)) ;  % Eq.(8)
    %2)  Whirling Movement
    for i = 1:size(FW,2)
        WhirIdx      = Result{FW(i)};
        WhirlingPop = Pop(Ind(WhirIdx),:);  % Whirling Population or Xi
        I = randperm(size(Rep_Set,1),1)';
        X_RW = Rep_Pos(Rep_Set(I,:),:);  % randomly selected from the representative members of those flocks
        idx = randperm (size(WhirlingPop,1),size(WhirlingPop,1))';
        XN = WhirlingPop(idx, :);
        WhirlingPop = WhirlingPop + cos(rand).*(X_RW - XN);  %Eq. (16)
        NewPop (Ind(WhirIdx),:) = WhirlingPop;
    end
    %     clearvars WhirlingPop; clearvars WhirlingVal;
    %% D)  Diving search strategy (exploration)
    [~,FD] = find(F_Quality <=  mean(F_Quality)); % Eq.(8)
    for q = 1: size(FD,2)
        DivIdx  = Result{FD(q)};        % Flock q
        DivingPop = Pop(Ind(DivIdx),:);   % Diving popualtion
        R_D = repmat (Rep_Pos(FD(q),:),size(DivingPop,1),1); % representative member of the flock
        % Definition 4 (Quantum Random Dive Operator)
        %  1) Inverse Gaussian Distribution % Eq. (15)
        mu  = 1;            lambda = 20;
        IGD_1 = random('InverseGaussian',mu,lambda,size(DivingPop,1),1);
        pos = find(IGD_1>1);
        while ~isempty(pos)
            IGD_1(pos) = random('InverseGaussian',mu,lambda,size(pos,1),1);
            pos = find(IGD_1>1);
        end
        IGD_2 = random('InverseGaussian',mu,lambda,size(DivingPop,1),1);
        pos = find(IGD_2>1);
        while ~isempty(pos)
            IGD_2(pos) = random('InverseGaussian',mu,lambda,size(pos,1),1);
            pos = find(IGD_2>1);
        end
        %  2) Rotation matrix
        C  = [cos(rand).*(angle(exp(1i.*rand./2))) sin(.5).*(angle(exp(1i.*1.8./2)));...
            -sin(.5).*(angle(exp(-1i.*1.8./2))) cos(-.5*rand).*(angle(exp(-1i.*(-.5*rand)./2)))];
        UP   = C(1,1).* IGD_1 + C(1,2).* IGD_1;  % Eq. (14)
        Down = C(2,1).* IGD_2 + C(2,2).* IGD_2;
        Downward  =  find (UP <= Down); %  Downward quantum probability defined in Definition 4
        Upward=  find (UP > Down);      %  Upward quantum probability defined in Definition 4
        
        % 2) Upward quantum random dive
        a1  = randperm(size(DivingPop,1));
        UnionSet = [Pop ; PosConvergance]; % Union set of the current population and the best starlings set
        X_j = UnionSet(a1,:); % randomly selected from the union set
        
        % 3) random position selected from the current population and
        %  the best starlings set obtained from the first iteration so far.
        VminPop = min(UnionSet,[],1) ;
        VmaxPop = max(UnionSet,[],1) ;
        Si_delta = repmat(VminPop(1,:),size(DivingPop,1),1)+ rand(size(DivingPop,1),ProblemSize).*(repmat(VmaxPop(1,:)- VminPop(1,:),size(DivingPop,1),1));
        
        % 4) Movement
        r1 = randperm(size(DivingPop,1),size(DivingPop,1))';
        DivingPop(Downward,:) = R_D(Downward,:)-Down(Downward).*(DivingPop(Downward,:)- DivingPop(r1(Downward),:));     % Eq.(12)
        DivingPop(Upward,:)   = R_D(Upward,:)+ UP(Upward).*(DivingPop(Upward,:)- X_j(Upward,:)+ Si_delta(Upward,:));    % Eq.(13)   
        NewPop (Ind(DivIdx),:)= DivingPop;
    end
    %     clearvars DivingPop; clearvars DivingVal;
    %% Transfer functions
    old_pos = Pop;
    NewbinaryPos = QTFs(NewPop,size(NewPop,1),ProblemSize);
    
    for Mid = 1:size(NewPop,1)
        while sum(NewbinaryPos(Mid,:),2) == 0
            NewbinaryPos(Mid,:) = old_pos(Mid,:);
        end
        output = Knn_Classifier(NewbinaryPos(Mid,:),Dataset,Label);
        NewPnt(Mid).Fit = output.Fit ;
        NewPnt(Mid).Accuracy = output.Accuracy ;
        NewPnt(Mid).binaryPos = NewbinaryPos(Mid,:); 
        NewPnt(Mid).selected_features=output.selected_features;
    end
%     binaryVal  = [Pnt.Fit];
    %3) Update the new position if is better
%     [Pop,Val] = Update(Pop,Val,binaryPos,binaryVal,ProblemSize);
    [Pop,Pirnt,binaryVal] = Update(NewbinaryPos,NewPnt,Pop,Pnt,ProblemSize);
    
    for i = 1:PopSize
        Pnt(i).Fit =Pirnt.Fit(i);
        Pnt(i).Accuracy =Pirnt.Accuracy(i);
        Pnt(i).selected_features= sum (Pop(i,:));
        Pnt(i).binaryPos = Pop(i,:);
    end
    Val = [Pnt.Fit];
    
    Accuracy(t) = max([Pnt.Accuracy]); 
    [BestVal , IdBst] = min(Val);
    BestPos = Pop(IdBst,:);
    PosConvergance(t,:) = BestPos;
    Fitness (t) = min(BestVal);
    t =  t  + 1;
end
[FinalFitness,IdB ] = min (Fitness);
BinaryPos = PosConvergance(IdB,:);
FinalAccuracy = max (Accuracy);
end



%% Quantum harmonic oscillator (QHO)
function Out = QHO(y)
n = 1;
h_bar =1.05457168e-34; m=9.1093826e-31; k=2*pi*1e6;
alpha=(m*k/h_bar)^(1/4);
H = zeros(n+1,n+1);
H(1,1)=1; H(2,2)=2;
for i=3:n+1
    H(i,:)= 2*[0,H(i-1,1:end-1)]-2*(i-2)*H(i-2,:);
end
Tmp = sqrt(alpha./((2.^n).*factorial(n).*sqrt(pi)));
Phi=cell(size(n));
for i=1:length(n)
    Phi{i}=zeros(size(y));
    for j=1:max(n)+1
        Phi{i}=Phi{i}+H(n(i)+1,j)*((alpha*y).^(j-1));
    end
    Phi{i}=[y,exp(-0.5.*alpha^2.*y.^2).*Tmp(i).*Phi{i}];
end
Out = cell2mat(Phi);
Out = Out(1:size(y,1),1:size(y,2));
end

