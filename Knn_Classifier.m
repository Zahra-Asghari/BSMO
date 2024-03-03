%%    Binary Starling Murmuration Optimizer Algorithm to Select Effective Features from Medical Data
%        Mohammad H Nadimi-Shahraki, Zahra Asghari Varzaneh, Hoda Zamani, Seyedali Mirjalili
%        Journal Applied Sciences, Publisher Multidisciplinary Digital Publishing Institute
%        https://doi.org/10.3390/app13010564
%------------------------------------------------------------------------------------------------------------

function Output = Knn_Classifier(solution,Dataset,Label)
global Training Testing;
idSel = solution==1;
meas   = Dataset(:,idSel);

class = fitcknn(meas(Training,:),Label(Training,:),'NumNeighbors',5,'Distance','euclidean');
[prediction,~] = predict(class,meas(Testing,:));
Actual = Label(Testing,:);
MisClass= sum(Actual ~= prediction)/length(Actual);
Output.Accuracy = (1 - MisClass) *100;
%% Fitness function
b = 0.01; a = 1-b; % parameters for fitness function
Output.selected_features = size(meas,2);
Output.Fit = (a * MisClass) + (b * (Output.selected_features/size(Dataset,2)));
end