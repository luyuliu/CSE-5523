% HW3 of Machine Learning Class Problem 1 about DT and Ensemble Learning
train = load('train79.mat');
X_train=train.d79;
test = load('test79.mat');
X_test=test.d79;
% the number of observated data in training dataset
n = length(train.d79);
y1=7*ones(1000,1);
y2=9*ones(1000,1);
Y = [y1;y2];
rng(123); % For reproducibility

%% fit a decision tree with 10-fold cross validation
Dtree = fitctree(X_train,Y, 'CrossVal','on');
numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, Dtree.Trained);
figure;
histogram(mdlDefaultNumSplits)

MaxNumSplitsList=1:1:37;
ErrorList=ones(37,1)*-1;

% % Decision tree
% for i=1:37
%     max_num_splits=MaxNumSplitsList(i);
%     Dtree_cv = fitctree(X_train,Y, 'CrossVal','on','MaxNumSplits',max_num_splits);
%     error_DT = kfoldLoss(Dtree_cv);
%     ErrorList(i,:)=error_DT;
% end
% m=min(ErrorList);
% idx=find(ErrorList==m);
% 
% DtreeFinal=fitctree(X_train,Y,'MaxNumSplits',MaxNumSplitsList(idx));
% Y_predicted=predict(DtreeFinal,X_test);
% diff=abs(Y_predicted-Y)/2;
% DT_err = (sum(diff))/2000
%% Bagged trees
BaggedTree = fitcensemble(train.d79,Y,'Method','Bag','NumLearningCycles',200,'Kfold',10);
kflc_bagged=kfoldLoss(BaggedTree,'mode','cumulative');
Error_BaggedT = kflc_bagged(end)
figure(1)
plot(kflc_bagged,'r.');
title('performance of bagged decision trees');
xlabel('Num of Learning Cycles');
ylabel('Misclassification Rate');

%% Boosted trees
BoostedTree = fitcensemble(train.d79,Y,'Method','AdaBoostM1','NumLearningCycles',200,'Kfold',10);
kflc_boo=kfoldLoss(BoostedTree,'mode','cumulative');
Error_BooT = kflc_boo(end)
figure(2)
plot(kflc_boo,'r.');
xlabel('Num of Learning Cycles');
ylabel('Misclassification Rate');