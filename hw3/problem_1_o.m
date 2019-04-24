train = load('train79.mat');
train=train.d79;
test = load('test79.mat');
test=test.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

%% Decision trees
DT = fitctree(train,label, 'CrossVal','on');
crossValFun = @(x)sum(x.IsBranch);
crossValResult = cellfun(crossValFun, DT.Trained);
figure;
histogram(crossValResult) % Find the maximum split.

lossList = zeros(38,1);

for split=1:38
    sDT=fitctree(train, label, 'CrossVal','on','MaxNumSplits',split);
    lossDT = kfoldLoss(sDT);
    lossList(split)=lossDT;
end

[lossListSorted , inx] = sort(lossList);

sDT=fitctree(train, label, 'CrossVal','on','MaxNumSplits',inx(1));
L = kfoldLoss(sDT)


%% Bagged trees

for split = 40:60
    t=templateTree('MaxNumSplits',inx(1));
    BAT = fitcensemble(train,label,'Learners',t,'Method','Bag','CrossVal','on');
end
kFoldLossFunBaT=kfoldLoss(BAT,'mode','cumulative');
kFoldLossBaT = kFoldLossFunBaT(end)
figure(1)
plot(kFoldLossFunBaT,'r.');
xlabel('Learning Cycles');
ylabel('Loss Rate');

%% Boosted trees
BOT = fitcensemble(train,label,'Method','AdaBoostM1','NumLearningCycles',500,'Kfold',8);
kFoldLossFunBoT=kfoldLoss(BOT,'mode','cumulative');
kFoldLossBoT = kFoldLossFunBoT(end)
figure(2)
plot(kFoldLossFunBoT,'r.');
xlabel('Learning Cycles');
ylabel('Loss Rate');