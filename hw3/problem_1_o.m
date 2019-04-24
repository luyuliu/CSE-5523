train = load('train79.mat');
train=train.d79;
test = load('test79.mat');
test=test.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

N=2000;

%% Decision trees
DT = fitctree(train,label, 'CrossVal','on');
crossValFun = @(x)sum(x.IsBranch);
crossValResult = cellfun(crossValFun, DT.Trained);
figure;
max(crossValResult) % Find the maximum split.

lossList = zeros(38,1);

for split=1:38
    sDT=fitctree(train, label, 'CrossVal','on','MaxNumSplits',split);
    lossDT = kfoldLoss(sDT);
    lossList(split)=lossDT;
end

[lossListSorted , inx] = sort(lossList);

sDT=fitctree(train, label, 'MaxNumSplits',inx(1));
testDTResult = sDT.predict(test);
DTDiff = testDTResult - label;
DTLoss = transpose(DTDiff)*DTDiff/4/N


%% Bagged trees

BT = fitcensemble(train,label,'Method','Bag','NumLearningCycles',500,'CrossVal','on');
kFoldLossList=kfoldLoss(BT,'mode','cumulative');
[optimalBT, index]=sort(kFoldLossList);
BTOptimal=fitcensemble(train,label,'NumLearningCycles',index,'Method','Bag');
testBTResult=BTOptimal.predict(test);
BTDiff = testBTResult - label;
BTLoss = transpose(BTDiff)*BTDiff/4/N


%% Boosted trees
BoT = fitcensemble(train,label,'Method','AdaBoostM1','NumLearningCycles',500,'CrossVal','on');
kFoldLossBoTList=kfoldLoss(BoT,'mode','cumulative');
[optimalBoT, index2]=sort(kFoldLossBoTList);
BoTOptimal=fitcensemble(train,label,'NumLearningCycles',index2,'Method','AdaBoostM1');
testBoTResult=BoTOptimal.predict(test);
BoTDiff = testBoTResult - label;
BoTLoss = transpose(BoTDiff)*BoTDiff/4/N