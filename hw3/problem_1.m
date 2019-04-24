test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

N=2000;

% Decision tree
DT = fitctree(train,label);
testDTResult = DT.predict(test);
DTDiff = testDTResult - label;
DTLoss = transpose(DTDiff)*DTDiff/4/N

% Bagged tree
BT = fitensemble(train,label,'Bag',500,'Tree','Type', 'classification');
testBTResult = (BT.predict(test));
BTDiff = testBTResult - label;
BTLoss = transpose(BTDiff)*BTDiff/4/N

% Boosted tree
BO = fitensemble(train,label,'AdaBoostM1',500,'Tree');
testBOResult = (BO.predict(test));
BODiff = testBOResult - label;
BOLoss = transpose(BODiff)*BODiff/4/N

