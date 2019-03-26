test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

N=2000;

% SVM
lambda = logspace(-8,1,20); % lambda = 1/C
svmModel = fitclinear(train,label,'Regularization','ridge','lambda',lambda);
svmResult = predict(svmModel,train);
svmLossList=zeros(length(lambda),1);
% prediction accuracy rate
for i=1:length(lambda)
    diff=svmResult(:,i)-label;
    svmLossList(i)=(transpose(diff)*diff)/(4*N);
end

% LSLC
w=lsqlin(train,label);
testResult=sign(test*w);
svmLoss = 1/2*(sum(abs(testResult-label)))/2000

