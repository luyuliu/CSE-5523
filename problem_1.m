test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

N=2000;

% Least Square Linear
lambda = logspace(-8,1,20); % lambda = 1/C
LSLCModel = fitclinear(train,label,'Regularization','ridge','lambda',lambda);
LSLCResult = predict(LSLCModel,train);
LSLCLossList=zeros(length(lambda),1);
% prediction accuracy rate
for i=1:length(lambda)
    diff=LSLCResult(:,i)-label;
    LSLCLossList(i)=(transpose(diff)*diff)/(4*N);
end

% SVM
SVMModel = fitcsvm(train, label);

SVMResult = predict(SVMModel, test);

SVMDiff = SVMResult - label;

loss = transpose(SVMDiff)*SVMDiff/(4*N)


