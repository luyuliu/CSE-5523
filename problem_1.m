test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*7, ones(1000,1)*9);

N=2000

% SVM
SVMModel = fitcsvm(train, label);

inferenceResult = predict(SVMModel, test);

diff = inferenceResult - label;

loss = transpose(diff)*diff/4/2000


% Least Square Linear
b = [ones(1000, 1); -ones(1000, 1)]

z = lsqlin(train, b);

