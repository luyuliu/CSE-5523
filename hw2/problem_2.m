
test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

N=2000;
d=784;

w = pinv(train'*train)*train'*label;


result = sign(test*w);
loss = 1/2*(sum(abs(result-label)))/N