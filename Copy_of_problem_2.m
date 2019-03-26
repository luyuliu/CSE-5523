
test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*7, ones(1000,1)*9);

N=2000;
d=784;

w = ones(d+1, 1)*0; % Bias trick
train=[ones(n,1);train]

nIterations = 200;
it = 0;
learningRate = 10e-7;

objFun = @(X, Y, w) ( transpose(w)*transpose(X)*X*w-w*transpose(w)*transpose(X)*Y+transpose(Y)*Y)/2000;
gradient = @(X, Y, w) 2*transpose(X)*X*w-2*transpose(X)*Y;

while it <= nIterations
  % computes objective
  R = objFun(train, label, w);
  
  grad = gradient(train, label, w);
  w = w - learningRate*grad;
  

  it = it + 1;
  R
    
  if isnan(w(1))
     break;
  end
  if it > nIterations
    break;
  end
  %transpose(w(1:20))
end


