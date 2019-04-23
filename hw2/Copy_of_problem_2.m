
test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

N=2000;
d=784;

w = ones(d+1, 1)*0; % Bias trick
train=[ones(N,1),train];
test=[ones(N,1),test];

nIterations = 2000;
it = 0;
learningRate = 1*10e-11;

objFun = @(X, Y, w) ( transpose(w)*transpose(X)*X*w-2*w*transpose(w)*transpose(X)*Y+transpose(Y)*Y)/N;
gradient = @(X, Y, w) 2*transpose(X)*X*w-2*transpose(X)*Y;


wBackup = ones(d+1, 1)*0
while it <= nIterations
  % computes objective
  R = objFun(train, label, w);
  
  grad = gradient(train, label, w);
  wBackup=w;
  w = w - learningRate*grad;
  

  it = it + 1;
  R
    
  if isnan(w(1))
     break;
  end
  if sum(abs(w-wBackup))<10e-5
      break;
  end
  if it > nIterations
    break;
  end
  %transpose(w(1:20))
end

result = sign(test*w);
loss = 1/2*(sum(abs(result-label)))/N


