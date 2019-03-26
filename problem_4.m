
N=50; % Change on demand
d=784;
testRaw=load('test79.mat');
testRaw=testRaw.d79;
trainRaw=load('train79.mat');
trainRaw=trainRaw.d79;
label = vertcat(ones(N,1)*1, ones(N,1)*-1);
train = [trainRaw(1:N,:);trainRaw(1001:1000+N,:)];
test = [testRaw(1:N,:);testRaw(1001:1000+N,:)];
w = ones(d+1, 1)*0; % Bias trick
train=[ones(N*2,1),train];
test=[ones(N*2,1),test];
nIterations = 2000;
it = 0;
learningRate = 1*10e-11;
objFun = @(X, Y, w) ( transpose(w)*transpose(X)*X*w-2*w*transpose(w)*transpose(X)*Y+transpose(Y)*Y)/N;
gradient = @(X, Y, w) 2*transpose(X)*X*w-2*transpose(X)*Y;

loss_rates=[];
wBackup = ones(d+1, 1)*0
while it <= nIterations
  % computes objective
  R = objFun(train, label, w);
  grad = gradient(train, label, w);
  wBackup=w;
  w = w - learningRate*grad;
  it = it + 1;
  if isnan(w(1))
     break;
  end
  if sum(abs(w-wBackup))<10e-6
      break;
  end
  if it > nIterations
    break;
  end
  result = sign(test*w);
  loss = 1/2*(sum(abs(result-label)))/N
  loss_rates=[loss_rates;loss];
end

