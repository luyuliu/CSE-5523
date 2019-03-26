
function problem_2()
test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

N=2000;
d=784;

w = zeros(d, 1);


nIterations = 100;
it = 0;
learningRate = 0.00001;

while it <= nIterations
  % computes objective
  R = objFunction(train, label, w);
  
  grad = objFunctionDev(train, label, w);
  w = w - learningRate*grad;

  it = it + 1;

  if it > nIterations
    break;
  end
end

w
end 

objFun = @(X, Y, w) ( transpose(w)*transpose(X)*X*w-w*transpose(w)*transpose(X)*Y+transpose(Y)*Y)/2000;

function R = objFunction(X, Y, w)
    R =( transpose(w)*transpose(X)*X*w-w*transpose(w)*transpose(X)*Y+transpose(Y)*Y)/2000;
end

objFunPrime = @(X, Y, w) 2*transpose(X)*X*w-2*transpose(X)*Y;

function RPrime = objFunctionDev(X,Y,w)
    RPrime = 2*transpose(X)*X*w-2*transpose(X)*Y;
end
