test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

N=2000;

X=train;
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);
sigma = std(X_norm);

[U, S] = pca_jialin(X_norm);
k=2;
Ureduce=U(:, 1:k);
Z=X*Ureduce;
% 
% figure(1)
% scatter(Z(1:1000,1),Z(1:1000,2),'.b');
% hold on
% scatter(Z(1001:2000,1),Z(1001:2000,2),'.r');

% subproblem 2
figure(2)
subplot(1,2,1)
colormap(gray);
x = reshape (Ureduce(:,1),28,28);
y = x(:,28:-1:1);
pcolor(y);
title('subplot1: first eigen-digit');
daspect([1 1 1])
subplot(1,2,2)
colormap(gray);
x = reshape (Ureduce(:,2),28,28);
y = x(:,28:-1:1);
pcolor(y);
title('subplot2: second eigen-digit');
daspect([1 1 1])