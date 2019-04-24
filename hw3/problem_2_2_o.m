% HW3 of Machine Learning Class Problem 2 about PCA subproblem 2
clear
train = load('train79.mat');
test = load('test79.mat');
X=train.d79;
% the number of observated data in training dataset
n = length(train.d79);
[X_norm, mu, sigma] = featureNormalize(X);
% Both two classes
[U, S] = pca_jialin(X_norm);
k=2;
Ureduce=U(:, 1:k);

figure(1)
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

% Just for class 7
[U, S] = pca_jialin(X_norm(1:1000,:));
k=2;
Ureduce=U(:, 1:k);

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

% Just for class 9
[U, S] = pca_jialin(X_norm(1001:2000,:));
k=2;
Ureduce=U(:, 1:k);

figure(3)
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