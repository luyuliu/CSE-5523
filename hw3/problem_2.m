test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

N=2000;
d=784;
k=2;

train_7 = train (1:1000,:);
train_9 = train (1001:2000, :);

[PCA, newTrain] = PCA_eig(train, k)



%% Visualization

%figure(1)
%scatter(newTrain(1:1000,1),newTrain(1:1000,2),'.b');
%hold on
%scatter(newTrain(1001:2000,1),newTrain(1001:2000,2),'.r');

figure(2)
subplot(1,2,1)
x = reshape (PCA(:,1),28,28);
pcolor(x);
title('Eigendigit 1');
daspect([1 1 1])

subplot(1,2,2)
x = reshape (PCA(:,2),28,28);
pcolor(x);
title('Eigendigit 2');
daspect([1 1 1])