

% HW2 of Machine Learning Class Problem 3
train = load('train79.mat');
test = load('test79.mat');
% the number of observated data in training dataset
n = length(train.d79);
train_size=[25:25:1000];
l=length(train_size);
%% PCA reduce the dimension
train_coeff = pca(train.d79,'NumComponents', 400);
test_coeff= pca(test.d79,'NumComponents', 400);
train_reduced=train.d79*train_coeff;
test_reduced=test.d79*train_coeff;
svm_err_list=ones(l,1);
ls_err_list=ones(l,1);
for i=1:l
    train_rn=train_reduced([1:train_size(i) 1001:1000+train_size(i)],:);
    test_rn=test_reduced([1:train_size(i) 1001:1000+train_size(i)],:);
    y1=7*ones(train_size(i),1);
    y2=9*ones(train_size(i),1);
    Y = [y1;y2];
    
    %% SVM Classifier
    SVM_Md1 = fitclinear(train_rn,Y);
    % test the SVM classifier
    [SVM_label,SVM_score] = predict(SVM_Md1,test_rn);
    % prediction accuracy rate
    diff=abs(SVM_label-Y)/2;
    SVM_err = sum(diff)/(train_size(i)*2);
    svm_err_list(i)=SVM_err;
    %% least squares linear classifier
    Y_ls=[ones(train_size(i),1);-1*ones(train_size(i),1)];
    X_ls=[train_rn,ones(train_size(i)*2,1)];
    W=lsqlin(X_ls,Y_ls);
    X_ls_test=[test_rn,ones(train_size(i)*2,1)];
    LS_label=sign(X_ls_test*W);
    LS_err = 1/2*(sum(abs(Y_ls-LS_label)))/(train_size(i)*2);
    ls_err_list(i)=LS_err;
end