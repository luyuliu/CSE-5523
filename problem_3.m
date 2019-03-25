testRaw=load('test79.mat');
testRaw=testRaw.d79;
trainRaw=load('train79.mat');
trainRaw=trainRaw.d79;
labelRaw = vertcat(ones(1000,1)*7, ones(1000,1)*9);



trainPCA = pca(transpose(trainRaw));
testPCA = pca(transpose(testRaw));

NList = [25:25:1000];

SVMLossList=ones(length(NList),1);
LSLCLossList=ones(length(NList),1);

for i = 1:length(NList)
    N= NList(i);
    
    train = [trainPCA(1:N,:);trainPCA(1001:1000+N,:)];
    test = [testPCA(1:N,:);testPCA(1001:1000+N,:)];
    label = [ones(N,1)*7; ones(N,1)*9];
    
    % SVM
    SVMModel = fitcsvm(train, label);

    SVMResult = predict(SVMModel, test);

    SVMDiff = SVMResult - label;

    SVMLossList(i) = transpose(SVMDiff)*SVMDiff/4/(N*2);


    % Least Square Linear
    lambda = 0.00000001; % lambda = 1/C
    LSLCModel = fitclinear(train,label,'Regularization','ridge','lambda',lambda);
    LSLCResult = predict(LSLCModel,test);
    diff=LSLCResult-label;
    LSLCLossList(i)=transpose(diff)*diff/4/(N*2);
    
    
end


