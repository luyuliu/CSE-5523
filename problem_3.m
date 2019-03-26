testRaw=load('test79.mat');
testRaw=testRaw.d79;
trainRaw=load('train79.mat');
trainRaw=trainRaw.d79;
labelRaw = vertcat(ones(1000,1)*0, ones(1000,1)*1);


trainPCA_coeff =pca(trainRaw,'NumComponents', 400);
testPCA_coeff =pca(testRaw,'NumComponents', 400);
trainPCA = trainRaw*trainPCA_coeff;
testPCA = testRaw*trainPCA_coeff;


NList = [25:25:1000];

lslcLossList=ones(length(NList),1);
svmLossList=ones(length(NList),1);

for i = 1:length(NList)
    N= NList(i);
    
    train = [trainPCA(1:N,:);trainPCA(1001:1000+N,:)];
    test = [testPCA(1:N,:);testPCA(1001:1000+N,:)];
    label = [ones(N,1)*0; ones(N,1)];
    
    % SVM
    svmModel = fitclinear(train,label);
    svmResult = predict(svmModel,test);
    diff=abs(svmResult-label)/2;
    svmLossList(i)=sum(diff)/(N*2);
    
    % LSLC
    lslcW = lsqlin(train, label);
    lslcResult = sign (test*lslcW);
    diff =abs(lslcResult-label)/2;
    lslcLossList(i)=sum(diff)/(N*2);
    
    
end


