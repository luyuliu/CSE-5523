test=load('test79.mat');
test=test.d79;
train=load('train79.mat');
train=train.d79;
label = vertcat(ones(1000,1)*1, ones(1000,1)*-1);

n=size(train, 1);
d=size(train, 2);

kList=[2,5,10,50];
lossList=ones(length(kList),1);
for i=1:length(kList)
    k=kList(i);
    [kMeansResult] = kmeans(train,k); % K-means
    for cluster=1:k
        isMember=ismember(kMeansResult,cluster);
        % number of points belong to 7 and 9
        seven=sum(isMember(1:1000,1));
        nine=sum(isMember(1001:2000,1));
        if seven>=nine
            thisClass=7;
            kMeansResult(isMember)=1;
        else
            thisClass=9;
            kMeansResult(isMember)=-1;
        end
    end
    
    
    diff = kMeansResult - label;
    loss = diff'*diff/4/n;
    lossList(i)=loss;
end
lossList