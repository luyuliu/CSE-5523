function [PCA, newTrain] = PCA_eig(train, k)
normalizedTrain = train - repelem(mean(train),size(train,1),1);
covTrain = normalizedTrain'*normalizedTrain/size(train,1);
[eigenvectors,eigenvaluesMatrix] = eig(covTrain);
[eigenvaluesOrdered,ind] = sort(diag(eigenvaluesMatrix),'descend');
eigenvectorsOrdered = eigenvectors(:,ind);
PCA = eigenvectorsOrdered(:, 1:k);
newTrain = train*PCA;
end