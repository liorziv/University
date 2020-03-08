% narrow the data
gunzip('expMatrix.txt.gz');
A = importdata('expMatrix.txt');
B = A.data;
B(B <= 1) = 0;
B(B > 1) = 1;
expCells = A.data(:, sum(B) >= 2000);


[coeff, score,~,~,explained] = pca(expCells);
cnt = score(1:300,:);

cnt(cnt<=0) =1;
cnt(cnt ~=1)=0;
a = sum(cnt);
reducedMat = log(score(1:300,:));
reducedMat(imag(reducedMat)~=0) = 0;

kclusts = kmeans(reducedMat', 8);


figure;
[s, a] = silhouette(expCells',kclusts);
D = [reducedMat' kclusts];
[sortedMat, sortedIdx] = sortrows(D, 301);
figure;
subplot(1,4,1:3);
imagesc(sortedMat(:, 1:end-1));
subplot(1,4,4);
imagesc(sortedMat(:,end));
%suptitle('PCA + kmeans');

distMat = zeros(905,905);
for i = 1:905
    for j = 1:905
       distMat(i,j) = norm(expCells(:,i) - expCells(:,j));
    end
end

%things we tried 
ego = clustergram(reducedMat,'Standardize','row');

    
    

[C, L, U] = SpectralClustering(distMat, 8, 2);

D= full(C);
idx = zeros(905,1);
for i=1:8
    idx(find(D(:,i)==1))= i;
end

D = [reducedMat' idx];
[sortedMat, sortedIdx] = sortrows(D, 301);
figure;
subplot(1,4,1:3);
imagesc(sortedMat(:, 1:end-1));
subplot(1,4,4);
imagesc(sortedMat(:,end));

vbls = cell2mat(A.textdata(1:3));
biplot(coeff(:,1:3),'scores',score(:,1:3),'varlabels',vbls);

a = zeros(20,1);
%checking with silhouette
for i = 1:20
    kclusts = kmeans(reducedMat', i, 'MaxIter', 1000);
%     [C, L, U] = SpectralClustering(distMat, i, 2);
%     D= full(C);
%     idx = zeros(905,1);
%     for j=1:i
%         idx(find(D(:,j)==1))= j;
%     end

    tmp = silhouette(reducedMat,kclusts);
    a(i) = mean(tmp);
    disp(i);


end

figure;
bar(a);