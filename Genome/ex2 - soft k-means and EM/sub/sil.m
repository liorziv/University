function [si,ai,bi]=sil(X,idx)
clusterNumCenters= unique(idx);
clusterGroupSize = size(clusterNumCenters);
xLen = length(X);
cj = cell(clusterGroupSize(1),1);
si = zeros(xLen,1);
clusterCenters = zeros(clusterGroupSize(1),2);

%finds the cluster groups and calculates the centroids
for i = 1:clusterGroupSize(1)
    tmp = find(idx == i);
    cj{i} = X(tmp,:);
    clusterCenters(i,:) = sum(cj{i})/length(cj{1});
   
end

[ai, bi] = findAvrageDistance(X, cj, clusterCenters, idx);

%calculets S with ai,bi results
for j = 1: xLen
    si(j) = (bi(j) - ai(j))/ max(ai(j),bi(j));
end
end

%calculates ai and bi
function [ai, bi] = findAvrageDistance(X, cj, clusterCenters,idx)
xLen = length(X);
ai = zeros(xLen,1);
bi = zeros(xLen,1);
copyOfClusterCenters = clusterCenters;


for i = 1:xLen
    
    clustGroup1 = cj(idx(i));
    ai(i) = sum(pdist2(X(i,:), clustGroup1{1}))/length(clustGroup1{1});
    clusterCenters(idx(i),:) = [inf, inf];
    [~, secondMinIdx] = min(pdist2(X(i,:), clusterCenters));
    clustGroup2 = cj(secondMinIdx);
    bi(i) = sum(pdist2(X(i,:), clustGroup2{1}))/length(clustGroup2{1});
    clusterCenters = copyOfClusterCenters;
 
end    
end

