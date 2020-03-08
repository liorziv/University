function [ai]=sil(X,idx)
clusterCentersIdx = unique(idx);
clusterGroupSize = size(clusterCentersIdx);
xLen = length(X);
cj = cell(clusterGroupSize(1),1);
si = zeros(xLen,1);

%finds the cluster groups

clusterCenters = X(clusterCentersIdx,:);
for i = 1:length(clusterCentersIdx)
    tmp = find(idx == clusterCentersIdx(i));
    cj{i} = X(tmp,:);
   
end
[ai, secondMinIdx] = findAvrageDistance(X, cj, clusterCentersIdx, clusterCenters, idx, 1);
[bi, ~] = findAvrageDistance(X, cj, clusterCentersIdx, clusterCenters, secondMinIdx,  0);
for j = 1: xLen
    si(j) = (ai(j) + bi(j))/ max(ai(j),bi(j));
end
disp('1');
end
function [ai, secondMinIdx] = findAvrageDistance(X, cj, clusterCentersIdx, clusterCenters, idx, flag)
xLen = length(X);
ai = zeros(xLen,1);
secondMinIdx = zeros(xLen,1);

for i = 1:xLen
    clusterGroup = cj(find(clusterCentersIdx==idx(i)));
    
    if(isempty(clusterGroup))
        return;
    end
    clusterGroup = clusterGroup{1};
    clusterGroupLen = length(clusterGroup);
    minDist = pdist2(X(i,:), clusterGroup);
    ai(i) = (sum(minDist))/clusterGroupLen;
    if(flag)
        distanceFromCenters = pdist2(X(i,:), clusterCenters);
        [~, deleteIdx] = min(distanceFromCenters);
        distanceFromCenters(deleteIdx) = inf;
        [~, secondMinIdx(i)] = min(distanceFromCenters);
        
    end
end
    
%     if(flag)
%     [~, deleteIdx] = min(distanceFromCenters);
%     distanceFromCenters(deleteIdx) = [];
%     [~, secondMinIdx] = min(distanceFromCenters);
%     if(~isempty(secondMinIdx(i)))
%         secondMinIdx(i,:) = clusterCenters(secondMinIdx,:);
%     else
%         secondMinIdx(i) = 0;
%     end
%     end
%     
    
end

