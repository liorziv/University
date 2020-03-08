function [] = plotClusters(clust, X, type)
group1 = find(clust == 1);
group2 = find(clust == 2);
figure;scatter(X(group1,1),X(group1,2));
hold on;
scatter(X(group2,1),X(group2,2));
title(['Cluster - ' num2str(type)]);
end