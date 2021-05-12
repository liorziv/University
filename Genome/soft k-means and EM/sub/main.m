
% Q1 
X = load('data2d.mat');
X = struct2cell(X);
X = X{1};

%b.1
%Average linkage
T1 = aggClust(X, 1);

figure; dendrogram(T1,0);
title('dendrogram using Average linkage');

%Single linkage
T2 = aggClust(X, 2);
figure;dendrogram(T2,0);
title('dendrogram using Single linkage');

%b.2-3
clust1 = cluster(T1,'maxclust',2);
plotClusters(clust1, X, 'Avrage linkage');

clust2 = cluster(T2,'maxclust',2);
plotClusters(clust2, X, 'Single linkage');

% Q2 A
restartVec = [1:10 20:10:100];
scores = zeros(size(restartVec));
X = load('dkmeans1.mat');
X = struct2cell(X);
X = X{1};
i = 1;
for restartNum =  restartVec
    [idx,scores(i)] = rrkmeans(X,10, restartNum);
    i = i + 1;
end

figure; plot(restartVec,scores);
xlabel('Restart Number');
ylabel('Best score');
title('restartNum Vs bestScore');


% Q2 B
perVec = [1:10 20:10:100];
X = load('dkmeans2.mat');
time = zeros(length(perVec),1);
scores = zeros(length(perVec),1);
 X = struct2cell(X);
 X = X{1};
i = 1;
for per = perVec
     [idx1,scores(i), time(i)] = sskmeans(X,3,per);
    i = i + 1;
end
figure; plot(perVec, time);
xlabel('percentage from data');
ylabel('Time it took');
title('percentage Vs Time');

figure; plot(perVec, scores);
xlabel('percentage from data');
ylabel('Best score');
title('percentage Vs Best score');


%Q3
X = load('dkmeans3.mat');
xLen = length(X.dkmeans3);
X = struct2cell(X);
X = X{1};

s = zeros(xLen,10);
a = zeros(xLen,10);
b = zeros(xLen,10);
for k = 2:10
    [idx, ~] = rrkmeans(X,k, 10);
    [s(:,k),a(:,k),b(:,k)] = sil(X,idx);
end

figure;histogram(s(2:end,4),9);
title('Histogram for K = 4');

figure;histogram(s(2:end,8),9);
title('Histogram for K = 8');

figure;bar(2:10, mean(s(:,2:end)));
title('Silhouette means per k');

