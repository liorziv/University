%%chooses (uniformly) per% observations from X, executes kmeans on them and
%then assigns each of the remaining observations to the nearest cluster
function [idx,score, time] = sskmeans(X,k,per)
    %sampled data
    sampledDataIdx= randsample(length(X),(length(X)*per)/100);
    sampledData = X(sampledDataIdx,:);
    score = 0;
    idx = zeros(length(X),1);
    %In order to mesure the time
    tic; 
    [startIdx,centroids] = kmeans(sampledData,k, 'Start' ,'sample','MaxIter', 300, 'Replicates' , 20);
    time = toc;
   
    for i = 1:length(X)
       %finds the euclidan distance 
       eucDist = pdist2(X(i,:), centroids);
       [minVal,minIdx] = min(eucDist);
       score  = score + minVal;
       idx(i) = minIdx;
    end
end
