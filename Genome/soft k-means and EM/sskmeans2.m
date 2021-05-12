function [idx,score, time] = sskmeans(X,k,per)
   
    sampledData = datasample(X,length(X)*per/100);
    totaldistScore = 0;
    idx = zeros(length(X),1);
    tic;
    [startIdx] = kmeans(sampledData,k);
    time = toc;
    centerIdx = unique(startIdx);
    centromes = sampledData(centerIdx,:);
    minArr = zeros(k,1);
   
    for i = 1:length(X)
        
        for j = 1:length(centerIdx)
            
            diff = X(i,:)- centromes(j,:);
            minArr(j) = sqrt(diff*diff');
        end
        [minVal,minIdx] = min(minArr);
        score  = totaldistScore + minVal;
        idx(i) = find(X == centromes(minIdx));
    end
    
end
