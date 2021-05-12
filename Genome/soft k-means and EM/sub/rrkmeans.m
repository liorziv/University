%%executes kmeans clustering restartNum times and returns the result which
%had the best objective function value (bestScore).
function [idx,bestScore] = rrkmeans(X,k,restartNum)
[idx,~,score] = kmeans(X,k, 'Start', 'uniform', 'MaxIter',100, 'Replicates', restartNum);
bestScore = sum(score);  
end

