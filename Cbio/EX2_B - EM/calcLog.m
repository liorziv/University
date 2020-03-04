function [logSum] = calcLog(vec) 
    maxXi = max(vec);
    if(maxXi == -inf)
        logSum = -inf;
    else
        expByXi = exp(vec - maxXi);
        logSum = maxXi + log(sum(expByXi));
        
    end