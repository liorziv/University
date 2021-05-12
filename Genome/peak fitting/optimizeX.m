
function [X,H,RMSE] = optimizeX(Y, k)
%% given the data (Y - a vector of length 50) and the expected number of
% peaks in the data (k) finnds the vector X that were most likely to
% generate the data and calcalues H by it. The function  return the root mean squared
% error (RMSE) of the result.
    
    len = length(Y);
    opts = optimset('TolX', 0.01,'MaxIter', 100,'Display',  'off');
    
    %rand for X 
  
    theta0 = len.*rand(k,1)+ 1;
    
    %rand for H
 
    H = max(Y).*ones(k,1);
    func = @(theta0) (norm(Y - predict(theta0,H)')); 
    X = fminsearch(func, theta0, opts);
    centersMat = zeros(len,k);
    for i = 1:k
        centersMat(:,i) = F(X(i));
    end
    B = pinv(centersMat)*Y;
    H = B.*(max(centersMat)');
    RMSE = norm(Y - predict(X,H)');
    X = X';
    
    
end

