
function [X,H,RMSE] = optimize(Y, k)
%% given the data (Y - a vector of length 50) and the expected number of
% peaks in the data (k) finds the vectors X;H that were most likely to
% generate the data. The function  return the root mean squared
% error (RMSE) of the result.
    theta0 = zeros(k,2);
    opts = optimset('TolX', 0.01,'MaxIter', 100,'Display',  'off');
    
    %rand for X 
  
    theta0(:, 1) = length(Y).*rand(k,1)+ 1;
    
    %rand for H
    
 
    theta0(:, 2) = max(Y).*rand(k,1);
    func = @(theta0) (norm(Y - predict(theta0(:, 1),theta0(:, 2))')); 
    tmp = fminsearch(func, theta0, opts);
    H = tmp(k+1:2*k);
    X = tmp(1:k);
    RMSE = func(tmp);
    
end