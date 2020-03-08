function [X,H,RMSE] = optimizeWrapper(N, Y, k)

%%  wrapper function that runs the optimization N times, records
% the RMSE value of each run and chooses the best set of parameters.
RMSE = inf;

for i = 1:N
    [iterX,iterH,iterRMSE] = optimize(Y, k);
    if(iterRMSE < RMSE)
        X = iterX;
        H = iterH;
        RMSE = iterRMSE;
    end 
end
yhat = predict(X,H);
figure;plot(Y);
hold on;
plot(yhat);
legend('Y', 'yhat')
xlabel('Location in Genomic Region');
ylabel('Strength of the Signal');
  
end