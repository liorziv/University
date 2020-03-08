function [X,H,RMSE] = optimizeXWrapper(N, Y, k)

%%  wrapper function that runs the optimization for X N times, records
% the RMSE value of each run and chooses the best set of parameters.
RMSE = inf;

for i = 1:N
    [iterX,iterH,iterRMSE] = optimizeX(Y, k);
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