% ps = F(52);
% figure;
% plot(ps);
% 
% X = [10, 20, 50];
% H = [12,nan,20];
% 
% yhat = predict2(X,H);
% figure;
% plot(yhat);
% 
% optimize(yhat, 3);

[X,H,RMSE] = optimize(yhat, 1);
ps = predict2(X,H);
figure;plot(yhat);
hold on;
plot(ps);




