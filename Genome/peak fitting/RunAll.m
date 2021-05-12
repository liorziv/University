%%%%%%%%%%%%%%%%%% e %%%%%%%%%%%%%%%%%% 
%on d1
Y1 = load('d1.mat');
Y1 = Y1.Y;
[X1,H1,RMSE1]  = optimizeWrapper(10, Y1, 1);


%on d2
Y2 = load('d2.mat');
Y2 = Y2.Y;
[X2,H2,RMSE2]  = optimizeWrapper(10, Y2, 2);


%on d3
Y3 = load('d3.mat');
Y3 = Y3.Y;
[X3,H3,RMSE3]  = optimizeWrapper(10, Y3,2);

%on d4
Y4 = load('d4.mat');
Y4 = Y4.Y;
[X4,H4,RMSE4]  = optimizeWrapper(10, Y4, 3);

%%%%%%%%%%%%%%%%%% F %%%%%%%%%%%%%%%%%% 
% K vs RMSE
RMSEVec1 = zeros(10,1);
for k = 1:10
    [X5,H5,RMSEVec1(k)]  = optimizeWrapper(10, Y4, k);
end

figure;plot(1:k,RMSEVec1);
xlabel('K size - Number of Centers');
ylabel('RMSE');

% N Vs RMSE
RMSEVec2 = zeros(15,1);
for N = 1:15
    [X6,H6,RMSEVec2(N)]  = optimizeWrapper(N, Y4, 4);
end
figure;plot(1:N,RMSEVec2);
xlabel('Iterations Amount');
ylabel('RMSE');

%%%%%%%%%%%%%%%%%% G %%%%%%%%%%%%%%%%%%
% optimize X on d4
tic
[X7,H7,RMSE7]  = optimizeWrapper(1000, Y4, 3);
t1 = toc;
display('Running time :');
disp(t1);
display('RMSE :');
disp(RMSE7);
tic
[X8,H8,RMSEX]  = optimizeXWrapper(1000, Y4, 3);
t2 = toc;
display('RMSE :');
disp(RMSEX);





