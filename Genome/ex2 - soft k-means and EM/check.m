rng(1)
% Q2 B
perVec = [1:10 20:10:100];
X = load('dkmeans2.mat');
time = zeros(length(perVec),1);
scores = zeros(length(perVec),1);
 X = struct2cell(X);
 X = X{1};
i = 1;
for per = perVec
    disp(i);
     [idx1,scores(i), time(i)] = sskmeans(X,3,per);
    i = i + 1;
end
[idx,a] = rrkmeans(X,3, 1);
figure; plot(perVec, time);
xlabel('precentage from data');
ylabel('Time it took');
title('Precentage Vs Time');

figure; plot(perVec, scores);
xlabel('precentage from data');
ylabel('Best score');
title('Precentage Vs Best score');

