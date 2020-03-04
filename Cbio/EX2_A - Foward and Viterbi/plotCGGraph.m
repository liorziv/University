function [] = plotCGGraph(seq,solution)

%finds all the 'C' chars
numOfC = find(solution == 'C') ;
%will be used for plotting the graph
forPlot = zeros(length(solution),1);
counter = 0;
for i = 1: length(numOfC) -1
    if(numOfC(i+1) - numOfC(i) == 1)
        forPlot(numOfC(i)) = 1; 
    end
end

%creates a vector of ones only in places there are above 200 'C' in a row
start = 1;
for i = 1 : length(forPlot)
    if(forPlot(i) == 0 && counter < 200)
        counter = 0;
        forPlot(start:i) = 0;
    end
    counter = counter +1;
end

%calcs the GC ration in 100bs tiels
gcRatio = zeros(round(length(seq)/100),1);
j = 0;
for k = 1:100:length(seq)-100
    j = j + 1;
    
    a = sum(seq(k:k+100)=='G');
    e = sum(seq(k:k+100)=='C');
    gcRatio(j) = (a+e)/100;
end


%plots what i submitted
figure
x = 1:100:length(seq);

flag = 0;

for idx = 1 : length(forPlot)
    if(forPlot(idx) == 1)
       
       x2 = plot([idx idx], [0 1],'Color',[0.4 0.4 0.4]);h.Color(4)=0.3;
       if(flag == 0)
           hold on
           flag = 1;
       end
       

    end
  
end


x1 = plot (x, gcRatio, 'r');


legend([x1 x2],'CG - Ratio','CpG - Viterbi')

xlabel('Location in the sequence') % x-axis label
ylabel('CpG ratio') % y-axis label


