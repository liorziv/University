function [] = ex3()

letterDict = ['A','C','G','T'];

% O1
sampleLetterFewTimes('A',[10,100,1000],[0.15,0.4,1.1],letterDict, 0.25);

% Q2
createBoxPlot(100,500,[0.15,0.4,1.1],letterDict, 0.25);

% O3
tList(1000, 100, [[0.1,0.1];[0.5,0.1];[0.1,0.5];[0.5,0.5]],letterDict, 0.25);

end

%implemention of the func P(a->b)
function [retValue] = PJC(a, b, t, alpha)
retValue = 0.25*(1-exp(-4*alpha*t));
if(a==b)
    retValue = 0.25*(1+3*exp(-4*alpha*t));
end
end

%calculates the probilities vector for rach letter
function [prob] = calcProbVec(a,t, lettersDict,alpha)
letterListSize = length(lettersDict);
prob = zeros(letterListSize,1);

for i = 1:letterListSize
    prob(i) = PJC(a,lettersDict(i),t,alpha);
end
end

%creates a sequnce with length of sampelsNum, by the probilities vector
function [chosenBase] =  sampleLetter(a,t,N,letterDict, alpha)
prob = calcProbVec(a,t,letterDict,alpha);
chosenBase = randsample(letterDict,N,1,prob);
end

%calculets the precentage of each letter from the sequence
function [precVec] = calcPrcentage(calcSeq, lettersList)
letSize = length(lettersList);
precVec = zeros(letSize,1);
for i = 1:letSize
    precVec(i) = sum(length(find(calcSeq == lettersList(i))))/sum(length(calcSeq));
end
end

%generates smapleLetter for a few values of t and sample number 
function [] = sampleLetterFewTimes(a, N, tVec, lettersDict,alpha)
tLen = length(tVec);
sLen = length(N);
for i = 1:sLen
    for j = 1:tLen
        samplesLet = sampleLetter(a,tVec(j),N(i),lettersDict,alpha);
        precVec = calcPrcentage(samplesLet, lettersDict);
        figure;
        bar(precVec);
        set(gca,'XTickLabel', num2cell(lettersDict)); 
        ylabel('Frequency') % y-axis label
        title(['Base ',a, ' with t = ',num2str(tVec(j)), ' and ' , num2str(N(i)),'  Samples'])
    end
end
end

%calculets the MLE of two sequences
function [mleValue] = calcMLE(seq1,seq2,letterDict)
    seqLen = length(seq1);
    seq1 = convertToNumVec(seq1,letterDict);
    seq2 =  convertToNumVec(seq2,letterDict);
    C = length(find((seq1-seq2) == 0));
    mleValue = log(double(3*seqLen)/(4*C-seqLen));


end
    
%convert a seqence string vector into a int vector
function [seqVec] =  convertToNumVec(seq,lettersList)
seqLen = length(seq);
seqVec = zeros(seqLen,1);
for i = 1 : seqLen
    seqVec(i) = find(strcmp(num2cell(lettersList),seq(i)));
end
end

%generates two seqences, the first is random the second is sampled by the
%first
function [seq1,seq2] = createPair(N, t, lettersDict, alpha)

seq1 = randseq(1);
seq2 = sampleLetter(seq1,t,1,lettersDict, alpha);
for i = 1:N - 1
    seq1 = strcat(seq1, randseq(1));
    seq2 = strcat(seq2, sampleLetter(seq1(end),t,1,lettersDict, alpha));
end
end

%creates a box plot from diffrenet pairs of seqences - Q1 code
function createBoxPlot(M, N, tVec, lettersDict,alpha)
tLen = length(tVec);
resVec = zeros(M,tLen);
tmpVec = zeros(M,1);
  
for i = 1:tLen
    for j = 1:M
        [seq1, seq2] = createPair(N,tVec(i),lettersDict, alpha);
        tmpVec(j) = calcMLE(seq1,seq2,lettersDict); 
    end
    resVec(:,i) = tmpVec;
    
end
figure
boxplot(resVec);
set(gca,'XTickLabel', tVec); 
ylabel('Precentage of distrubution of the estimated distances (MLE)') % y-axis label
xlabel('T values - distance');
title('Box plot with 100 sequences of length 500, and a given distance(t)');


end

%creates tree out of a sample size with two fathers and alpha,beta
%distances
function [tree] = createTree(alphaT1, betaT2, sizeOfSeq, lettersDict, alpha)
tree = cell(6);

father1 =  randseq(sizeOfSeq);
child1_2 = '';
child1_1 = '';
father2 = '';
for i = 1:sizeOfSeq
    child1_2 = strcat(child1_2, sampleLetter(father1(i), alphaT1, 1, lettersDict, alpha));
    child1_1 = strcat(child1_1, sampleLetter(father1(i), betaT2, 1, lettersDict, alpha));
    father2 = strcat(father2, sampleLetter(father1(i), alphaT1, 1, lettersDict, alpha));
end

child2_3 = '';
child2_4 = '';

for i = 1:sizeOfSeq

    child2_3 = strcat(child2_3, sampleLetter(father2(i), betaT2, 1, lettersDict, alpha));
    child2_4 = strcat(child2_4, sampleLetter(father2(i), alphaT1, 1, lettersDict, alpha));
end

tree{1} = father1;
tree{2} = child1_2;
tree{3} = child1_1;
tree{4} = child2_4;
tree{5} = child2_3;
tree{6} = father2;
end

function [] = tList(N, M, tVec,lettersDict, alpha)
tLen = length(tVec);
iterList = zeros(tLen, 1);
iterCtor = 0;
QList = zeros(6,1);
for i = 1:tLen
    for j = 1:M
        alphaT1 = tVec(i ,1);
        betaT2 = tVec(i,2);
        tree = createTree(alphaT1, betaT2, N, lettersDict, alpha);
        
        d_1_2 = calcMLE(tree{2}, tree{3}, lettersDict);   
        d_3_1 = calcMLE(tree{4}, tree{2}, lettersDict); 
        d_1_4 = calcMLE(tree{2}, tree{5}, lettersDict);
        d_2_3 = calcMLE(tree{3}, tree{4}, lettersDict); 
        d_2_4 = calcMLE(tree{3}, tree{5}, lettersDict); 
        d_3_4 = calcMLE(tree{4}, tree{5}, lettersDict);      
        
        r1 = (d_1_2 + d_3_1 +  d_1_4)/2;
        r2 = (d_1_2 + d_2_3 + d_2_4)/2;
        r3 = (d_3_1 + d_2_3 + d_3_4)/2;
        r4 = (d_1_4 + d_2_4 + d_3_4)/2;
        
        
        QList(1) = r1 + r2 + - d_1_2;
        QList(2) = r1 + r3 + - d_3_1;
        QList(3) = r1 + r4 + - d_1_4;
        QList(4) = r2 + r3 + - d_2_3;
        QList(5) = r2 + r4 + - d_2_4;
        QList(6) = r3 + r4 + - d_3_4;
        
        [~, maxIdx] = max(QList);
        
        if(maxIdx == 6 | maxIdx == 1)
            iterCtor = iterCtor + 1;
            
        end
            
     end
        
        
        iterList(i) = iterCtor;
        iterCtor = 0;
        
        
        
end

figure;
bar(iterList);
set(gca,'XTickLabel', tVec); 
ylabel('Precantage of correct reconstructions') % y-axis label
xlabel('(alpha, beta) values') % x-axis label
title('Precantage of correct reconstructions per different alpha and beta values')
    
end


