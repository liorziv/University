%% Finds the best alignment of s1  and s2 
% fastaFile - fasta file of sequence 1 (e.g. 'fastas/chrom17.fasta')
% initialEmission - the emission matrix in .tsv fileformat 
% initialTransition - the state transition matrix in .tsv fileformat
%%
function [ ] = viterbi(fastaFile, initialEmission, initialTransition)

%reads the fasta file
[~,seq] = fastaread(fastaFile);
letters = ['A','C','G','T']; %todo convert take it from the matrix

%import both Emission and state transition matrix(and change them to log values)
initialEmissionMat = log(ImportMatrix(initialEmission));
[initialTransitionMat,states] = ImportMatrix(initialTransition);
initialTransitionMat = log(initialTransitionMat);


%length of sequance and amouth of states
seqLen = length(seq);
numOfStates = size(initialTransitionMat);

%initalize the viterbi matrixm, ptr matrix to keep track of the
%best states path 
viterbiMat = zeros(numOfStates(1), seqLen);
ptrMat = zeros(numOfStates(1), seqLen);
prevProbs = zeros(numOfStates(1),1);

%initalizion of the first colunm

currletterIdx = find(letters == seq(1));
viterbiMat(:,1) = initialTransitionMat(1:1,: ) + initialEmissionMat(:,currletterIdx:currletterIdx)';


%creates the viterbi matrix
for i = 2:seqLen
    for k = 1:numOfStates(1)
        
        %gets the current sequence letter index
        currletterIdx = find(letters == seq(i));
        
        %takes the previews colunm
        prevStep = viterbiMat(:,i-1:i-1);
        
        %mult the previews column cell with the probility to move to the current
        %state (each cell seperatly)
        prevProbs(:) = prevStep(:) + initialTransitionMat(:,k:k);
  
        %takes the max value and the state it came from(insert to ptrMat)
        [mostLikelyPrevState, row] = max(prevProbs);
        ptrMatrix(k,i) = row;
        %stores the sum of all
        viterbiMat(k,i) = initialEmissionMat(k,currletterIdx) + mostLikelyPrevState;
       
        
    end
end
%takes the max of the last column
[~,maxIdx] = max(viterbiMat(:,seqLen:seqLen));
%convert the idx to state
statesStr = strjoin(states(maxIdx));
solution =  blanks(seqLen) ;
prevStateIdx = ptrMatrix(ptrMatrix(maxIdx, seqLen),seqLen -1);
solution(end) =  statesStr(1);
len = seqLen -1;

%retrives the best path of states
while(len >= 2)
    len = len - 1;
    prevState = strjoin(states(prevStateIdx));
    solution(len)=  prevState(1);
    prevStateIdx = ptrMatrix(prevStateIdx, len);
    
end

[gcRatioSeq,gcRationRes] = calcCGRatio(seq,solution);
%prints the seq and the most probable states
printSeqAndStates(seq,solution,fastaFile);    


%this function prints the results
function [ ] = printSeqAndStates(seq,solution,fileName)
display('The most proable path is:')
if(length(seq) < 50)
    disp(solution(1:end));
    disp(seq);
else
    disp(solution(1:50));
    disp(seq(1:50));
        
end
    

        
function [gcRatioSeq,gcRationRes] = calcCGRatio(seq,result)
gcRatioSeq = zeros(round(length(seq)/100),1);
gcRationRes = zeros(round(length(seq)/100),1);
gcRationRes = zeros(round(length(seq)/100),1);

j = 0;
for i = 1:100:length(seq)-100
    j = j + 1;
    
    g = sum(seq(i:i+100)=='G');
    c = sum(seq(i:i+100)=='C');
    gcRatioSeq(j) = (g+c)/100;
    
    i = sum(seq(i:i+100)=='C');
    gcRationRes(j) = i/100;
    
end


