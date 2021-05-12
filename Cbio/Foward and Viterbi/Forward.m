%% Finds the best alignment of s1  and s2 
% fastaFile - fasta file of sequence 1 (e.g. 'fastas/chrom17.fasta')
% initialEmission - the emission matrix in .tsv fileformat 
% initialTransition - the state transition matrix in .tsv fileformat
%%
function [ ] = Forward(fastaFile, initialEmission, initialTransition)

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
forwardMat = zeros(numOfStates(1), seqLen);
prevProbs = zeros(numOfStates(1),1);

%initalizion of the first colunm

currletterIdx = find(letters == seq(1));
forwardMat(:,1) = initialTransitionMat(1:1,: ) + initialEmissionMat(:,currletterIdx:currletterIdx)';


%creates the viterbi matrix
for i = 2:seqLen
    for k = 1:numOfStates(1)
        
        %gets the current sequence letter index
        currletterIdx = find(letters == seq(i));
        
        %takes the previews colunm
        prevStep = forwardMat(:,i-1:i-1);
        
        %mult the previews column cell with the probility to move to the current
        %state (each cell seperatly)
        prevProbs(:) = prevStep(:) + initialTransitionMat(:,k:k);
  
        %takes the max value and the state it came from(insert to ptrMat)
        maxXi = max(prevProbs);
        if(maxXi == -inf)
            forwardMat(k,i) = -inf;
        else
            expByXi = exp(prevProbs - maxXi);
            specialSum = maxXi + log(sum(expByXi));
            forwardMat(k,i) = initialEmissionMat(k,currletterIdx) + specialSum;
        end
        
    end
end
%takes the max of the last column
lastCol = forwardMat(:,seqLen:seqLen);
[~,maxIdx] = max(lastCol);
%convert to log likelihood
expByXi = exp(lastCol - maxXi);
logLikelihood = maxXi + log(sum(expByXi));
display('The result is :');
display(logLikelihood);