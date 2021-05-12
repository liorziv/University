%% Finds the best alignment of s1  and s2 
% fastaFile - a string containig the seqenece
% initialEmission - the emission matrix 
% initialTransition - the state transition matrix 
%%
function [forwardMat] = Forward(seq, initialEmission, initialTransition, letters, states)





%change Emission and state transition matrix to log values
initialEmissionMat = log(initialEmission);
initialTransitionMat = log(initialTransition);


%length of sequance and amouth of states
seqLen = length(seq{1});
numOfStates = size(initialTransitionMat);

%initalize the forward matrix
forwardMat = zeros(numOfStates(1), seqLen);
prevProbs = zeros(numOfStates(1),1);

%initalizion of the first colunm
forwardMat(:,1) = -inf;
forwardMat(1,1) = 0;


for i = 2:seqLen-1
    for k = 1:numOfStates(1)
        
        %gets the current sequence letter index
        currletterIdx = find(letters == seq{1}(i));
        
        %takes the previews colunm
        prevStep = forwardMat(:,i-1);
        
        %mult the previews column cell with the probility to move to the current
        %state (each cell seperatly)
        prevProbs(:) = prevStep(:) + initialTransitionMat(:,k:k);
  
        %takes the max value and the state it came from(insert to ptrMat)
        forwardMat(k,i) = initialEmissionMat(k,currletterIdx) + calcLog(prevProbs);
        
        
    end
end

%fill the last row of forward
forwardMat(:,end) = -inf;
forwardMat(end,end) = calcLog(forwardMat(:,end-1)+ initialTransitionMat(:,strcmp(states, 'End')));
