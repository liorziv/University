%% Finds the best alignment of s1  and s2 
% fastaFile - a string containig the seqenece
% initialEmission - the emission matrix 
% initialTransition - the state transition matrix 
%%
function [backwardMat] = Backward(seq, initialEmission, initialTransition, letters, states)


%import both Emission and state transition matrix(and change them to log values)
initialEmissionMat = log(initialEmission);
initialTransitionMat = log(initialTransition);


%length of sequance and amouth of states
seqLen = length(seq{1});
numOfStates = size(initialTransitionMat);

%initalize the backward matrix
backwardMat = zeros(numOfStates(1), seqLen);
prevProbs = zeros(numOfStates(1),1);

%initalizion of the last colunm
backwardMat(:,seqLen) = -inf;
backwardMat(end,end) = 0;



for i = seqLen-1:-1:2
    for k = 1: numOfStates(1)
        if(i == seqLen -1)
           backwardMat(k,i) = calcLog(backwardMat(:,end) + initialTransitionMat(k,:)');
        else
            %gets the current sequence letter index
            nextletterIdx = find(letters == seq{1}(i+1));
        
            %takes the next colunm
            prevStep = backwardMat(:,i+1); 

            %calcs the probability to move from k to any of the states *
            %the probility of that state to emit the next letter * the next
            %column
            prevProbs(:) =  initialEmissionMat(:,nextletterIdx) + prevStep(:) + initialTransitionMat(k,:)';

            %calc in special log sum function
            backwardMat(k,i) = calcLog(prevProbs);



        end
    end
end
%fill the last column ( first at backward)
backwardMat(:,1) = -inf;
letterIdx = find(letters == seq{1}(end-1));
stateIdx = find(strcmp(states, 'Start'));
backwardMat(1,1) = calcLog(initialEmissionMat(:,letterIdx) + backwardMat(:,2)+ initialTransition(stateIdx,:)');


