%% creates both the emission and transition matrices accrding to 
%the given values
function [emissionMat, transitionMat, states]  = INIT_EM(motif, k, p, q, L1, alpha)

%initialization for all states and letters
allStates = 9 + k;
letters = ['A','C','G','T']; 
states = cell(9+k,1);

%The states order
states = {'Start', 'B1', 'B2', 'B3'}; 

mCtor = 1;
for i = 5:k+4
    states{i} = strcat('Motif' , num2str(mCtor));
    mCtor = mCtor +1;
end

padCtor = 1;
for i = k+5:k+9
    states{i} = strcat('Padding' , num2str(padCtor));
    padCtor = padCtor +1;
end 
states{end} = 'End';

%creates transition and emission matrices
transitionMat = zeros(allStates , allStates);
emissionMat = zeros(allStates,4);

%%init of the emisson matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%init for b1,b2,b3
emissionMat(idx(states,'B1'):idx(states,'B3'),1:4) = repmat(repmat(0.25,1,4),3,1);

%init for the motif states
for i = 1: k
    currLetterIdx = idx(letters,motif(i));
    emissionMat(i+4,1:4) = alpha;
    emissionMat(i+4,currLetterIdx) = 1 - 3*alpha;
   
end

%init the 4 padding states
emissionMat(5+k: 8+k, 1:4) = 0.25;

%emissionMat(find(letters == 'S'),1) = 1;

%init of the end state 
emissionMat(end, find(letters == '$')) = 1;


%%init of the transition matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%put zero in all start states besides those which go
%to a b1,b2 
transitionMat(:,idx(states,'Start')) = 0;
transitionMat(idx(states,'Start'),:) = 0;
%for b1
transitionMat(idx(states,'Start'),idx(states,'B1')) = 1-q;
%for b2
transitionMat(idx(states,'Start'),idx(states,'B2')) = q;

%b1->b1 (P) b2->b2 , b3->b3
for i = 2:4
    transitionMat(i,i) = p;
end 

%b2-> -2
transitionMat(3,5+k) = 1 - p;


%putting ones for all padding and motif transitions
for i = 1:k-1
    transitionMat(4+i, 5+i) = 1;
end

transitionMat(10,13) = 1;
transitionMat(13,14) = 1;
transitionMat(11,12) = 1;
%padding4 ->b3
transitionMat(8+k, 4) = 1;

%-1 -> first motif letter
transitionMat(6+k, 5) = 1;

%b3 -> end
transitionMat(4,end) = 1 - p;
%b1 -> end
transitionMat(2,end) = 1 - p;

function [i] = idx(con, var)
if(strcmp(class(con),'cell'))
    i = find(strcmp(con,var));
else
    i = find(con == var);
end
    
