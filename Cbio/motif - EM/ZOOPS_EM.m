%% Finds the most occuring L1 motifes, and create the most probable
%emission and transition matrices for each motif.
% trainSeq - fasta file contiaing sequences 
%k - the size of the motif
%convergenceThr - the thershold for stopping
%p,q,alpha - probilities for the emission and transition matrices
%L1 - the num of most occring motifes we want to learn.

function [] = ZOOPS_EM(trainSeq, k , convergenceThr, p, q, L1, alpha)

motifOutputFile = 'motifes.txt';
motifK = k;
%reads the fasta file and get all seqences
%add all a start and end char to each seqence
[headers, seqences] = fastaread(trainSeq);

%add two fack letters to each sequence
numOfSequences = length(seqences);
for i = 1:numOfSequences
    seqences(i) = strcat('S', seqences(i));
    seqences(i) = strcat(seqences(i), '$');
end


letters = ['A','C','G','T'];

%gets the motifes and tne number of their occurences
perl('countWords.pl',trainSeq, motifOutputFile, num2str(motifK));


%parse the motfis file into cell
fid1 = fopen(motifOutputFile);
motifs = textscan(fid1, '%s %d');
fclose(fid1);

%sort the motifes
[~,sortedMotifsIdx] = sort(motifs{2},'descend');
%will hold the logL at the end in order to print
LLToPrint = cell(L1,1);
motifsInSequences = zeros(L1, numOfSequences);


  

for i = 1:L1
    
    %creates emssion and transition marices + q
     motiftmp = motifs{1}(sortedMotifsIdx(i));
     motifOccurences = motifs{2}(sortedMotifsIdx(i));
     motif = motiftmp{1};
     motifEmission = cell(L1);
     
     %spacedMotif = strcat(motif, '    ');
     LLToPrint{i} = motif;
    
     q = motifOccurences / numOfSequences;
     [emissionMat, transitionMat, states] = INIT_EM(motif, motifK, p, q, L1, alpha);
     %init the LL variable - will hold the LL for each iteration
     logL = cell(1,1);
     logL{i}(1) = 0;
     logL{i}(2) = inf;
     LLCtor = 1;
   
     %Baum Welch
     while(abs(logL{1}(end) - logL{1}(end-1)) > convergenceThr)
         
         
         %keeps the forward and backward loglikelhood results
         forwardMatrices = cell(numOfSequences,1);
         backwardMatrices = cell(numOfSequences,1);
         tmpSum = 0;
         
         for j = 1:numOfSequences 
             tmpFind = strfind([seqences{j}],motif);
             if(tmpFind)
                 motifsInSequences(i,j) = tmpFind(1);
             end
             
             forwardMatrices{j,1} = Forward(seqences(j), emissionMat, transitionMat, letters, states);
             backwardMatrices{j,1} = Backward(seqences(j), emissionMat, transitionMat, letters, states);
             tmpSum = tmpSum + forwardMatrices{j,1}(end,end);
             
         end
         if(length(logL{1}) == 2)
             logL{LLCtor}(1) = 0;
             logL{LLCtor}(2) = -inf;
             logL{LLCtor}(3) = tmpSum;
         else
             logL{LLCtor}(end+1) = tmpSum;
         end
         
         
         %calcultes NKX and NKL in order to update the emission and
         %transition matrices
         stateAmount = find(strcmp(states,'Padding4')) - find(strcmp(states,'Motif1'))+1;
         NKX = cell(stateAmount, length(letters));
         nklTmp = cell(1,1);
         for k = 1:numOfSequences
             
           
             F = forwardMatrices{k, 1};
             B = backwardMatrices{k, 1};
             currSeq = seqences{k};
             currSeqLen = length(currSeq);
             NKXPerCellSum = zeros(size(NKX));
         
             for l = 2:currSeqLen-1
                 nklTmp{1}(end+1) = F(find(strcmp(states,'Start')),l-1) + log(transitionMat(find(strcmp(states,'Start')),find(strcmp(states,'B2')))) + log(emissionMat(find(strcmp(states,'B2')), find(letters == currSeq(l)))) +B(find(strcmp(states,'B2')),l) - F(end,end); 

                 
                 for stateIdx = 5:stateAmount +4
                     for baseLetter = 1: length(letters)
                         currletterIdx =  find(letters == currSeq(l));
                         if(currletterIdx ==  baseLetter);
                             NKX{stateIdx-4, currletterIdx}(end+1) = F(stateIdx,l) + B(stateIdx,l) - F(end,end);
                         end
                     
                     end
                     
                end
             end
             
             
             
            
             
         end
         
         %calcs the new values accoring to NKL and NKX
         NKXStateTot = zeros(stateAmount,1);
        
         for stateIdx = 1:stateAmount
             S = [NKX{stateIdx,:}];
             NKXStateTot(stateIdx) = calcLog(S);
             for letter = 1: length(letters)
                 NKXPerCellSum(stateIdx,letter) = calcLog([NKX{stateIdx,letter}]);
                 emissionMat(stateIdx + 4,letter) = exp(NKXPerCellSum(stateIdx,letter) - NKXStateTot(stateIdx));
             end
                 
         end
         
         %if q > 1 than it settes it as 1
         newQ = exp(calcLog(nklTmp{1})- log(numOfSequences));
         if(newQ > 1)
             transitionMat(find(strcmp(states,'Start')),find(strcmp(states,'B2'))) = 1;
         else
             transitionMat(find(strcmp(states,'Start')),find(strcmp(states,'B2'))) = newQ;
         end
        
         transitionMat(find(strcmp(states,'Start')),find(strcmp(states,'B1'))) = 1 -  transitionMat(find(strcmp(states,'Start')),find(strcmp(states,'B2')));
             
        
     end
     
    motifEmission{LLCtor} =   round(emissionMat(5:end-1, :),2);
    LLCtor = 1+ LLCtor;
    
end

writeHistory(logL,LLToPrint, L1);
writeMotif(motifEmission,motifsInSequences, L1,headers);

% %plots the mean of LL for each iteration 
% plot(logL{1}(3:end)./56)
% xlabel('LL Value Per Iteration') % x-axis label
% ylabel('Mean of LL Per Iteration') % y-axis label

end

%write the motif1.txt file
function [] = writeMotif(motifEmission,motifsInSequences, L1,headers)

fileIDtmp = fopen('motif1.txt', 'wt');
fileID = fopen('motif1.txt', 'a');

numOfSequnces = length(headers);
for i = 1:L1
    dlmwrite('motif1.txt',motifEmission{i}' ,'-append', 'delimiter', ' ', 'precision',2);
    for j = 1:numOfSequnces
        
        fprintf(fileID, '%s\t',headers{j});
        fprintf(fileID, '%s\n',int2str(motifsInSequences(i,j)) );
    end
    
end
fclose(fileID);
end

% writes the log likelihood history log
function [] = writeHistory(logL, motifNames, L1)

maxLen = length(logL{1});
if(L1 ~= 1)
    maxLen = max(cellfun(@(field) length(field),logL));

end

fileID = fopen('History.tab', 'wt');
printFormat = strcat(repmat('%s\t\t',1,L1), ' ');
fprintf(fileID,printFormat,motifNames{:});
fprintf(fileID,'\n');

for j = 3: maxLen
    for i = 1:L1
        if(length(logL{i}) > j)
            if(i == L1)
                fprintf(fileID, '%.4f\t\n', logL{i}(j));
            else
                fprintf(fileID, '%.4f\t', logL{i}(j));
            end
            
        else
             fprintf(fileID,'\n');
        end
        
        
    end
end
end