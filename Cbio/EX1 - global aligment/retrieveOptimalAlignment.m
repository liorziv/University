%% reconstructs the optimal alignment sequences
%input:
% s1 - fasta file of sequence 1 (e.g. 'fastas/HomoSapiens-SHH.fasta')
% s2 - fasta file of sequence 2
% alignOrigMatrix - holds the paths for all possible alignments 
% alignScoreMatrix - holds the scores for all the possible alignments
% type - type of alignment (e.g. 'local')
% bestScoreCor - the coordinates of the best score location in origin
% matrix
%output:
%s1 - the first strand alignment
%s2 - the second strand alignment
%%

function [seq1,seq2] = retrieveOptimalAlignment(s1, s2, alignScoreMatrix, alignOrigMatrix, type, bestScoreCor)

%for the origin matrix
s1Orig = 1;
s2Orig = 2;


%%retriving the alignment
seq1 = '';
seq2 = '';
i = bestScoreCor(1);
j = bestScoreCor(2);


%reconstructs the best score sequence by the help of the origin matrix
while((i > 1 && j > 1 & strcmp(type, 'global')) | alignScoreMatrix(i:i, j:j) ~= 0 & strcmp(type, 'local'))
    
    %the x,y coordinates of the tile we came from
    s1_idx = alignOrigMatrix(i - 1, j - 1, s1Orig);
    s2_idx = alignOrigMatrix(i - 1, j - 1, s2Orig);
    
    %diagonal tile
    if(i - 1 == s1_idx && j - 1 == s2_idx)
        seq1 = strcat(s1(s1_idx), seq1);
        seq2 = strcat(s2(s2_idx), seq2);
       
    %left tile
    else if(i - 1 == s1_idx && j == s2_idx)
        seq2 = strcat('-', seq2);
        seq1 = strcat(s1(s1_idx), seq1);
    
    %upper tile
    else
        seq2 = strcat(s2(s2_idx), seq2);
        seq1 = strcat('-', seq1);
     
        end
    end
    
    i = s1_idx;
    j = s2_idx;
end

%if there are gaps at the end of one sequence
if(strcmp(type, 'global'))
     if(i >= 1)
        seq1 = strcat((s1(1:i)), seq1);
        seq2 = strcat(repmat('-', 1,i - 1), seq2);
     end

    if(j >= 1)
        seq2 = strcat((s2(1:j)), seq2);
        seq1 = strcat(repmat('-', 1, j - 1), seq1);
    end

end


    


