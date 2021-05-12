%% crates an alignment matrix and origin matrix 
%in order to find the best score alignment
%input:
% s1 - fasta file of sequence 1 (e.g. 'fastas/HomoSapiens-SHH.fasta')
% s2 - fasta file of sequence 2
% type - type of alignment (e.g. 'local')
% scoreMatrix - the score matrix in .tsv fileformat (e.g. 'score_matrix.tsv')
%output:
%alignScoreMatrix - matrix with scores of all possible alignments
%alignOrigMatrix - matrix which hold the path to each tile
%bestScore - the best score (relevant to local alignment
%bestScoreCor - the x,y coordinates of the best score (for local)
%%
function [alignScoreMatrix,alignOrigMatrix, bestScore, bestScoreCor ] = alignSequences( s1, s2, scoreMatrix, type )


%holds all the possible moves to a tile
locsKeySet = [[1,1]; [0,1];[1,0]; [0,0]];

%holds all the letters of a sequence
letterKeySet =   ['A', 'C', 'G', 'T','gap'];

%for the origin matrix
s1Orig = 1;
s2Orig = 2;

%will hold the best score and i'ts location in the origin matrix (only for
%the local case)
bestScore = 0;
bestScoreCor = 0;

%saves the length of s1,s2 and initializes an empty martrix
s1Len = length(s1);
s2Len = length(s2);
alignScoreMatrix = zeros(s1Len+1, s2Len+1);
alignOrigMatrix = zeros(s1Len, s2Len, 2);
gapIdx = 5;


%initializion for global matrices
if(strcmp('global', type))
    gapPenalty = - scoreMatrix(gapIdx, gapIdx - 1);
    gapVecs1 = (0:gapPenalty:gapPenalty*s1Len)*-1;
    gapVecs2 = (0:gapPenalty:gapPenalty*s2Len)*-1;
    alignScoreMatrix(1:1, :) = gapVecs2;
    alignScoreMatrix(:, 1:1) = gapVecs1;

end
    



%builds the score matrix for the current alignment
for seq1_idx = (2 : s1Len + 1)
    
    seq1_letter = find(letterKeySet == s1(seq1_idx - 1));
    for seq2_idx = (2 : s2Len + 1)
        seq2_letter = find(letterKeySet == s2(seq2_idx - 1));
        
        %the recursion rule
        alignBoth = scoreMatrix(seq1_letter, seq2_letter) + alignScoreMatrix(seq1_idx - 1, seq2_idx - 1);
        gap_s2 = scoreMatrix(gapIdx, seq2_letter) + alignScoreMatrix(seq1_idx - 1, seq2_idx);
        s1_gap = scoreMatrix(seq1_letter, gapIdx) + alignScoreMatrix(seq1_idx, seq2_idx - 1);
        
        %gets the best score and the tile we came from (in case of local
        %choose the maximum out of 4 arguments - zero is added)
        if(strcmp('global', type))
            [iterMaxScore, maxScoreIdx] = max([alignBoth, s1_gap, gap_s2]);
        else
            [iterMaxScore, maxScoreIdx] = max([alignBoth, s1_gap, gap_s2, 0]);
            
           
        end
        
        %saves the tile we came from(alignOrigMatrix) and the max score for
        %this move (alignScoreMatrix)
        alignScoreMatrix(seq1_idx, seq2_idx) = iterMaxScore;
        alignOrigMatrix(seq1_idx - 1, seq2_idx - 1, s1Orig:s1Orig)  = seq1_idx - locsKeySet(maxScoreIdx:maxScoreIdx, s1Orig:s1Orig);
        alignOrigMatrix(seq1_idx - 1, seq2_idx - 1, s2Orig:s2Orig)  = seq2_idx - locsKeySet(maxScoreIdx:maxScoreIdx, s2Orig:s2Orig);
        
        
        %for local nullify the location 
        if(maxScoreIdx == 4)
            alignOrigMatrix(seq1_idx, seq2_idx, s1Orig:s2Orig) = [0, 0];
        end 
    end 
end

%extract the best score and i'ts index - for local
if(strcmp('local',type))
    [bestColScore, xAxisIdx] = max(alignScoreMatrix(:,:));
    [bestScore, yAxisIdx] = max(bestColScore);
    bestScoreCor = [xAxisIdx(yAxisIdx), yAxisIdx];
end

   

