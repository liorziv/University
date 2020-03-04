%% Finds the best alignment of s1  and s2 
% s1 - fasta file of sequence 1 (e.g. 'fastas/HomoSapiens-SHH.fasta')
% s2 - fasta file of sequence 2
% type - type of alignment (e.g. 'local')
% score - the score matrix in .tsv fileformat (e.g. 'score_matrix.tsv')
%%
function [ ] = seq_align(s1, s2, type, score)


%imports score_matrix.tsv into a matrix
scoreMatrix = ImportScoreMatrix(score);

%reads the fasta files 
[~,s1] = fastaread(s1);
[~,s2] = fastaread(s2);



switch type
    case 'global'
  
        %creates an origin matrix and alignment score matrix
        [alignScoreMatrix, alignOrigMatrix, ~, ~ ] = alignSequences(s1, s2, scoreMatrix, type);
        
        %exract the best alignment sequences
        bestScoreCor = [length(s1) + 1,length(s2) + 1];
        [seq1,seq2] = retrieveOptimalAlignment(s1, s2, alignScoreMatrix, alignOrigMatrix, type, bestScoreCor);
        
        
        display(alignScoreMatrix(end:end, end:end), 'The score is');
        display(seq1(1:50));
        display(seq2(1:50));
        
    case 'local'
        
        %creates an origin matrix and alignment score matrix
        [alignScoreMatrix,alignOrigMatrix, bestScore, bestScoreCor ] = alignSequences(s1, s2, scoreMatrix, type);
        
        %exract the best alignment sequences
        [seq1,seq2] = retrieveOptimalAlignment(s1, s2, alignScoreMatrix, alignOrigMatrix, type, bestScoreCor);
        
        display(bestScore, 'The score is');
        display(seq1(1:50));
        display(seq2(1:50));
end



 