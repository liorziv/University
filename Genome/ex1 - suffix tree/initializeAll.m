%initializes a suffix tree for each sequemce in the Seqs cell array
function [G] = initializeAll(Seqs)
seqSize = size(Seqs.seqs);
G = cell(seqSize(2),1);
for k = 1:seqSize(2)
    tmp = initialize(Seqs.seqs(k));
    G{k} = tmp;
    
end
end