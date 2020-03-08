
function [suffTree] = suffixTree(S,startNode)

%initialize the root
suffTree = cell(1,2);
suffTree{1}{1} = 'R'; %label
suffTree{1}{2} = zeros(1,1); %childran index
nodeCtr = 2;
for i = 1:length(S)
    s = S(i:end);
[suffTree,nodeCtr] = createTree(suffTree, S(i:end),startNode,nodeCtr);
end

end
function [suffTree,nodeCtr] = createTree(suffTree, S, startNode, nodeCtr)
    if(isempty(S))
        return;
    end

    str = S;
    childIdx = findInKidsList(suffTree,startNode,S(1));
    if(~isempty(childIdx)) %checks if there is a child with the same first letter
        childIdx = suffTree{startNode}{2}(childIdx);
        %str2 = suffTree{childIdx}{1};
        [common_word,s] = prefix({suffTree{childIdx}{1},S});
        if(length(common_word) < length(suffTree{childIdx}{1}))
            
            newNodeIdx = nodeCtr;
            nodeCtr = nodeCtr + 1;

            suffTree{newNodeIdx}{1} = common_word;
            suffTree{newNodeIdx}{3} = startNode;
            suffTree{childIdx}{1} = suffTree{childIdx}{1}(length(common_word)+1:end); 
            suffTree{childIdx}{3} = newNodeIdx;
            suffTree{newNodeIdx}{2} = zeros(1,1);
            suffTree{newNodeIdx}{2}(end+1) = childIdx;
            suffTree{nodeCtr}{1} = S(length(common_word)+1:end); 
            suffTree{nodeCtr}{2} = zeros(1,1);
            suffTree{nodeCtr}{3} = newNodeIdx; 
            suffTree{newNodeIdx}{2}(end+1) = nodeCtr; 
            nodeCtr = nodeCtr + 1;
            suffTree{startNode}{2}(suffTree{startNode}{2} == childIdx) = newNodeIdx;
        else
            [suffTree,nodeCtr] = createTree(suffTree, S(2:end), childIdx, nodeCtr);
         end
        

        
            
    else %creates a new node
        suffTree{startNode}{2}(end + 1) = nodeCtr; %adds s child 
        suffTree{nodeCtr}{1} = S;
        suffTree{nodeCtr}{2} = zeros(1,1);
        suffTree{nodeCtr}{3} = startNode; 
        nodeCtr = nodeCtr + 1; 

    end
end

function [idx] = findInKidsList(suffTree, startNode, preS)
childranIdx = suffTree{startNode}{2};
childran = cell(length(childranIdx)-1,1);
ctr = 1;
for j = childranIdx(2:end)
    childran{ctr} = suffTree{j}{1}(1);
    ctr = ctr + 1;
end
idx = find(cell2mat(childran) == preS) + 1;
end


function [ST] = initialize(S)
startNode = 1;
suffTree = suffixTree(S,startNode);
end

function [G] = initializeAll(Seqs)
seqSize = length(Seqs);
G = cell(seqSize,1);
for i = 1:seqSize
    G(i) = initialize(Seqs(i));
    
end
end

