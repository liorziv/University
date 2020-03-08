function [suffTree] = suffixTree(S,startNode)

%initialize the root
suffTree = cell(1,2);
suffTree{1}{1} = 'R'; %label
suffTree{1}{2} = zeros(1,1); %childran index
nodeCtr = 2;
for i = 1:length(S)
suffTree = createTree(suffTree, S,startNode,nodeCtr);
end

end
function [suffTree,nodeCtr] = createTree(suffTree, S, startNode, nodeCtr)


    str = S;
    childIdx = findInKidsList(suffTree,startNode,S);
    if(~isempty(childIdx)) %checks if there is a child with the same first letter
        childIdx = suffTree{startNode}{2}(childIdx);
        str2 = suffTree{childIdx}{1};
        [common_word,s] = prefix({suffTree{childIdx}{1},S(i:end)});
        if(length(common_word) < length(suffTree{childIdx}{1}))
            
            newNodeIdx = nodeCtr;
            nodeCtr = nodeCtr + 1;

            suffTree{newNodeIdx}{1} = common_word;
            suffTree{newNodeIdx}{3} = startNode;
            suffTree{childIdx}{1} = suffTree{childIdx}{1}(length(common_word)+1:end); 
            suffTree{childIdx}{3} = newNodeIdx;
            suffTree{newNodeIdx}{2} = zeros(1,1);
            suffTree{newNodeIdx}{2}(end+1) = childIdx;
            suffTree{nodeCtr}{1} = strcat(S(i+length(common_word):end),'$'); 
            suffTree{nodeCtr}{2} = zeros(1,1);
            suffTree{nodeCtr}{3} = newNodeIdx; 
            suffTree{newNodeIdx}{2}(end+1) = nodeCtr; 
            nodeCtr = nodeCtr + 1;
            suffTree{startNode}{2}(suffTree{startNode}{2} == childIdx) = newNodeIdx;
        else
            suffTree = createTree(suffTree, S(i+1:end), childIdx, nodeCtr);
         end
        

        
            
    else %creates a new node
        suffTree{startNode}{2}(end + 1) = nodeCtr; %adds s child 
        suffTree{nodeCtr}{1} = strcat(S(i:end),'$');
        suffTree{nodeCtr}{2} = zeros(1,1);
        suffTree{nodeCtr}{3} = startNode; 
        nodeCtr = nodeCtr + 1; 

    end
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