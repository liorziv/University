%Go over the tree with the root as nodeIdx
function [dispStr] = treeTraversal2(nodeIdx,suffTree)
leafNum = countNumOfLeafs(suffTree);
leafIndex = cell(1,leafNum);


suff = '';
leafIndex = trav(1,suffTree,suff); 

end
function [suff] = trav(nodeIdx,suffTree,suff)
for i = suffTree{nodeIdx}{2}
    if(isempty(suffTree{i}{4}))%internalnode
        suff = trav(i, suffTree);
    else
        suff = strcat(suff,suffTree{i}{4});

    end
end
end
function [numOfLeafs] = countNumOfLeafs(suffTree)

numOfLeafs = 0;

for i = 2:length(suffTree)
    if(length(suffTree{i}) == 4)
        if(~isempty(suffTree{i}{4}))
            numOfLeafs = numOfLeafs + 1;
        end
    end
end
end
