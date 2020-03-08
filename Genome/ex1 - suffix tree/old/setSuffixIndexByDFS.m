function [] = setSuffixIndexByDFS(nodeIdx, labelH, suffTree)
treeSize = size(suffTree);
if(nodeIdx > treeSize(2) | nodeIdx < 1)
    return;
end
if(suffTree{nodeIdx}{1} ~= 'R')
    disp(suffTree{nodeIdx}{1});
end

end