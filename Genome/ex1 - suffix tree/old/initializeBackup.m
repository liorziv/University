
%initialize a single suffix tree
function [ST] = initialize(S)
startNode = 1;
ST = suffixTree(S,startNode);
end

%Helper to create the suffix tree
function [suffTree] = suffixTree(S,startNode)

%initialize the root
suffTree = cell(1,2);
suffTree{1}{1} = 'R'; %label
suffTree{1}{2} = zeros(1,1); %childran index
nodeCtr = 2;
%in order to avoid problem with diffrent input foramts
if(strcmp(class(S),'cell'))
    S = cell2mat(S);
end 

S = strcat(S,'$');
for k = 1:length(S)-1
[suffTree,nodeCtr] = createTree(suffTree, S(k:end),startNode,nodeCtr);

end
for i = 1:nodeCtr-1
    if(length(suffTree{i}{2}) == 1)
        suffTree{i}{2} = [];
    else
        suffTree{i}{2}= suffTree{i}{2}(2:end);
    end
    
end

end

%updates the tree  each different suffix
function [suffTree,nodeCtr] = createTree(suffTree, S, startNode, nodeCtr)
    if(isempty(S))
        return;
    end

    str = S;
    childIdx = findInKidsList(suffTree,startNode,S(1));
    if(~isempty(childIdx)) %checks if there is a child with the same first letter
        childIdx = suffTree{startNode}{2}(childIdx);
        %str2 = suffTree{childIdx}{1};
        [commonSub,~] = prefix({suffTree{childIdx}{1},S});
        if(class(commonSub) == 'cell')
            commonSub = commonSub{1};
        end
        if(length(commonSub) < length(suffTree{childIdx}{1}))
            
            newNodeIdx = nodeCtr;
            nodeCtr = nodeCtr + 1;

            suffTree{newNodeIdx}{1} = commonSub;
            suffTree{newNodeIdx}{3} = startNode;
            suffTree{childIdx}{1} = suffTree{childIdx}{1}(length(commonSub)+1:end); 
            suffTree{childIdx}{3} = newNodeIdx;
            suffTree{newNodeIdx}{2} = zeros(1,1);
            suffTree{newNodeIdx}{2}(end+1) = childIdx;
            suffTree{nodeCtr}{1} = S(length(commonSub)+1:end); 
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
        %suffTree{startNode}{2}= suffTree{startNode}{2}(2:e
        suffTree{nodeCtr}{1} = S;
        suffTree{nodeCtr}{2} = zeros(1,1);
        suffTree{nodeCtr}{3} = startNode; 
        nodeCtr = nodeCtr + 1; 

    end
end

function [idx] = findInKidsList(suffTree, startNode, preS)
childranIdx = suffTree{startNode}{2};
if(childranIdx == 0)
    idx = [];
    return;
else
    childran = cell(length(childranIdx)-1,1);
    ctr = 1;
    for j = childranIdx(2:end)
        childran{ctr} = suffTree{j}{1}(1);
        ctr = ctr + 1;
    end
    
end
idx = find(cell2mat(childran) == preS);
if(~isempty(idx))
     idx = idx +1;
   
end

end





