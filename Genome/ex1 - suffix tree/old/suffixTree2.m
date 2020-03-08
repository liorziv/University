function [suffTree] = suffixTree(S,startNode)

%initialize the root
suffTree = cell(1,2);
suffTree{1}{1} = 'R'; %label
suffTree{1}{2} = zeros(1,1); %childran index
nodeCtr = 2;

suffTree = createTree(suffTree, S,startNode,nodeCtr);

end
function [suffTree,nodeCtr] = createTree(suffTree, S, startNode, nodeCtr)
% if(length(S) == 1)
%     return;
% end
for i = 1:length(S)
    if(length(suffTree{startNode}{2}) == 1)%if the tree has only a root

        subStringIdx = strmatch(S(i:end),suffTree{startNode}{1});
        if(subStringIdx)%not the root, splits
            
            suffTree{startNode}{2}(end + 1) = nodeCtr; %adds s child 
            suffTree{nodeCtr}{1} = '$'; 
            suffTree{nodeCtr}{2} = zeros(1,1);
            suffTree{nodeCtr}{3} = startNode; 
            nodeCtr = nodeCtr + 1;
            suffTree{startNode}{2}(end + 1) = nodeCtr;
            suffTree{nodeCtr}{1} = suffTree{startNode}{1}(subStringIdx+length(S(i:end)):end);
            suffTree{nodeCtr}{2} = zeros(1,1);
            suffTree{nodeCtr}{3} = startNode;
            nodeCtr = nodeCtr + 1;
            suffTree{startNode}{1} = S(i:end);
            return;
            
        else %the root
            suffTree{startNode}{2}(end + 1) = nodeCtr; %adds s child 
            suffTree{nodeCtr}{1} = strcat(S(i:end),'$');
            suffTree{nodeCtr}{2} = zeros(1,1);
            suffTree{nodeCtr}{3} = startNode; 
            nodeCtr = nodeCtr + 1; 
            
        end
            

    else %checks whether there is a suffix with that starting letter
        childranIdx = suffTree{startNode}{2};
        childran = cell(length(childranIdx)-1,1);
        ctr = 1;
        for j = childranIdx(2:end)
            childran{ctr} = suffTree{j}{1}(1);
            ctr = ctr + 1;
        end
        suffIdx = find(cell2mat(childran) == S(i));
        if(isempty(suffIdx))
            suffTree{nodeCtr}{1} = strcat(S(i:end),'$');
            suffTree{nodeCtr}{2} = zeros(1,1);
            suffTree{nodeCtr}{3} = startNode;
            suffTree{startNode}{2}(end + 1) = nodeCtr;
            nodeCtr = nodeCtr + 1;
        else
            
            [suffTree,nodeCtr] = createTree(suffTree, S(i:end), suffIdx +1, nodeCtr);
        end
           
        
    end
    
   
end
  
end

%wont retrive the actuall tree from the recursion
%plus the structure is not perfect yet.
    
%   
% 
%     addCounter = 0;
%     for j = 2:length(suffTree):2
%         if(strcmp(suffTree{i}{j},S(i)))
%             continue;
%         else
%             display('else');
%             addCounter = addCounter + 1;
%             if(addCounter == length(suffTree))
%                 
%             end
%         end
% function [boolVal] = checkIfAlreadyContained[suffTree