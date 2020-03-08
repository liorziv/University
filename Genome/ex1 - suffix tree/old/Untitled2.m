function suffixTree[suffTree] = initialize(S,startNode)

%initialize the root
suffTree = cell(1,2);
suffTree{1}{1} = 'R'; %label
suffTree{1}{2} = zeros(1,1); %childran index
nodeCtr = 2;

for i = 1:length(S)
    if(length(suffTree{1}{2}) == 1)%if the tree has only a root
        suffTree{nodeCtr}{1} = strcat(S(i:end),'$');
        suffTree{nodeCtr}{2} = ' ';
        suffTree{nodeCtr}{3} = 1;
        suffTree{1}{2}(end + 1) = nodeCtr;
        nodeCtr = nodeCtr + 1;
    else
        childranIdx = suffTree{1}{2};
        childran = cell(length(childranIdx)-1,1);
        ctr = 1;
        for j = childranIdx(2:end)
            childran{ctr} = suffTree{j}{1}(1);
            ctr = ctr + 1;
        end
        if(isempty(find(cell2mat(childran) == S(i))))
            suffTree{nodeCtr}{1} = strcat(S(i:end),'$');
            suffTree{nodeCtr}{2} = ' ';
            suffTree{nodeCtr}{3} = nodeCtr + 1 - length(childranIdx);
            suffTree{1}{2}(end + 1) = nodeCtr;
            nodeCtr = nodeCtr + 1;
        else
            
        end
           
        
    end
    
   
end
  
    
    
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