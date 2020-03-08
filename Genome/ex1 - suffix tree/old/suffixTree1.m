function [suffTree] = suffixTree(S,startNode)

%initialize the root
suffTree = cell(1,2);
suffTree{1}{1} = 'R'; %label
suffTree{1}{2} = zeros(1,1); %childran index
nodeCtr = 2;

suffTree = createTree(suffTree, S,startNode,nodeCtr);

end
function [suffTree,nodeCtr] = createTree(suffTree, S, startNode, nodeCtr)
display(startNode);
if(length(S) == 1)
    subStringIdx2 = strmatch(S,suffTree{startNode}{1});
    if(length(suffTree{startNode}{2}) == 1 | subStringIdx2==0)
        suffTree{startNode}{2}(end + 1) = nodeCtr; %adds s child 
        suffTree{nodeCtr}{1} = strcat(S,'$');
        suffTree{nodeCtr}{3} = startNode; 
        return;
    else
        
            suffTree{startNode}{2}(end + 1) = nodeCtr; %adds s child 
            suffTree{nodeCtr}{1} = '$'; 
            suffTree{nodeCtr}{2} = zeros(1,1);
            suffTree{nodeCtr}{3} = startNode; 
            nodeCtr = nodeCtr + 1;
            suffTree{startNode}{2}(end + 1) = nodeCtr;
            suffTree{nodeCtr}{1} = suffTree{startNode}{1}(subStringIdx2+1:end);
            suffTree{nodeCtr}{2} = zeros(1,1);
            suffTree{nodeCtr}{3} = startNode;
            nodeCtr = nodeCtr + 1;
            suffTree{startNode}{1} = S;
            for j = 2:suffTree{startNode}{2} 
                suffTree{j}{3} = nodeCtr-1;
            end
        return;
        
    end  
    
end
for i = 1:length(S)
    if(length(suffTree{startNode}{2}) == 1)%if the node has no kids
       
        
        
        %checks for the substring case
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
            
        else %the root since it has no kids and it's not a substring
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
        %if not child has that start including the root
        if(isempty(suffIdx) & S(i) ~= suffTree{startNode}{1}(1))
            suffTree{nodeCtr}{1} = strcat(S(i:end),'$');
            suffTree{nodeCtr}{2} = zeros(1,1);
            suffTree{nodeCtr}{3} = startNode;
            suffTree{startNode}{2}(end + 1) = nodeCtr;
            nodeCtr = nodeCtr + 1;
        else
            
            if(suffTree{startNode}{1}(1) ~= S(i))
                newStart = suffIdx+1; 
            else
                newStart = startNode;
            end
            
            [suffTree,nodeCtr] = createTree(suffTree, S(i:end), newStart, nodeCtr);
        end
           
        
    end
    
   
end
  
end

