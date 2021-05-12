%The main function - finds a read index inside each 
%sequence from the G genome structure
function [dispStr] = findRead(r,G)
dispStr = '';
Glen = size(G);
if(iscell(r))
    r = r{1};
end

for i = 1:Glen(1)
dispStr = strcat(dispStr,findReadHelper(1,r,G(i),i));
end
end

%A helper function, finds the first common prefix with a node
%-> searches all the childran of that node for the read 
function [str] = findReadHelper(nodeIdx,read,suffTree,chromNum)
subRead = read;
readLen = length(read);
contianedInTree = 1;
if(iscell(suffTree))
    suffTree = suffTree{1};
end
%goes in the loop untill finds all the read
while(~isempty(subRead))
    commonSub = '';
    str = '';
    for i = suffTree{nodeIdx}{2}
        str1 = suffTree{i}{1};
        %finds the common part
        [commonSub,~] = prefix({suffTree{i}{1},subRead});
        if(~isempty(commonSub))
            subRead = subRead(length(commonSub)+1:end);
            if(length(subRead) ~= 0 & length(commonSub) < length( suffTree{i}{1}))
                contianedInTree = 0;
            end
            break;
        end
    end
    if(strcmp(commonSub,''))
        contianedInTree = 0;
        break;
    end
    %continue with the parent of the common part
    nodeIdx = i;  
end

%when we are here it means we found the read in the tree, now we are
%running on it's chilran to find all the substrings contained with that
%read inside
if(contianedInTree)
    if(length(suffTree{nodeIdx}{2}) > 1) %this is an internal node
        str = getLeafIndex(nodeIdx,readLen,str,suffTree, chromNum);
    else
        str = strcat(str ,'chr' ,num2str(chromNum) , ':' , num2str(suffTree{i}{4}) , '-' , num2str(suffTree{i}{4} + readLen - 1) ,';');

    end
        
end
end

%Go over the tree with the root as nodeIdx
function [dispStr] = getLeafIndex(nodeIdx,readLen,dispStr,suffTree,chromNum)
for i = suffTree{nodeIdx}{2}
    if(isempty(suffTree{i}{4}))%internalnode
        dispStr = getLeafIndex(i, readLen, dispStr, suffTree,chromNum);
    else
        dispStr = strcat(dispStr ,'chr' ,num2str(chromNum) , ':' , num2str(suffTree{i}{4}) , '-' , num2str(suffTree{i}{4}) + readLen - 1  ,';');

    end
end
end



 