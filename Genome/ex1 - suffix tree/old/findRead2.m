
%The main function - finds a read index inside each 
%sequence from the G genome structure
function [dispStr] = findRead2(r,G)
dispStr = '';
Glen = size(G);
if(iscell(r))
    r = r{1};
end

for i = 1:Glen(1)
dispStr = strcat(dispStr,findReadHelper(1,r,G(i),i));
end
end

function [str] = findReadHelper(nodeIdx,read,suffTree,chromNum)
subRead = read;
readLen = length(read);
contianedInTree = 1;
if(class(suffTree) == 'cell')
    suffTree = suffTree{1};
end
while(~isempty(subRead))
    commonSub = '';
    str = '';
    for i = suffTree{nodeIdx}{2}
        str1 = suffTree{i}{1};
        [commonSub,~] = prefix({suffTree{i}{1},read});
        if(~isempty(commonSub))
            subRead = subRead(length(commonSub):end);
            break;
        end
    end
    if(strcmp(commonSub,''))
        contianedInTree = 0;
        break;
    end
    nodeIdx = i;  
end

if(contianedInTree)
    if(length(suffTree{nodeIdx}{2}) > 1) %this is an internal node
        str = treeTraversal(nodeIdx,readLen,str,suffTree,chromNum);
    else
        str = strcat(str ,'chr' ,num2str(chromNum) , ':' , num2str(i) , '-' , num2str(i + readLen - 1) ,';');

    end
        
end
end

%Go over the tree with the root as nodeIdx
function [dispStr] = treeTraversal(nodeIdx,readLen,dispStr,suffTree,chromNum)
for i = suffTree{nodeIdx}{2}
    if(isempty(suffTree{i}{4}))%internalnode
        dispStr = treeTraversal(i, readLen, dispStr, suffTree,chromNum);
    else
        dispStr = strcat(dispStr ,'chr' ,num2str(chromNum) , ':' , num2str(i) , '-' , num2str(i + readLen - 1) ,';');

    end
end
end

 