%%receives a dataset X and calculates its hierarchical agglomerative clustering.
function [T] = aggClust(X, type)
sizeOfSet = size(X);    %for the outer loop
newSet = num2cell(X,2);
xLen = length(newSet);  %for the inner loops - changes through iterations
T = zeros(xLen-1,3);
for i = 1: sizeOfSet(1) -1
    minimalVal = inf;
    
    for k = 1: xLen - 1
        a = newSet{k,:};
        
        %checks that the cell is not empty
        if(checkIfValid(a))
            for j = k+1: xLen
                b = newSet{j,:};
           
                if(checkIfValid(b))
                    
                    %calculates the distance according to the given type
                    dist = linkageType(a,b,type);

                    if(minimalVal >= dist)
                    minimalVal = dist;
                    node1 = k;
                    node2 = j;

                    end
                end
            end   
        end
    end
        
    %add the merged cells to the end of the set
    newSet{end + 1} =  [newSet{node2};newSet{node1}];
    
    %mark the merged cells in order not to use them again
    newSet{node1} = [];
    newSet{node2} = [];
    %update the length
    xLen = length(newSet);
    
    %update the tree 
    T(i,1) = node1;
    T(i,2) = node2;
    T(i,3) = minimalVal;
end
    
end
%%checks if a given variable is not empty
function [flag] = checkIfValid(val)
flag = 1;
if(~iscell(val))
    if(isempty(val))
        flag = 0;
    end
else
    if(isempty(val{1}))
        flag = 0;
    end
end

end
    

