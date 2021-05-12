function [] = aggClust(X, type)
sizeOfSet = size(X);
xLen = length(X);
newSet = num2cell(X,2);
T = zeros(xLen-1,3);
for i = 1: sizeOfSet(1) -1
    minmalVal = inf;
    node1 = 0;
    node2 = 0;
    for k = 1: xLen
        a = newSet{k,:};
        for j = k: xLen
            b = newSet{j,:};
            if(k == j)
                continue;
            end
            if(type == 1) %avrage linkage
                dist = linkageType(a,b,type);
            end
            if(minimalVal > dist)
            minimalVal = dist;
            node1 = k;
            node2 = j;
            end
        end
    end
    newSet = updateNewSet(newSet,node1,node2);
   
    xLen = length(newSet);
    
%     T(i,1) = node1;
%     T(i,2) = node2;
%     T(i,3) = dist/2;
    
end
    
    
end
function [newSet] = updateNewSet(newSet,node1,node2)
for a = newSet{node1}
          
% out{1,1}(end+1) = {1}
        newSet{node2:node2}(end+1) = a; 
end
     newSet(node1,:) = [];
end
