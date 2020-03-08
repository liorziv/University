

function [updatedCC] = deleteUnder20Pix(CC)
%delets ROIs which are under 20 pixels
%%Input : 
% CC - a connected components object including all the detected ROIs
%% Output :
% updateCC - the set contianing only ROIs with 20 pixels and above.

updatedCC = CC;    
for i = CC.NumObjects:-1:1
    if(length(updatedCC.PixelIdxList{i}) < 20)
        updatedCC.PixelIdxList(i) = [];
        updatedCC.NumObjects = updatedCC.NumObjects -1;
    end
end

for i = 1:updatedCC.NumObjects
    if(length(updatedCC.PixelIdxList{i}) < 20)
        updatedCC.PixelIdxList{i} = [];
    end
end
end
