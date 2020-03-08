function [toAddToImage1, addedToImage1SortedIdx, idx1, idx2] = findIntersection(translatedROIs, CC)

%finds intersection in the data
%% Input :
% translted ROIs are the moved ones, CC is the original ones
%% Output : 
% toAddToImage1 - is a cell array contianing all the indexs to add to the
% original picture, no intersection (addToBefore)
% addedToImage1SortedIdx - The order of added index from translatedROIs to
% the image. The index from translatedROIs which match the added group to toAddToImage1
% idx1 - the order of ROI's which intersect from translatedROIs
% idx2 - the order of ROI's which intersect from CC

addedToImage1SortedIdx = {};
toAddToImage1 = {};
idx1 = {};
idx2 = {};
counter1 = 1;
for i = 1:length(translatedROIs)
    intersects = 0;
    group1 = translatedROIs{i};
    for j = 1:length(CC)
        group2 = CC{j};
        AandBIntersect = (~isempty(intersect(group1,group2)));
        if(AandBIntersect)
            intersects = 1;
            idx1{end+1} = i;
            idx2{end+1} = j;
        end
    end
    if(~intersects)
        toAddToImage1{counter1} = group1;
        counter1 = counter1 + 1;
        %list of indexes from the translated ROI
        %in order to add them in the same order at the original(before
        %translation
        addedToImage1SortedIdx{end+1} = i;
    end
            
end
    
end
