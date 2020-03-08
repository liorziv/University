

function [imageWithmarkedROIs] = markROIsInImage(numFrames, CC,ImageSeq) 
% Marks all the given ROIs in the image
%% Input :
% numFrames - the total number of images in one time frame
% CC - the ROIs we want to mark
% imageSeq - the image we want to display the ROIs on
%% Output : 
% imageWithmarkedROIs - the image with the given ROIs marked

%containes both ROI's on the current image
imageWithmarkedROIs = ImageSeq;


%creates markedROIsBeforeOnlyCC which containes only the
%pixels which appear in CCOrig
for frame = 1:numFrames
    for groupNum = 1:length(CC)
        
        
        for i = CC{groupNum}
 
           
            imageWithmarkedROIs(i) =1000;
            
            
        end
       
    end
end
 end



