

function [translatedROIs, inFrameROIs,picXOffset, picYOffset] = findCorr(sourceImage, moveImage, BW1, BW2, imaheTitle)

% Finds the needed translation between two images
%% Input:
% sourceImage - the image which statys the same, no translation
% moveImage - the image which we calculate the tranlsation accoring to
% BW1 - before image contianing filled ROIS
% BW2 - same with after

%% Output:
% translatedROIs - the ROIs coordinates from the moved image, after tranlsation
% inFrameTOIs - finds the original ROIs before translation
% picXOffset - the x offset
% picYOffset - the y offset
rect1 = [200 200 200  200]; %start frame for the moving image

subMove = imcrop(moveImage,rect1);     %crops the image accrdingly


c=normxcorr2(subMove,sourceImage); %cross corelation
% check that there is a global peak, good match
% figure, surf(c), shading flat

%the peak and offset here are the same
[ypeak, xpeak] = find(c==max(abs(c(:))));



 %calculets the offset
 offset = [(xpeak-size(subMove,2)) (ypeak-size(subMove,1))];
 xoffset = offset(1);
 yoffset = offset(2);
 
  
% In order to find the intersection between the two images
 translatedImage = imtranslate(BW1,[xoffset-200, yoffset-200]);
 translatedROIs = bwconncomp(translatedImage);
 
 picXOffset = xoffset-200;
 picYOffset = yoffset-200;

 
 %in order to find points which exist only in one of the frames (the
 %tranlation between images caused misses of few ROI's)
 originalFrame = imtranslate(translatedImage,[-xoffset+200, -yoffset+200]);
 inFrameROIs = bwconncomp(originalFrame);
 
 
 

 figure, imshowpair(BW2,translatedImage,'falsecolor');
 title(imaheTitle);
end