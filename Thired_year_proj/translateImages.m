
function [existInBothBeforeIdx, existInBothAfterIdx, onlyAfter, onlyBefore, addToAfter, addToBefore, translatedROIsBefore, translatedROIsAfter] = translateImages(BW1, BW2, imageSeqBeforeStd, imageSeqAfterStd, CC1After, CC1Before, numOfFrames, dispTranslation)
% ************************************************************************************
% Use cross correlation in order to find the best translation between the
% two time frames
% ************************************************************************************

%% Input :
% BW1 - image of before where all the detected ROIs are filled 
% BW2 - image of after where all the detected ROIs are filled 
% imageSeqBeforeStd - an std image of the before sequence
% imageSeqAfterStd - an std image of the after sequence
% CC1After - connected components object of after time frame
% CC1Before - connected components object of Before time frame
%numOfFrames - the number of frames at a single time frame
% dispTranslation - if true display an figure of translation steps
%% Output : 
% existInBothBeforeIdx - shared ROIs index by before time frame
% existInBothAfterIdx - shared ROIs index by after time frame
% onlyAfter - ROIs which appear only after
% onlyBefore - ROIs which appear only before
% addToAfter - ROIs to add to after
% addToBefore - ROIs to add to before
% translatedROIsBefore - ROIS of before time frame translated
% translatedROIsAfter - ROIS of after time frame translated 


%translation in order to fit before time frame to the after time frame
[translatedROIsBefore, inFrameBeforeROIsTmp, xoffset1, yoffset1] = findCorr(imageSeqAfterStd, imageSeqBeforeStd, BW1, BW2, 'Move befoer according to after - Image after translation, white is overlapping points');
%translation in order to fit after time frame to the before time frame
[translatedROIsAfter, inFrameAfterROIsTmp,xoffset2, yoffset2] = findCorr(imageSeqBeforeStd, imageSeqAfterStd, BW2, BW1, 'Move after according to before - Image after translation, white is overlapping points');


%check which tranalation took more ROIs, therefore we should use it
if(translatedROIsAfter.NumObjects > translatedROIsBefore.NumObjects)
    translatedROIsBefore = CC1Before;
    inFrameBeforeROIsTmp = CC1Before;
    xoffset = xoffset2;
    yoffset = yoffset2; 
    display('translate after');
else
    translatedROIsAfter = CC1After;
    inFrameAfterROIsTmp = CC1After;
    xoffset = xoffset1;
    yoffset = yoffset1;
    display('translate before');
end



%finds points in image1 which dont intersect with points in image2 
%and makes a cell array of all this points in order to add them to the images 
%returns the CC since with the cross corrlation we might droped points that
%are present only in one figure(since it is translated)
%addToBefore - only in After list - we want to add them to before
%image(from translatedROIsAfter)
%onlyAfter - appear in the after frame not shared
%existInBothAfterIdx1/existInBothBedoreIdx1 - the list of shared indexs
%by index order from each time frame 

[addToBefore, onlyAfter,existInBothAfterIdx1, existInBothBeforeIdx1] = findIntersection(translatedROIsAfter.PixelIdxList, inFrameBeforeROIsTmp.PixelIdxList);
[addToAfter, onlyBefore, existInBothBeforeIdx2, existInBothAfterIdx2] = findIntersection(translatedROIsBefore.PixelIdxList,inFrameAfterROIsTmp.PixelIdxList);

%not in frame are ROIs that are loacted only in the before frame
% in frame ROI's are the tranlsated back to the original cc ROI's (only
% those that are in the checked frame)

[~, ~, ~, inFrameBeforeROIsIdx ] = findIntersection(CC1Before.PixelIdxList, inFrameBeforeROIsTmp.PixelIdxList);
[~, ~, ~, inFrameAfterROIsIdx] = findIntersection(CC1After.PixelIdxList, inFrameAfterROIsTmp.PixelIdxList);





if(length(existInBothBeforeIdx1)> length(existInBothBeforeIdx2))
    existInBothBeforeIdx = cell2mat(existInBothBeforeIdx1);
else
    existInBothBeforeIdx = cell2mat(existInBothBeforeIdx2);
end

if(length(existInBothAfterIdx1)> length(existInBothAfterIdx2))
    existInBothAfterIdx = cell2mat(existInBothAfterIdx1);
else
    existInBothAfterIdx = cell2mat(existInBothAfterIdx2);
end

% all ROIs which appear in a specific time frame (after translation)
inFrameBeforeROIsIdx = cell2mat(inFrameBeforeROIsIdx);
inFrameAfterROIsIdx = cell2mat(inFrameAfterROIsIdx);

% ROIs which after translation got out of one of the time frames images, 
% therfore we exclude them from our comparsion since we can't really
% compare them.
notInFrameBeforeROIsIdx = setdiff(1:CC1Before.NumObjects,inFrameBeforeROIsIdx);
notInFrameAfterROIsIdx = setdiff(1:CC1After.NumObjects,inFrameAfterROIsIdx);

% ROIs which were detected only in one of the time frames
onlyAfter = cell2mat(onlyAfter);
onlyBefore = cell2mat(onlyBefore);
  
%% sanity check 
if(dispTranslation)    
    % In order to see that shared ROIs are loacted in a logical place
    % compared to each other
    [markedROICombined11] = markROIsInImage(numOfFrames, translatedROIsBefore.PixelIdxList(existInBothBeforeIdx), imageSeqBeforeStd); 
    [markedROICombined12] = markROIsInImage(numOfFrames, translatedROIsAfter.PixelIdxList(existInBothAfterIdx), imageSeqAfterStd); 


    figure
    imagesc(markedROICombined11)
    title('Shared Before');

    figure
    imagesc(markedROICombined12)
    title('Shared After');

 
    % In order to check the isolated ROIs are located logically
    [markedROICombined13] = markROIsInImage(numOfFrames, CC1Before.PixelIdxList(notInFrameBeforeROIsIdx),imageSeqBeforeStd); 
    [markedROICombined14] = markROIsInImage(numOfFrames, CC1After.PixelIdxList(notInFrameAfterROIsIdx),imageSeqAfterStd);

    figure
    imagesc(markedROICombined13)
    title('excluded from before');
    figure
    imagesc(markedROICombined14)
    title('excluded from after');

    % In order to see the exclusive ROIs
    [markedROICombined1] = markROIsInImage(numOfFrames, inFrameAfterROIsTmp.PixelIdxList(onlyAfter),imageSeqAfterStd); 
    [markedROICombined2] = markROIsInImage(numOfFrames, inFrameBeforeROIsTmp.PixelIdxList(onlyBefore),imageSeqBeforeStd); 


    figure
    imagesc(markedROICombined1)
    title('exist only after');
    figure
    imagesc(markedROICombined2)
    title('exist only before');

end
