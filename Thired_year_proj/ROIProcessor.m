function [] = ROIProcessor(workingDirectoryBefore, workingDirectoryAfter, compFlag, dispEdgeDec, dispTranslation, runNum)

%% Gets file/s path and :
% If compFlag is true meaning we got two time frames we want to compare,
% otherwise we just want to exctact the first file data
% workingDirectoryBefore -  is the path to the before exposure images
% workingDirectoryAfter -  is the path to the after exposure images
% dispEdgeDec - if true display figures after edge detection phase
% dispTranslation - if true display figures after translation
if(compFlag)
    % loads the images from both time frames
    [imageSeqBeforeStd, imageSeqAfterStd, imageSeqBefore, imageSeqAfter, numOfFrames] = loadAndConvertImagesC(workingDirectoryAfter, workingDirectoryBefore);

    % Detect ROIs using the std image per time frame
    % displayFig - if true will display the edge detection image of both of the
    % time frames
    [CCBefore, CCAfter, BW1, BW2] = detectROIsC(imageSeqBeforeStd, imageSeqAfterStd, dispEdgeDec);
    
    %Gets the total ROI in each picture includeing image translation
    [existInBothBeforeIdx, existInBothAfterIdx, addedFromTransToAfterIdx, addedFromTransToBeforeIdx, addToAfter, addToBefore, translatedROIsBefore, translatedROIsAfter] = translateImages(BW1, BW2, imageSeqBeforeStd, imageSeqAfterStd, CCAfter, CCBefore, numOfFrames, dispTranslation);
    
    %analysis part 
    dataAnalysisC(existInBothBeforeIdx, existInBothAfterIdx, addedFromTransToAfterIdx, addedFromTransToBeforeIdx, addToAfter, addToBefore, translatedROIsBefore, translatedROIsAfter, imageSeqAfter, imageSeqBefore, numOfFrames, runNum);


else
    % loads the images from a given time frame
    [imageSeqStd, imageSeq, numOfFrames] = loadAndConvertImagesS(workingDirectoryBefore);
    
    % Detect ROIs using the std image per time frame
    % displayFig - if true will display the edge detection image 
    CC = detectROIsS(imageSeqStd, dispEdgeDec);
    
    %analysis part 
    dataAnalysisS(CC,imageSeq, numOfFrames, runNum);

    

end


cd ..

% end