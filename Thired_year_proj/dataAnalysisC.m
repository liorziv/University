function [] = dataAnalysisC(existInBothBeforeIdx, existInBothAfterIdx, addedFromTransToAfterIdx, addedFromTransToBeforeIdx, addToAfter, addToBefore, translatedROIsBefore, translatedROIsAfter, imageSeqAfter, imageSeqBefore, numOfFrames, runNum)

%creates a new folder for this run
mkdir (runNum)

%sorts accoring to size in order to compare same ROI's in before and after
maxLen =  max(translatedROIsBefore.NumObjects, translatedROIsAfter.NumObjects);

%finds the mean for shared ROIs
sharedMeanBeforePerFrame = getROIMean(existInBothBeforeIdx, translatedROIsBefore.PixelIdxList, imageSeqBefore , maxLen, numOfFrames); 
sharedMeanAfterPerFrame = getROIMean(existInBothAfterIdx, translatedROIsAfter.PixelIdxList, imageSeqAfter , maxLen, numOfFrames); 

sharedMeanBeforePerFrameNormelized = normalizeData(sharedMeanBeforePerFrame); 
sharedMeanAfterPerFrameNormelized = normalizeData(sharedMeanAfterPerFrame); 



%finds the mean for the exclusive before ROIs
existOnlyBeforeMeanPerFrame =  getROIMean(addedFromTransToBeforeIdx, translatedROIsBefore.PixelIdxList, imageSeqBefore , maxLen, numOfFrames); 
addedToAfterFromBeforeMeanPerFrame = getROIMean(addedFromTransToBeforeIdx, addToAfter, imageSeqAfter , maxLen, numOfFrames); 

existOnlyBeforeMeanPerFrameNormelized = normalizeData(existOnlyBeforeMeanPerFrame); 
addedToAfterFromBeforeMeanPerFrameNormelized = normalizeData(addedToAfterFromBeforeMeanPerFrame); 


%finds the mean for the exclusive after ROIs
existOnlyAfterMeanPerFrame =  getROIMean(addedFromTransToAfterIdx, translatedROIsAfter.PixelIdxList, imageSeqAfter , maxLen, numOfFrames); 
addedToBeforeFromAfterMeanPerFrame = getROIMean(addedFromTransToAfterIdx, addToBefore,  imageSeqBefore , maxLen, numOfFrames); 
existOnlyAfterMeanPerFrameNormelized = normalizeData(existOnlyAfterMeanPerFrame); 
addedToBeforeFromAfterMeanPerFrameNormelized = normalizeData(addedToBeforeFromAfterMeanPerFrame); 



cd (char(runNum))

if(~isempty(addedFromTransToAfterIdx))
    save('existOnlyAfterMeanPerFrameNormelized.mat', 'existOnlyAfterMeanPerFrameNormelized');
    save('addedToBeforeFromAfterMeanPerFrameNormelized.mat', 'addedToBeforeFromAfterMeanPerFrameNormelized');
end
if(~isempty(addedFromTransToBeforeIdx))  
    save('existOnlyBeforeMeanPerFrameNormelized.mat', 'existOnlyBeforeMeanPerFrameNormelized');
    save('addedToAfterFromBeforeMeanPerFrameNormelized.mat', 'addedToAfterFromBeforeMeanPerFrameNormelized');
end

save('sharedMeanBeforePerFrameNormelized.mat', 'sharedMeanBeforePerFrameNormelized');
save('sharedMeanAfterPerFrameNormelized.mat', 'sharedMeanAfterPerFrameNormelized');


function [pixMeanPerFrame] = getROIMean(idxs, CC, imageSeq, maxLen,numOfFrames) 
pixMeanPerFrame = zeros(numOfFrames,maxLen);
elem = 1;idxs =  unique(idxs);

for i = idxs
    numOfElements =  length(CC{elem});
    for frame = 1:numOfFrames
        currImage = imageSeq(:, :, frame);
        pixMeanPerFrame(frame,i) =  sum(currImage(CC{elem}))/numOfElements;
        
    end
    elem = elem +1;
end
end


function [] = plotROIOverTime(ROIdataBefore,ROIdataAfter, idx)
    figure;findpeaks(smooth(ROIdataBefore, 40), 'minpeakwidth',10);
    hold on;
    findpeaks(smooth(ROIdataAfter,40),'minpeakwidth',10);
    title(['ROI #' , num2str(idx)]);
    
end

end


