function [] = dataAnalysisS(CC,imageSeq, numOfFrames, runNum)


%creates a new folder for this run
mkdir (char(runNum))

%sorts accoring to size in order to compare same ROI's in before and after
len = CC.NumObjects;

%finds the mean for shared ROIs
totalROIMean = getROIMean(1:len, CC.PixelIdxList, imageSeq , len, numOfFrames); 
totalROIMeanNormelized = normalizeData(totalROIMean); 



cd (char(runNum))
save('totalROIMeanNormelized.mat', 'totalROIMeanNormelized');

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


