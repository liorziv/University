
function [CCBefore, CCAfter, BW1, BW2] = detectROIsC(imageSeqBeforeStd, imageSeqAfterStd, displayFig)

% ************************************************************************************
% edge detection in order to find ROIs
% performing CC algorithem in order to find the actual index of ROIs.
% 
% ************************************************************************************

%% Input :
% imageSeqBeforeStd - an std image of the before sequence
% imageSeqAfterStd - an std image of the after sequence
% displayFig - if true diplays the ROIs dectection results
%% Output :
% CCBefore - the before connected components
% CCBefore - the After connected components
% BW1 - image of before where all the detected ROIs are filled 
% BW2 - image of after where all the detected ROIs are filled 

sensitivity = 0.100; %much more signals
edgeAfter = edge(imageSeqAfterStd, 'Canny',sensitivity);

sensitivity = 0.250;
edgeBeore = edge(imageSeqBeforeStd, 'Canny',sensitivity);

if(displayFig)
    figure
    imshowpair(edgeBeore, edgeAfter,'montage');
    title('Before - After using Edge detection- Canny')
end

% ************************************************************************************
% using imdilate in order to close the componenets
% ************************************************************************************
SE = strel('rectangle',[2,2]); %here should be changed if the ROIs are too big
IM1 = imdilate(edgeAfter,SE);
BW2= imfill(IM1,'holes');
IM2 = imdilate(edgeBeore,SE);
BW1= imfill(IM2,'holes');

if(displayFig)
    figure
    imagesc(BW1);
    title('Before filled ROIs')
    figure
    imagesc(BW2);
    title('After filled ROIs')
end
% ************************************************************************************
% CC is applied again in order to find more accurte connected
% components(after filling)
% ************************************************************************************

CCBefore = bwconncomp(BW1);
CCAfter =  bwconncomp(BW2);

[CCBefore] = deleteUnder20Pix(CCBefore);
[CCAfter] = deleteUnder20Pix(CCAfter);

end
