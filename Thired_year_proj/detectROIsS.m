
function [CC] = detectROIsS(imageSeqStd, displayFig)

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

% sensitivity = 0.100; %much more signals
% edgeAfter = edge(imageSeqAfterStd, 'Canny',sensitivity);

sensitivity = 0.250;
edgeDec = edge(imageSeqStd, 'Canny',sensitivity);


% ************************************************************************************
% using imdilate in order to close the componenets
% ************************************************************************************
SE = strel('rectangle',[2,2]); %here should be changed if the ROIs are too big
IM2 = imdilate(edgeDec,SE);
BW= imfill(IM2,'holes');

if(displayFig)
    figure
    imagesc(edgeDec);
    title('Edge detection- Canny')
    figure
    imagesc(BW);
    title('Filled ROIs')
end
% ************************************************************************************
% CC is applied again in order to find more accurte connected
% components(after filling)
% ************************************************************************************

CC = bwconncomp(BW);
[CC] = deleteUnder20Pix(CC);

