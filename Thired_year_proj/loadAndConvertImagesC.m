 
function [imageSeqBeforeStd, imageSeqAfterStd, imageSeqBefore, imageSeqAfter, numOfFrames] =  loadAndConvertImagesC(workingDirectoryBefore, workingDirectoryAfter)

% ************************************************************************************
% load the tiff images of the two given time frames
% convert them to STD images
% ************************************************************************************
%% Input :
% workingDirectoryBefore -  is the path to the before exposure images
% workingDirectoryAfter -  is the path to the after exposure images
%% Output :
% imageSeqBeforeStd - an std image of the before sequence
% imageSeqAfterStd - an std image of the after sequence
% imageSeqBefore - sequence of all the before images
% imageSeqAfter - sequence of all the After images
% numOfFrames - the total number of images in both of the sequences

%% load image sequence
filesBefore = dir([workingDirectoryBefore '*.tif']);
imageSequenceInfoBefore = imfinfo([workingDirectoryBefore filesBefore(1).name]);
imageWidth = imageSequenceInfoBefore(1).Width;
imageHeight = imageSequenceInfoBefore(1).Height;
filesAfter = dir([workingDirectoryAfter '*.tif']);
numOfFrames = length(filesAfter);
imageSeqBefore = zeros([imageHeight,imageWidth,numOfFrames],'uint16');
imageSeqAfter = zeros([imageHeight,imageWidth,numOfFrames],'uint16');


for frame = 1:numOfFrames
     imageSeqBefore(:,:,frame) = imread([workingDirectoryBefore filesBefore(frame).name]);
     imageSeqAfter(:,:,frame) = imread([workingDirectoryAfter filesAfter(frame).name]);
end


%% convert the images into STD image
imageSeqBeforeStd = zeros([imageHeight,imageWidth],'uint16');
imageSeqAfterStd = zeros([imageHeight,imageWidth],'uint16');

for i = 1:imageHeight
    for j = 1 : imageWidth 
        
        tmp = squeeze(double(imageSeqBefore(i, j, 1 : numOfFrames)));
        imageSeqBeforeStd(i, j) = std(tmp);
        tmp = squeeze(double(imageSeqAfter(i, j, 1 : numOfFrames)));
        imageSeqAfterStd(i, j) = std(tmp);
    end
end

