 
function [imageSeqStd, imageSeq, numOfFrames] =  loadAndConvertImagesS(workingDirectory)

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
files = dir([workingDirectory '*.tif']);
imageSequenceInfo = imfinfo([workingDirectory files(1).name]);
imageWidth = imageSequenceInfo(1).Width;
imageHeight = imageSequenceInfo(1).Height;
numOfFrames = length(files);
imageSeq = zeros([imageHeight,imageWidth,numOfFrames],'uint16');
imageSeqAfter = zeros([imageHeight,imageWidth,numOfFrames],'uint16');


for frame = 1:numOfFrames
     imageSeq(:,:,frame) = imread([workingDirectory files(frame).name]);
end


%% convert the images into STD image
imageSeqStd = zeros([imageHeight,imageWidth],'uint16');

for i = 1:imageHeight
    for j = 1 : imageWidth 
        
        tmp = squeeze(double(imageSeq(i, j, 1 : numOfFrames)));
        imageSeqStd(i, j) = std(tmp);
      
    end
end

