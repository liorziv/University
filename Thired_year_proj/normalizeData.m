% Normelize a given data by the common, each data is an ROI(7200X1)
function [normalizedData] = normalizeData(data) 
normalizedData = data;
dataSize = size(data);
for i = 1:dataSize(2)
    histStruct = histogram(data(:,i));
    [~, maxIdx] = max(histStruct.Values);
    common = histStruct.BinEdges(maxIdx);
    tmp = ((normalizedData(:,i) - common)/common)*100;
    tmp(tmp == -100)=0;
    normalizedData(:,i) = tmp;
   
  
end

