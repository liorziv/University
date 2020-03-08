function [yhat] = predict(X,H)
%% given a vector of peak
% centers (X) and peak heights (H) calculates the expected signal (yhat
% - a vector of length 50) in the genomic region. If one of the values of
% H is negative return a vector of NaN values.
    len = length(X);
    yhat = 0;
    psList = zeros(len, 50);
    for i = 1:len
        psList(i,:) = F(X(i));
        if(H(i) < 0)
            yhat =  NaN(1, 50);
            return;
        end
        psList(i,:) = (psList(i,:)*H(i))/max(psList(i,:));
        yhat = yhat + psList(i,:);
    end
end