function [ps] = F(pc)
%% given a peak center (pc), returns
% a vector of values of length 50 (ps) which represents the expected
% shape of the ChIP-seq for a single binding event.

    X = 1:50;
    ps = normpdf(X,pc,3);
end
