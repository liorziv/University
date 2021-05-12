# Peak Fitting

**F** - given a peak center (pc), returns a vector of values of length 50 (ps) which represents the expected shape of the ChIP-seq for a single binding event.

**predict** - given a vector of peak centers (X) and peak heights (H) calculates the expected signal (yhat a vector of length 50) in the genomic region. If one of the values of H is negative return a vector of NaN values.

**optimize** - given the data (Y - a vector of length 50) and the expected number of
peaks in the data (k)finds the vectors X;H that were most likely to
generate the data. The function  return the root mean squared
error (RMSE) of the result.

**optimizeWrapper** - wrapper function that runs the optimization N times, records
the RMSE value of each run and chooses the best set of parameters.

**optimizeX** - given the data (Y - a vector of length 50) and the expected number of
peaks in the data (k) finds the vector X that were most likely to
generate the data and calculates H by it. The function  return the root mean squared
error (RMSE) of the result.

**optimizeXWrapper** - wrapper function that runs the optimization for X N times, records
the RMSE value of each run and chooses the best set of parameters.

![Answers](Answers.pdf)