***********************************
The ROIProcessor pipeline input : 
***********************************
workingDirectoryBefore - the before time frame tiff images path
usually used to analyze before the exposure time frame, but can also use any other time frame

workingDirectoryAfter -  the after time frame tiff images path
usually used to analyze after the exposure time frame (10-60 minutes), but can also use any other time frame

compFlag - true runs a comparison between the two time frames, false only analyze one time frame (should have the same path to before and after directories)

dispEdgeDec - if true displays some images from the edge detection stage
Edge detection images - indicates the marked ROIs (when comparing two time frames -  montage of edge detection)
Filled edge detection images - indicates the pixels the pipeline will extract information from

dispTranslation - if false displays some images from the translation stage(relevant only for when comparing)
Always displaying (in compersion mode) the cross correlation outcome - before according to after and after accoriding to before the white is overlapping points
optional - to see filled ROI images with shared ROIs, excluded ROIs(out of one time frame coordinates), exclusive ROIs (only in one of the time frames)

runNum - the run number, used as folder name for the output files (needs to be different if you run in parallel)

***********************************
The two run options :
***********************************

1.Single time frame (all scripts names end with S - single)
 
Run as - ROIProcessor(path, path, false, true/false, false, runNum as a string - '1' )

output files - 
totalROIMeanNormelized -  contain normelized pixels mean per ROI for each image in the time frame (7200 frames -> 7200 means)


2. Comparing two different time frames (all scripts names end with C - comparison)

Run as - ROIProcessor(path1, path2, true, true/false, true/false, runNum as a string - '1' )

output files - 
All the output files contain normalized pixels mean per ROI for each image in the time frame (7200 frames -> 7200 means)

sharedMeanAfterPerFrameNormelized/ sharedMeanBeforePerFrameNormelized - common ROIs 

existOnlyBeforeMeanPerFrameNormelized - ROIs which exist only before
addedToAfterFromBeforeMeanPerFrameNormelized - ROIs which appear only in the before time frame and were added to the after time frame (so we can compare)

existOnlyAfterMeanPerFrameNormelized - ROIs which exist only after
addedToBeforeFromAfterMeanPerFrameNormelized - ROIs which appear only in the after time frame and were added to the before time frame (so we can compare)
