function [E,timestamps] = loadColumnEnergyProbe(outputFile)
% [E,timestamps] = loadColumnEnergyProbe(outputFile)
%
% outputDir is a string giving the path to the file containing the output of a column energy probe.
%
% E is an array.  The number of columns is the batchwidth and the number of rows is the number
% of timestamps.
%
% timestamps is a column vector with the same number of rows as E, that gives the timestamps.

[~,w] = system(['awk -F '','' ''{print $2" "$3" "$4}'' ' outputFile]);
A = sscanf(w,"%f %f %f\n");
A = reshape(A,3,numel(A)/3)';
batches = unique(A(:,2));
nbatches = numel(batches);
assert(isequal(batches, (0:nbatches-1)'));
timestamps = unique(A(:,1));
ntimestamps = numel(timestamps);
assert(size(A,1)==ntimestamps * nbatches);
E = reshape(A(:,3),ntimestamps, nbatches);
