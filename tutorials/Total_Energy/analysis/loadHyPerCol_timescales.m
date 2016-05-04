function [A,t] = loadHyPerCol_timescales(outputDir)
% [A,t] = loadHyPerCol_timescales(outputDir)
%
% outputDir is a string giving a directory containing the HyPerCol_timescales.txt to load
%
% A is a two-column array.  The first column is the values of timeScaleTrue
%    The second is the value of timeScale.
%
% t is a column vector with the same number of rows as A, that gives the timestamps.
timescalePath = [outputDir filesep 'HyPerCol_timescales.txt'];
[~,w] = system(['awk ''/batch/ {print $9","$6}'' ' timescalePath]);
A = sscanf(w,'%f,%f,\n');
A = reshape(A,[2 numel(A)/2])';
[~,w] = system(['awk ''/sim_time/ {print $3}'' ' timescalePath]);
t = sscanf(w,'%f\n');
