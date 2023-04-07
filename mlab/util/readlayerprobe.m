function [probeData, numNeurons] = readlayerprobe(directory, probeName, batchElement)
%% probeData = readlayerprobe(dirName, probeName, batchElement)
% dirName is the directory where the probe's output files are located
% probeName is the name of the probe
% batchElement is the batch element or list of batch elements to read
%   The default batchElement is 0.
%
% probeData is a structure containing two fields, 'time' and 'values'
%   'time' is a column vector consisting of the timestamps in the probe data.
%   'values' is an N-by-B matrix where N is the length of the 'time' field
%   and B is the length of the batchElement input argument.
% numNeurons is an integer giving the number of neurons in the layer.

if ~exist('hasHeader', 'var')
   hasHeader = false;
end%if ~exist(hasHeader)

if ~exist('batchElement', 'var') || isempty(batchElement)
    batchElement = 0;
end%if ~exist(batchElement)

N = numel(batchElement);
for n = 1:N
    filename = [directory filesep probeName "_batchElement_" num2str(batchElement(n)) ".txt"];
    fid = fopen(filename);
    fileData = fscanf(fid, '%f, %d, %d, %f\n', [4, Inf]);
    if n == 1
       timestamps = fileData(1,:)';
       values = zeros(numel(timestamps), N);
       numNeurons = fileData(3,1);
    end%if n==1
    values(:, n) = fileData(4,:)';
end%for n
probeData.time = timestamps;
probeData.values = values;
