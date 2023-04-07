function probeData = readenergyprobe(directory, probeName, batchElement)
%% probeData = readenergyprobe(directory, probeName, batchElement)
% directory is the directory where the probe's output files are located
% probeName is the name of the probe
% batchElement is the batch element or list of batch elements to read
%   The default batchElement is 0.
%
% probeData is a structure containing two fields, 'time' and 'values'
% 'time' is a column vector consisting of the timestamps in the probe data.
% 'values' is an N-by-B matrix where N is the length of the 'time' field
% and B is the length of the batchElement input argument.

if ~exist('batchElement', 'var') || isempty(batchElement)
    batchElement = 0;
end%if ~exist(batchElement)

N = numel(batchElement);
for n = 1:N
    filename = [directory filesep probeName "_batchElement_" num2str(batchElement(n)) ".txt"];
    fid = fopen(filename);
    headerline = fgetl(fid);
    fileData = fscanf(fid, '%f, %d, %f\n', [3, Inf]);
    if n == 1
       timestamps = fileData(1,:)';
       energy = zeros(numel(timestamps), N);
    end%if n==1
    energy(:, n) = fileData(3,:)';
end%for n
probeData.time = timestamps;
probeData.values = energy;
