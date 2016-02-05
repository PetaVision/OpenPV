function writeConfidenceTable(data,minrecon,maxrecon,mask,file)
% writeConfidenceTable(data,minrecon,maxrecon,file)

if ~exist('data','var') || ~ismatrix(data) || ~isnumeric(data)
    error('writeConfidenceData:dataNotMatrix', 'writeConfidenceTable error: first argument (data) is not 2-D array');
end%if

if ~exist('minrecon','var') || ~isscalar(minrecon) || ~isnumeric(minrecon)
    error('writeConfidenceData:minreconNotScalar', 'writeConfidenceTable error: second argument (minrecon) must be a scalar.');
end%if

if ~exist('maxrecon','var') || ~isscalar(maxrecon) || ~isnumeric(maxrecon)
    error('writeConfidenceData:maxreconNotScalar', 'writeConfidenceTable error: third argument (maxrecon) must be a scalar.');
end%if

if exist('mask','var') && ~isempty(mask) && (~isnumeric(mask) || ~isvector(mask) || numel(mask)~=size(data,2)) 
    error('writeConfidenceData:badMask', 'writeConfidenceTable error: fourth argument (mask) must be a vector whose length is the number of columns of data (%d)', size(data, 2));
end%if
mask = double(mask(:)'~=0); % makes a row vector of only ones and zeroes.

if ~exist('file','var') || ~isvector(file) || size(file,1)~=1 || ~isequal(class(file), 'char')
    error('writeConfidenceData:fileNotString', 'writeConfidenceTable error: fifth argument (file) is not a string');
end%if

if maxrecon <= minrecon
    error('writeConfidenceData:badInterval', 'writeConfidenceTable error: maxrecon (%f) must be greater than minrecon (%f)', maxrecon, minrecon);
end%if

[fid,errormsg] = fopen(file, 'w');
if fid < 0
    error('writeConfidenceData:badFileOpen', 'writeConfidenceTable error: unable to open file "%s" for writing: %s.', file, errormsg);
end%fid

maskdata = bsxfun(@times, data, mask);

hdrcount = fwrite(fid, 'convTabl', 'uchar');
dimcount = fwrite(fid, [size(data,1) size(data,2)], 'int32');
rvalcount = fwrite(fid, single([minrecon, maxrecon]), 'float');
datacount = fwrite(fid, maskdata(:), 'float');

fclose(fid);

if (hdrcount != 8) || (dimcount != 2) || (rvalcount != 2) || (datacount != numel(data))
    error('writeConfidenceData:badFileWrite', 'writeConfidenceTable error: error writing "%s"', file);
end%if
