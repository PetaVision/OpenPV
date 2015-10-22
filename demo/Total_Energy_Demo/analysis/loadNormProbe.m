function [normvalues, timestamps] = loadNormProbe(outputFile)
% [normvalues, timestamps] = loadNormProbe(outputFile)
%
% outputFile is a string giving the path to the file containing the output
% of a subclass of AbstractNormProbe
%
% normvalues is an array.  The number of columns is the batchwidth and the
% number of rows is the number of timestamps.
%
% timestamps is a column vector with the same number of rows as normvalues,
% tht gives the timestamps.

if ~ischar(outputFile) || isempty(outputFile) || ~isvector(outputFile) || size(outputFile,1)~=1
    error([mfilename ':notstring'], 'input argument to %s must be a string.', mfilename);
end%if

fid = fopen(outputFile);
if (fid<0)
    error([mfilename ':badfile'], '%s unable to open %s', mfilename, outputFile);
end%if

lin = fgets(fid);

% Count the lines for preallocation.
linenumber = 0;
while ischar(lin)
    linenumber = linenumber+1;    
    lin = fgets(fid);
end%while

rewindstatus = frewind(fid);
if (rewindstatus ~= 0)
    fclose(fid);
    error('loadNormProbe:');
end%if

A = zeros(linenumber, 3);
linenumber = 0;
lin = fgets(fid);
while ischar(lin)
    linenumber = linenumber+1;
    [~,~,~,~,T] = regexp(lin, 't *= *([0-9.Ee-]+) *b *= *([0-9]+).*= *([0-9.Ee-]+)$');
    if numel(T{1})~=3
        fclose(fid);
        error([mfilename ':badformat'], 'line number %d of %s does not have expected format.', linenumber, outputFile);
    end%if
    A(linenumber,1) = sscanf(T{1}{1}, '%f');
    A(linenumber,2) = sscanf(T{1}{2}, '%f');
    A(linenumber,3) = sscanf(T{1}{3}, '%f');
    lin = fgets(fid);
end%while

fclose(fid);

batchlist = unique(A(:,2));
timestamps = unique(A(:,1));
normvalues = nan(batchlist, timestamps);
for k=1:linenumber
    normvalues(timestamps==A(k,1), batchlist==A(k,2)) = A(k,3);
end%for