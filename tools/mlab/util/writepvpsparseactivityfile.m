function writepvpsparseactivityfile(filename, data, nx, ny, nf)
   %  writepvpsparseactivityfile.m
   %    Pete Schultz
   %
   % Note: the sparse-binary format is no longer supported (as of Mar 14, 2017)
   % Instead, use writepvpsparsevaluesfile to write files in the sparse-values
   % format (file-type = 6).
   % 
   % Usage: writepvpsparseactivityfile(filename, data, nx, ny, nf)
   % filename is the pvp file to be created.  If the file already
   % exists it will be clobbered.
   %
   % data is a cell structure representing the non-spiking neural activities, in the same
   % format returned by readpvpfile:
   %
   % nx, ny, nf are the size of the layers.  Needed in the header but not contained in the data
   %
   % data{k} is a structure with two fields, time and values.
   % data{k}.time is the timestamp of frame k.
   % data{k}.values is an integer vector giving the indices of active neurons.
   % If data{k}.values(j)==n, then neuron n is active.
   %     The neuron with zero-indexed coordinates (x,y,f) in a layer with
   %     dimensions (nx,ny,nf) has index y*(nx*nf)+x*(nf)+f
   
   if nargin ~= 5
       error('writepvpsparseactivityfile:missingargs', 'writepvpsparseactivityfile requires 5 arguments');
   end%if
   
   if ~ischar(filename) || ~isvector(filename) || size(filename,1)~=1
       error('writepvpsparseactivityfile:filenamenotstring', 'filename must be a string');
   end%if
   
   if ~iscell(data)
       error('writepvpsparseactivityfile:datanotcell', 'data must be a cell array, one element per frame');
   end%if
   
   if ~isvector(data)
       error('writepvpsparseactivityfile:datanotcellvector', 'data cell array must be a vector; either number of rows or number of columns must be one');
   end%if

   if isempty(data)
       error('writepvpsparseactivityfile:dataempty', 'data must have at least one frame');
   end%if

   numframes = length(data);
   numneurons = nx*ny*nf;

   for n=1:numframes
       if ~isempty(data{n}.values) && (~isvector(data{n}.values) || ~isnumeric(data{n}.values) || ~isequal(data{n}.values, round(data{n}.values)))
           error('writepvpsparseactivity:noninteger', 'data{%d}.values is not a vector of integers', n);
       end%if
       outofbounds = data{n}.values<0 | data{n}.values>=numneurons;
       if any(outofbounds)
          badindex = find(outofbounds, 1, 'first');
          badvalue = data{n}.values(badindex);
          error('writepvpsparseactivity:outofbounds', 'data{%d}.values must have values between 0 and nx*ny*nf-1=%d (first out-of-bounds values is entry %d, value %d)', n, numneurons-1, badindex, badvalue);
       end%if
   end%for
   
   fid = fopen(filename, 'w');
   if fid < 0
       error('writepvpsparseactivityfile:fopenerror', 'unable to open %s', filename);
   end%if
   
   errorpresent = 0;
   hdr = zeros(20,1);
   hdr(1)  = 80;       % Number of bytes in header
   hdr(2)  = 20;       % Number of 4-byte values in header
   hdr(3)  = 2;        % File type for sparse activity pvp files 
   hdr(4)  = nx;       % nx
   hdr(5)  = ny;       % ny
   hdr(6)  = nf;       % nf
   hdr(7)  = 1;        % Number of records
   hdr(8)  = 0;        % Record size, not used in sparse activity since record size is not fixed
   hdr(9)  = 4;        % Data size: ints are four bytes
   hdr(10) = 2;        % Types are 1=byte, 2=int, 3=float.  Active neuron indices are integers
   hdr(11) = 1;        % Number of processes in x-direction; no longer used
   hdr(12) = 1;        % Number of processes in y-direction; no longer used
   hdr(13) = nx;       % Presynaptic nxGlobal
   hdr(14) = ny;       % Presynaptic nyGlobal
   hdr(15) = 0;        % kx0, no longer used
   hdr(16) = 0;        % ky0, no longer used
   hdr(17) = 0;        % Presynaptic nb, not relevant for activity files
   hdr(18) = numframes; % number of frames 
   hdr(19:20) = typecast(double(data{1}.time),'uint32'); % timestamp
   fwrite(fid,hdr,'uint32');
   for frameno=1:numframes   % allows either row vector or column vector.  isvector(data) was verified above
       fwrite(fid,data{frameno}.time,'double');
       count = numel(data{frameno}.values);
       fwrite(fid,count,'uint32');
       fwrite(fid,data{frameno}.values,'uint32');
   end%for
   
   fclose(fid); clear fid;
   
   if errorpresent
       error(msgid, errmsg);
   end%if
   
end%function 
