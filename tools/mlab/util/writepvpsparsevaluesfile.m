function writepvpsparsevaluesfile(filename, data, nx, ny, nf, show_progress = false)
   %  writepvpsparsevaluesfile.m
   %    Pete Schultz
   % 
   % Usage: writepvpsparsevaluesfile(filename, data, nx, ny, nf)
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
   % data{k}.values is an array of two columns giving the indices and values of active neurons.
   % If data{k}.values(j,1)==n and data{k}.values(j,2)==x, then neuron n has activity x.
   %     The neuron with zero-indexed coordinates (x,y,f) in a layer with
   %     dimensions (nx,ny,nf) has index y*(nx*nf)+x*(nf)+f
   
   if nargin < 5 || nargin > 6
       error('writepvpsparsevaluesfile:missingargs', 'writepvpsparsevaluesfile requires 5 or 6 arguments');
   end%if
   
   if ~ischar(filename) || ~isvector(filename) || size(filename,1)~=1
       error('writepvpsparsevaluesfile:filenamenotstring', 'filename must be a string');
   end%if
   
   if ~iscell(data)
       error('writepvpsparsevaluesfile:datanotcell', 'data must be a cell array, one element per frame');
   end%if
   
   if ~isvector(data)
       error('writepvpsparsevaluesfile:datanotcellvector', 'data cell array must be a vector; either number of rows or number of columns must be one');
   end%if
   
   if isempty(data)
       error('writepvpsparsevaluesfile:dataempty', 'data must have at least one frame');
   end%if

   numframes = length(data);
   numneurons = nx*ny*nf;
   
   for n=1:numframes
       if ~isempty(data{n}.values)
           if ismatrix(data{n}.values)
               sz = size(data{n}.values);
               if numel(sz)~=2 || sz(2)~=2
                   error('writepvpsparsevalues:badnumcols', 'data{%d}.values must have two columns', n);
               end%if
           else
               error('writepvpsparsevalues:nonmatrix', 'data{%d}.values must be a matrix with two columns', n);
           end%if
           if ~isnumeric(data{n}.values)
               error('writepvpsparsevalues:nonmatrix', 'data{%d}.values is not a numeric data type', n);
           end%if
           if ~isequal(data{n}.values(:,1), round(data{n}.values(:,1)))
               error('writepvpsparsevalues:noninteger', 'data{%d}.values first column is not integral', n);
           end%if
           outofbounds = data{n}.values(:,1)<0 | data{n}.values(:,1)>=numneurons;
           if any(outofbounds)
               badindex = find(outofbounds, 1, 'first');
               badvalue = data{n}.values(badindex);
               error('writepvpsparsevalues:outofbounds', 'data{%d}.values first column must have values between 0 and nx*ny*nf-1=%d (first out-of-bounds value is entry (%d,1), value %d)', n, numneurons-1, badindex, badvalue);
           end%if

       end%if
   end%for
   
   fid = fopen(filename, 'w');
   if fid < 0
       error('writepvpsparsevaluesfile:fopenerror', 'unable to open %s', filename);
   end%if
   
   errorpresent = 0;
   hdr = zeros(20,1);
   hdr(1)  = 80;       % Number of bytes in header
   hdr(2)  = 20;       % Number of 4-byte values in header
   hdr(3)  = 6;        % File type for sparsevalues pvp files 
   hdr(4)  = nx;       % nx
   hdr(5)  = ny;       % ny
   hdr(6)  = nf;       % nf
   hdr(7)  = 1;        % Number of records
   hdr(8)  = 0;        % Record size, not used in sparse activity since record size is not fixed
   hdr(9)  = 8;        % Data size: int and float are four bytes each
   hdr(10) = 4;        % Types are 1=byte, 2=int, 3=float, 4=location-value pair.
   hdr(11) = 1;        % Number of processes in x-direction; no longer used
   hdr(12) = 1;        % Number of processes in y-direction; no longer used
   hdr(13) = nx;       % Presynaptic nxGlobal
   hdr(14) = ny;       % Presynaptic nyGlobal
   hdr(15) = 0;        % kx0, no longer used
   hdr(16) = 0;        % ky0, no longer used
   hdr(17) = 0;        % Presynaptic nb, not relevant for activity files
   hdr(18) = length(data); % number of frames 
   hdr(19:20) = typecast(double(data{1}.time),'uint32'); % timestamp
   fwrite(fid,hdr,'uint32');

   progress_timer = length(data) / 100; % display progress every 1%
   progress_amount = 0;
   
   for frameno=1:length(data)   % allows either row vector or column vector.  isvector(data) was verified above
       fwrite(fid,data{frameno}.time,'double');
       count = size(data{frameno}.values,1);
       fwrite(fid,count,'uint32');
       if count > 0
          fwrite(fid, data{frameno}.values(:, 2), 'float32', 4);
          fseek(fid, -count * 8 - 4, SEEK_CUR);
          fwrite(fid, data{frameno}.values(:, 1), 'uint32', 4);
          fseek(fid, 4, SEEK_CUR);
       end%if

       if show_progress
           progress_timer -= 1;
           if progress_timer <= 0
              progress_amount += 1;
              progress_timer = length(data) / 100;
              printf("%d%% ", progress_amount);
              fflush(stdout);
           end
       end
   end%for
   
   fclose(fid); clear fid;
   
   if errorpresent
       error(msgid, errmsg);
   end%if
   
end%function 
