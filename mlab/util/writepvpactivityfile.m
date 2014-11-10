function writepvpactivityfile(filename, data)
   %  writepvpactivityfile.m
   %    Dylan Paiton
   % 
   % Usage: writepvpactivityfile(filename, data)
   % filename is the pvp file to be created.  If the file already
   % exists it will be clobbered.
   %
   % data is a cell structure representing the non-spiking neural activities, in the same
   % format returned by readpvpfile:
   %
   % data{k} is a structure with two fields, time and values.
   % data{k}.time is the timestamp of frame k.
   % data{k}.values is a double array.
   % data{k}.values(x,y,f) is the activity of the neuron at location (x,y,f)
   
   if ~ischar(filename) || ~isvector(filename) || size(filename,1)~=1
       error('writepvpactivityfile:filenamenotstring', 'filename must be a string');
   end%if
   
   if ~iscell(data)
       error('writepvpactivityfile:datanotcell', 'data must be a cell array, one element per frame');
   end%if
   
   if ~isvector(data)
       error('writepvpactivityfile:datanotcellvector', 'data cell array must be a vector; either number of rows or number of columns must be one');
   end%if
   
   if isempty(data)
       error('writepvpactivityfile:dataempty', 'data must have at least one frame');
   end%if
   
   fid = fopen(filename, 'w');
   if fid < 0
       error('writepvpactivityfile:fopenerror', 'unable to open %s', filename);
   end%if
   
   errorpresent = 0;
   hdr = zeros(20,1);
   [nx ny nf] = size(data{1}.values);
   hdr(1)  = 80;       % Number of bytes in header
   hdr(2)  = 20;       % Number of 4-byte values in header
   hdr(3)  = 4;        % File type for non-sparse activity pvp files 
   hdr(4)  = nx;       % nx
   hdr(5)  = ny;       % ny
   hdr(6)  = nf;       % nf
   hdr(7)  = 1;        % Number of records
   hdr(8)  = nx*ny*nf; % Record size, the size of the layer in bytes 
   hdr(9)  = 4;        % Data size: floats are four bytes
   hdr(10) = 3;        % Types are 1=byte, 2=int, 3=float.  Type 1 is for compressed weights; type 2 isn't used for activity files
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

   for frameno=1:length(data)   % allows either row vector or column vector.  isvector(data) was verified above
       fwrite(fid,data{frameno}.time,'double');
       fwrite(fid,permute(data{frameno}.values(:,:,:),[3 1 2]),'single');
   end%for
   
   fclose(fid); clear fid;
   
   if errorpresent
       error(msgid, errmsg);
   end%if
   
end%function 
