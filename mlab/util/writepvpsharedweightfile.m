function writepvpsharedweightfile(filename, data)
   % Usage: writepvpsharedweightfile(filename, data)
   % filename is the pvp file to be created.  If the file already
   % exists it will be clobbered.
   %
   % data is a cell array representing the weights, in the same
   % format returned by readpvpfile:
   %
   % data{k} is a structure with two fields, time and values.
   % data{k}.time is the timestamp of frame k.
   % data{k}.values is a cell structure, one element per arbor.
   %     (Non-shared weight pvp files store their patches in a 3-d cell array,
   %      the first element for x-location of the patch, the second for
   %      y-location, and the third for arbor number.
   %      readpvpfile applied to a shared-weight file therefore returns a
   %      cell array of size (1,1,numArbors).  writepvpsharedweightfile
   %      just uses numel of the cell array to get numArbors.)
   % data{k}.values{j} is arbor number j, a 4-dimensional array.
   % data{k}.values{j}(x,y,f,k) is the weight at location (x,y,f)
   % of data patch k.
   
   if ~ischar(filename) || ~isvector(filename) || size(filename,1)~=1
       error('writepvpsharedweightfile:filenamenotstring', 'filename must be a string');
   end%if
   
   if ~iscell(data)
       error('writepvpsharedweightfile:datanotcell', 'data must be a cell array, one element per frame');
   end%if
   
   if ~isvector(data)
       error('writepvpsharedweightfile:datanotcellvector', 'data cell array must be a vector; either number of rows or number of columns must be one');
   end%if
   
   fid = fopen(filename, 'w');
   if fid < 0
       error('writepvpsharedweightfile:fopenerror', 'unable to open %s', filename);
   end%if
   
   errorpresent = 0;
   hdr = zeros(26,1);
   for frameno=1:length(data)   % allows either row vector or column vector.  isvector(data) was verified above
       numarbors = numel(data{frameno}.values);
       arbor1size = size(data{frameno}.values{1});
       wmax = max(data{frameno}.values{1}(:));
       wmin = min(data{frameno}.values{1}(:));
       for arbor=2:numarbors
           arborsize = size(data{frameno}.values{arbor});
           if ~isequal(arborsize,arbor1size)
               msgid='arborsunequalsizes';
               errmsg=sprintf('Frame %d has different sizes for arbors 1 and %d', frameno, arbor);
               errorpresent=1;
               break;
           end%if
           wmaxarbor = max(data{frameno}.values{arbor}(:));
           wmax = max(wmax,wmaxarbor);
           wminarbor = min(data{frameno}.values{arbor}(:));
           wmin = min(wmin,wminarbor);
       end%for
       if errorpresent, break; end%if
       if numel(arbor1size<4)
           arbor1size=[arbor1size ones(1,4-numel(arbor1size))];
       end%if
       nxp = arbor1size(1);
       nyp = arbor1size(2);
       nfp = arbor1size(3);
       numpatches = arbor1size(4);
       % Each frame has its own header
       hdr(1) = 104; % Number of bytes in header
       hdr(2) = 26;  % Number of 4-byte values in header
       hdr(3) = 5;   % File type for shared-weight connections
       hdr(4) = 1;   % Presynaptic nx, not used by weights
       hdr(5) = 1;   % Presynaptic ny, not used by weights
       hdr(6) = numpatches; % Presynaptic nf, assuming it's the same as the number of weight patches
       hdr(7) = numarbors;   % Number of records, for weight files one arbor is a record.
       hdr(8) = numpatches*(8+4*nxp*nyp*nfp);   % Record size, the data for the arbor is preceded by nx(2 bytes), ny(2 bytes) and offset(4 bytes)
       hdr(9) = 4;   % Data size: floats are four bytes
       hdr(10) = 3;  % Types are 1=byte, 2=int, 3=float.  Type 1 is for compressed weights; type 2 isn't used for weight files
       hdr(11) = 1;  % Number of processes in x-direction; no longer used
       hdr(12) = 1;  % Number of processes in y-direction; no longer used
       hdr(13) = 1;  % Presynaptic nxGlobal, not used by weights
       hdr(14) = 1;  % Presynaptic nyGlobal, not used by weights
       hdr(15) = 0;  % kx0, no longer used
       hdr(16) = 0;  % ky0, no longer used
       hdr(17) = 0;  % Presynaptic nb, not used by weights
       hdr(18) = numel(data{frameno}.values); % number of arbors
       hdr(19:20) = typecast(double(data{frameno}.time),'uint32'); % timestamp
       % hdr(21:26) is for extra header values used by weights
       hdr(21) = nxp;% Size of full patches in the x-direction
       hdr(22) = nyp;% Size of full patches in the y-direction
       hdr(23) = nfp;% Number of features in the connection
       hdr(24) = typecast(single(wmin),'uint32'); % Minimum of all weights over all arbors in this frame.
       hdr(25) = typecast(single(wmax),'uint32'); % Maximum of all weights over all arbors in this frame.
       hdr(26) = numpatches; % Number of weight patches
       fwrite(fid,hdr,'uint32');
       for arbor=1:numarbors
           for patchno=1:numpatches
               fwrite(fid,nxp,'uint16');
               fwrite(fid,nyp,'uint16');
               fwrite(fid,0,'uint32'); % Offset for shared weights is always 0.
               fwrite(fid,permute(data{frameno}.values{arbor}(:,:,:,patchno),[3 1 2]),'single');
           end%for
       end%for
   end%for
   
   fclose(fid); clear fid;
   
   if errorpresent
       error(msgid, errmsg);
   end%if
   
   
