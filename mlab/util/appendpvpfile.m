%
% appendpvpfile(inputFileNameA, inputFileNameB, outputFileName, progressInterval)
% Austin Thresher
%
% Appends file B to the end of file A and writes the result to a new file.
% The operation is performed one frame at a time to avoid memory issues.
% Only works on filetypes 4 and 6 (Activity and Sparse Values)
%
% progressInterval is an integer argument; if positive, it outputs a progress
% message after processing [progressInterval] frames. Default is 100.

function appendpvpfile(inputFileNameA, inputFileNameB, outputFileName, progressInterval)
   isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; % for compatibility
   if nargin < 4, progressInterval = 100; end

   [data, headerA] = readpvpfile(inputFileNameA, 0, 1, 1);
   if headerA.filetype ~= 4 && headerA.filetype ~= 6
      error('Unsupported filetype. Must be type 4 or 6.');
   end
   [data, headerB] = readpvpfile(inputFileNameB, 0, 1, 1);
   if headerA.filetype ~= headerB.filetype
      error('Both files must be the same filetype.');
   end
   if headerA.nx ~= headerB.nx || headerA.ny ~= headerB.ny || headerA.nf ~= headerB.nf
      error('Both files must have the same dimensions (nx, ny, nf).');
   end

   frames = headerA.nbands + headerB.nbands;
   nx     = headerA.nx;
   ny     = headerA.ny;
   nf     = headerA.nf;

   fprintf('Opening %s for writing...\n', outputFileName); if isOctave, fflush(stdout); end
   fid = fopen(outputFileName, 'w');
   if fid < 0
       error('Unable to open %s', outputFileName);
   end

   fprintf('Starting.\n'); if isOctave, fflush(stdout); end

   % Write the correct header for the given filetype

   if headerA.filetype == 4
  
      % Header output copied from writepvpactivityfile.m with minor name changes

      hdr = zeros(20,1);
      hdr(1)  = 80;       % Number of bytes in header
      hdr(2)  = 20;       % Number of 4-byte values in header
      hdr(3)  = 4;        % File type for non-sparse activity pvp files 
      hdr(4)  = nx;       % nx
      hdr(5)  = ny;       % ny
      hdr(6)  = nf;       % nf
      hdr(7)  = 1;        % Number of records
      hdr(8)  = nx*ny*nf; % Record size, the size of the layer in bytes 
      hdr(9)  = 4;        % Data size: floats are four bytes
      hdr(10) = 3;        
      hdr(11) = 1;        % Number of processes in x-direction; no longer used
      hdr(12) = 1;        % Number of processes in y-direction; no longer used
      hdr(13) = nx;       % Presynaptic nxGlobal
      hdr(14) = ny;       % Presynaptic nyGlobal
      hdr(15) = 0;        % kx0, no longer used
      hdr(16) = 0;        % ky0, no longer used
      hdr(17) = 0;        % Presynaptic nb, not relevant for activity files
      hdr(18) = frames;   % number of frames 
      hdr(19:20) = typecast(double(0),'uint32'); % timestamp
      fwrite(fid,hdr,'uint32');

      % End copied section

   else % filetype == 6

      % Header output copied from writepvpsparsevalues.m with minor name changes

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
      hdr(18) = frames;   % number of frames 
      hdr(19:20) = typecast(double(0),'uint32'); % timestamp
      fwrite(fid,hdr,'uint32');

      % End copied section
   end

   for index=1:headerA.nbands
      if ~mod(index, progressInterval)
         fprintf('File A: Frame %d of %d\n', index, headerA.nbands); if isOctave, fflush(stdout); end
      end
      
      % Read one frame of data
      data = readpvpfile(inputFileNameA, 0, index, index);

      % Write result to file
      fwrite(fid,data{1}.time,'double');
      if headerA.filetype == 4
         fwrite(fid,permute(data{1}.values(:,:,:),[3 1 2]),'single');
      else
         count = size(data{1}.values,1);
         fwrite(fid,count,'uint32');
         if count > 0
            fwrite(fid, data{1}.values(:, 2), 'float32', 4);
            fseek(fid, -count * 8 - 4, 'cof');
            fwrite(fid, data{1}.values(:, 1), 'uint32', 4);
            fseek(fid, 4, 'cof');
         end
      end
      clear data;
   end
   for index=1:headerB.nbands
      if ~mod(index, progressInterval)
         fprintf('File B: Frame %d of %d\n', index, headerB.nbands); fflush(stdout);
      end
      
      % Read one frame of data
      data = readpvpfile(inputFileNameB, 0, index, index);

      % Write result to file
      fwrite(fid,data{1}.time,'double');
      if headerA.filetype == 4
         fwrite(fid,permute(data{1}.values(:,:,:),[3 1 2]),'single');
      else
         count = size(data{1}.values,1);
         fwrite(fid,count,'uint32');
         if count > 0
            fwrite(fid, data{1}.values(:, 2), 'float32', 4);
            fseek(fid, -count * 8 - 4, 'cof');
            fwrite(fid, data{1}.values(:, 1), 'uint32', 4);
            fseek(fid, 4, 'cof');
         end
      end
      clear data;
   end

   fclose(fid); clear fid;
   fprintf('Finished writing %s.\n', outputFileName); fflush(stdout);

end
