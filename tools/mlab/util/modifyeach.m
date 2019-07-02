%
% modifyeach(inputFileName, outputFileName, callbackFunction, progressInterval)
% Austin Thresher
%
% Applies callbackFunction to each frame in the input pvp file
% and writes the result to outputFileName. The callback function
% must accept an NX x NY x NF size matrix and return the modified
% result. This reads and writes the input and output one frame at
% a time to avoid memory issues.
% Only works on filetype 4 (Activity, non-sparse)
%
% Example usage:
%
% We want to replace NaN and Inf with 0. The function makeFinite
% is defined in makeFinite.m and can be used as a reference for
% writing your own callback functions.
%
%   f = @makeFinite;   
%   modifyEach("fileToClean.pvp", "cleanFile.pvp", f);
%
% Note that Octave will not allow you to pass @makeFinite directly,
% so the preceding line is required.



function modifyeach(inputFileName, outputFileName, callbackFunction, progressInterval = 10)

   [data, header] = readpvpfile(inputFileName, 0, 1, 1);
   if header.filetype != 4
      error('Unsupported filetype. Must be type 4.');
   end

   frames = header.nbands;
   nx     = header.nx;
   ny     = header.ny;
   nf     = header.nf;

   printf('Opening %s for writing...\n', outputFileName); fflush(stdout);
   fid = fopen(outputFileName, 'w');
   if fid < 0
       error('Unable to open %s', outputFileName);
   end
  
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
   hdr(10) = 3;        % Types are 1=byte, 2=int, 3=float.  Type 1 is for compressed weights; type 2 isn't used for activity files
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

   printf('Starting.\n'); fflush(stdout);

   for index=1:frames
      if ~mod(index, progressInterval)
         printf('Frame %d of %d\n', index, frames); fflush(stdout);
      end
      
      % Read one frame of data
      data = readpvpfile(inputFileName, 0, index, index);
      currentValues = data{1}.values;

      % Apply callbackFunction to data
      result = callbackFunction(currentValues);

      % Write result to file
      fwrite(fid,data{1}.time,'double');
      fwrite(fid,permute(result(:,:,:),[3 1 2]),'single');

      clear currentValues;
      clear result;
      clear data;
   end

   fclose(fid); clear fid;
   printf('Finished writing %s.\n', outputFileName); fflush(stdout);

end
