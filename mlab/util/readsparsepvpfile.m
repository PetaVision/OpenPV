function [data,hdr] = readsparsepvpfile(filename,progressperiod,last_frame)
% Usage:[data,hdr] = readpvpfile(filename,progressperiod,last_frame)
%
% filename is a pvp sparse file type
% progressperiod is an optional integer argument.  A message is printed
%     to the screen every progressperiod frames.
% last_frame is the index of the last frame to read.  Default is all frames.
%
% data is an M-by-N sparse array containing the data, where
%     M is the number of frames in the .pvp file, and
%     N is the layer size (nx*ny*nf).
%     The ordering of each column is the same as in PetaVision: the
%     feature index varies fastest, then the x-coordinate, and finally
%     the y-coordinate varies slowest.
% hdr is a struct containing the information in the file's header

  filedata = dir(filename);
  if length(filedata) ~= 1
    error('readpvpfile:notonefile',...
          'Path %s should expand to exactly one file; in this case there are %d',...
          filename,length(filedata));
  end%if

  if filedata(1).bytes < 1
    error('readpvpfile:fileempty',...
          'File %s is empty',filename);
  end%if filedata(1).bytes

  fid = fopen(filename);
  if fid<0
    error('readpvpfile:badfilename','readpvpfile:Unable to open %s',filename);
  end%if

  errorident = '';
  errorstring = '';

  hdr = readpvpheader(fid);

  switch hdr.filetype
    case 6 % PVP_ACT_SPARSEVALUES_FILE_TYPE
      numframes = hdr.nbands;
				% framesize is variable
    otherwise
      errorident = 'readsparsepvpfile:badfiletype';
      errorstring = sprintf('readsparsepvpfile:File %s is not a sparse pvp file type %d',filename,hdr.filetype);
  end
  %% last_frames input argument overrides value of numframes
  if (exist('last_frame','var') && ~isempty(last_frame))
      lastframe = min(last_frame, numframes);
  else
    lastframe = numframes;
  end%if
  tot_frames = lastframe;

  if isempty(errorstring)
    data = sparse(tot_frames,hdr.nf*hdr.nx*hdr.ny);
    for f=1:lastframe
      data_tmp = struct('time',0,'values',[]);
      data_tmp.time = fread(fid,1,'float64');
      numactive = fread(fid,1,'uint32');
      data_tmp.values = fread(fid,[2,numactive],'uint32')';
      if (numactive>0)
        data_tmp.values(:,2) = typecast(uint32(data_tmp.values(:,2)),'single');
      end
      if exist('progressperiod','var')
        if ~mod(f,progressperiod)
          fprintf(1,'File %s: frame %d of %d\n',filename, f, lastframe);
          if exist('fflush')
            fflush(1);
          end%if
        end%if
      end%if
      data(f,1+data_tmp.values(:,1)) = data_tmp.values(:,2)';
    end%for %% f=1:lastframe
  end%if

  fclose(fid);

  if ~isempty(errorident)
    error(errorident,errorstring);
  end%if
