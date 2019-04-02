function [data,hdr] = readsparsepvpfile(filename,progressperiod,last_frame,start_frame,skip_frame)
% Usage: [data,hdr] = readsparsepvpfile(filename,progressperiod,last_frame,start_frame,skip_frame)
%
% filename is a pvp sparse file type
% progressperiod is an optional integer argument.  A message is printed
%     to the screen every progressperiod frames.
%     
% last_frame is the index of the last frame to read.  Default is all frames.
% start_frame is the index of the first frame to read. Default is skip_frame.
% skip_frame specifies to read only every nth frame. Default is 1 (all frames).
%     If start_frame is m and skip_frame is n, the frames read are m, m+n, m+2n, m+3n, etc.
%
% data is an M-by-N sparse metrix where M is the number of neurons in the layer (nf*nx*ny)
%     and N is the number of frames read.
%     The ordering of each column is the same as in PetaVision (except that here it is 1-indexed):
%     the feature index varies fastest, then the x-coordinate, and finally
%     the y-coordinate varies slowest.
% hdr is a struct containing the information in the file's header

  filedata = dir(filename);
  if length(filedata) ~= 1
    error('readsparsepvpfile:notonefile',...
          'Path %s should expand to exactly one file; in this case there are %d',...
          filename,length(filedata));
  end%if

  if filedata(1).bytes < 1
    error('readsparsepvpfile:fileempty',...
          'File %s is empty',filename);
  end%if filedata(1).bytes

  fid = fopen(filename);
  if fid<0
    error('readsparsepvpfile:badfilename','readsparsepvpfile:Unable to open %s',filename);
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

  if (exist('skip_frame','var') && ~isempty(skip_frame))
    skipframe = min(skip_frame, last_frame);
  else
    skipframe = 1;
  end%if

  if (exist('start_frame','var') && ~isempty(start_frame))
    startframe = start_frame;
  else
    startframe = skipframe;
  end%if
  if startframe < 1
      errorident = 'readsparsepvpfile:startnonpositive';
      errorstring = sprintf('readsparsepvpfile:startframe is %d but must be positive',filename,startframe);
  end%if
  if startframe > lastframe
      errorident = 'readsparsepvpfile:startbeforeend';
      errorstring = sprintf('readsparsepvpfile:File %s has %d frames but startframe is %d',filename,lastframe,startframe);
  end%if

  cum_active = 0;
  if isempty(errorstring)
    
    for f=1:lastframe
      data_tmp_time   = fread(fid,1,'float64');
      numactive       = fread(fid,1,'uint32');
      data_tmp_values = fread(fid,[2,numactive],'uint32')';
      if f < startframe || mod(f-startframe,skipframe)~=0
        continue;
      end%if      
      cum_active      = cum_active + numactive;
      if exist('progressperiod','var')
        if ~mod(f,progressperiod)
          fprintf(1,'File %s: frame %d of %d\n',filename, f, lastframe);
          if exist('fflush')
            fflush(1);
          end%if
        end%if
      end%if
    end%for %% f=1:lastframe

    frewind(fid)
    hdr = readpvpheader(fid);
    tot_active = cum_active;
    numframesread = floor((lastframe-startframe)/skipframe)+1;
    data = spalloc(hdr.nf*hdr.nx*hdr.ny, numframesread, tot_active);

    for f=1:lastframe
      data_tmp_time   = fread(fid,1,'float64');
      numactive       = fread(fid,1,'uint32');
      data_tmp_values = fread(fid,[2,numactive],'uint32')';
      if f < startframe || mod(f-startframe,skipframe)~=0
        continue;
      end%if      
      if (numactive>0)
	data_tmp_ndx   = data_tmp_values(:,1)+1;
        data_tmp_coef  = typecast(uint32(data_tmp_values(:,2)),'single');
	data_tmp       = sparse(data_tmp_ndx(:), repmat(1, numactive,1), data_tmp_coef(:), hdr.nf*hdr.nx*hdr.ny, 1);
	data(:, 1+(f-startframe)/skipframe)      = data_tmp;
      end
      if exist('progressperiod','var')
        if ~mod(f,progressperiod)
          fprintf(1,'File %s: frame %d of %d\n',filename, f, lastframe);
          if exist('fflush')
            fflush(1);
          end%if
        end%if
      end%if
    end%for %% f=1:lastframe

  end%if

  fclose(fid);
  
  if ~isempty(errorident)
    error(errorident,errorstring);
  end%if
