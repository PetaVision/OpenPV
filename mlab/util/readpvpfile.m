function [data,hdr] = readpvpfile(filename,progressperiod, num_frames, start_frame)
% Usage:[data,hdr] = readpvpfile(filename)
% filename is a pvp file (any type)
% data is a cell array containing the data.
%     In general, data has one element for each time step written.
%     Each element is a struct containing the fields 'time' and 'values'
%     For activities, values is an nx-by-ny-by-nf array.
%     For weights, values is a cell array, each element is an nxp-by-nyp-by-nfp array.

%% start_frame allows the user to only read frames from some starting point
if nargin < 4 || ~exist("start_frame") || isempty(start_frame)
  start_frame = 1;
end

filedata = dir(filename);
if length(filedata) ~= 1
    error('readpvpfile:notonefile',...
        'Path %s should expand to exactly one file; in this case there are %d',...
        filename,length(filedata));
end

if filedata(1).bytes < 1
    error('readpvpfile:fileempty',...
        'File %s is empty',filename);
end%if filedata(1).bytes

fid = fopen(filename);
if fid<0
    error('readpvpfile:badfilename','readpvpfile:Unable to open %s',filename);
end

errorident = '';
errorstring = '';

hdr = readpvpheader(fid);

switch hdr.filetype
    case 1 % PVP_FILE_TYPE
        framesize = hdr.recordsize*hdr.numrecords;
        numframes = (filedata(1).bytes - hdr.headersize)/framesize;
    case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking
        numframes = hdr.nbands;
        % framesize is variable
    case 3 % PVP_WGT_FILE_TYPE % HyPerConns that aren't KernelConns
        framesize = hdr.recordsize*hdr.numrecords+hdr.headersize;
        numframes = filedata(1).bytes/framesize;
    case 4 % PVP_NONSPIKING_ACT_FILE_TYPE
        nxprocs = hdr.nxGlobal/hdr.nx;
        nyprocs = hdr.nyGlobal/hdr.ny;
        framesize = hdr.recordsize*hdr.datasize*nxprocs*nyprocs+8;
        numframes = hdr.nbands;
    case 5 % PVP_KERNEL_FILE_TYPE
        framesize = hdr.recordsize*hdr.nbands+hdr.headersize;
        numframes = filedata(1).bytes/framesize;
    otherwise
        errorident = 'readpvpfile:badfiletype';
        errorstring = sprintf('readpvpfile:File %s has unrecognized file type %d',filename,hdr.filetype);
end
%% allow user to override value of numframes
if (exist("num_frames","var") && ~isempty(num_frames))
    numframes = num_frames;
end

if isempty(errorstring)
    if(numframes ~= round(numframes) || numframes <= 0)
        errorident = 'readpvpfile:badfilelength';
        errorstring = sprintf('readpvpfile:File %s has file length inconsistent with header',filename);
    end
end

if isempty(errorstring)
    data = cell(numframes-start_frame+1,1);
    switch hdr.datatype
        case 1 % PV_BYTE_TYPE
            precision = 'uchar';
        case 2 % PV_INT_TYPE
            precision = 'int32';
        case 3 % PV_FLOAT_TYPE
            precision = 'float32';
    end
    switch hdr.filetype
        case 1 % PVP_FILE_TYPE, used in HyPerCol::exitRunLoop
            numvalues = hdr.recordsize/hdr.datasize;
            for f=1:numframes
	      data_tmp = struct('time',hdr.time,'values',[]);
	      data_tmp.time = hdr.time;
	      Y = zeros(numvalues,hdr.numrecords);
              for r=1:hdr.numrecords
		     Y(:,r) = fread(fid, numvalues, precision);
              end
	      if f < 1
	      	 continue;
	      end
	      Z = zeros(hdr.nxGlobal,hdr.nyGlobal,hdr.nf);
	      r=0;
	      for y=1:hdr.nyprocs
		  yidx = (1:hdr.ny)+(y-1)*hdr.ny;
		  for x=1:hdr.nxprocs
                       xidx = (1:hdr.nx)+(x-1)*hdr.nx;
                       r = r+1;
                       for feature=1:hdr.nf
                          Z(xidx,yidx,feature) = reshape(Y(feature:hdr.nf:hdr.nx*hdr.ny*hdr.nf,r),hdr.nx,hdr.ny);
                       end
                   end
               end %% num_procs
               data_tmp.values = Z;
               if exist('progressperiod','var')
                  if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                  end
               end
	       if f < start_frame
	      	   continue;
	      	end
	       data{f - start_frame + 1} = data_tmp;
            end  %% num_frames
        case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking
            for f=1:numframes
                data_tmp = struct('time',0,'values',[]);
                data_tmp.time = fread(fid,1,'float64');
                numactive = fread(fid,1,'uint32');
                data_tmp.values = fread(fid,numactive,'uint32');
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                    end
                end
	 	if f < start_frame
	      	   continue;
	      	end
		data{f - start_frame + 1} = data_tmp;
            end %% num_frames
        case 3 % PVP_WGT_FILE_TYPE
            fseek(fid,0,'bof');
            for f=1:numframes
                hdr = readpvpheader(fid,ftell(fid));
                hdr = rmfield(hdr,'additional');
                numextrabytes = hdr.headersize - 80;
                fseek(fid,-numextrabytes,'cof');
                wgt_extra = [fread(fid,3,'int32');fread(fid,2,'float32');fread(fid,1,'int32')];
                numextra = numextrabytes/4 - 6;
                hdr.nxp = wgt_extra(1);
                hdr.nyp = wgt_extra(2);
                hdr.nfp = wgt_extra(3);
                hdr.wMin = wgt_extra(4);
                hdr.wMax = wgt_extra(5);
                hdr.numPatches = wgt_extra(6);
                if numextra > 0
                    hdr.additional = fread(fid,numextra,'int32');
                end
<<<<<<< .mine
                data_tmp = struct('time',hdr.time,'values',[],'nx',[],'ny',[],'offset',[]);
                data_tmp.values = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data_tmp.nx = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data_tmp.ny = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data_tmp.offset = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);                
=======
                data{f} = struct('time',hdr.time,'values',[],'nx',[],'ny',[],'offset',[]);
                data{f}.values = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data{f}.nx = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data{f}.ny = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data{f}.offset = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
>>>>>>> .r6729
                for arbor=1:hdr.nbands
                    for y=1:hdr.nyprocs
                        for x=1:hdr.nxprocs
                            cellindex = sub2ind([hdr.nxprocs hdr.nyprocs hdr.nbands],x,y,arbor);
                            % octave has trouble with multidim cell arrays
                            patchesperproc = hdr.numPatches/(hdr.nxprocs*hdr.nyprocs);
                            data_tmp.values{cellindex} = nan(hdr.nxp,hdr.nyp,hdr.nfp,patchesperproc);
                            for p=1:patchesperproc
                                patchnx = fread(fid,1,'uint16');
                                patchny = fread(fid,1,'uint16');
                                patchoffset = fread(fid,1,'uint32');
                                Z = fread(fid,hdr.nxp*hdr.nyp*hdr.nfp,precision);
                                tempdata = reshape(Z(1:hdr.nfp*hdr.nxp*hdr.nyp),hdr.nfp,hdr.nxp,hdr.nyp);
                                tempdata = permute(tempdata,[2 3 1]);
                                % Need to move shrunken patches
                                data_tmp.values{cellindex}(:,:,:,p) = tempdata;
                                data_tmp.nx{cellindex} = patchnx;
                                data_tmp.ny{cellindex} = patchny;
                                data_tmp.offset{cellindex} = patchoffset;
                            end
                            if hdr.datatype==1 % byte-type.  If float-type, no rescaling took place.
                                data_tmp.values{cellindex} = data_tmp.values{1}/255*(hdr.wMax-hdr.wMin)+hdr.wMin;
                            elseif hdr.datatype ~= 3
                                error('readpvpfile:baddatatype',...
                                    'Weight file type requires hdr.datatype of 1 or 3; received %d',...
                                    hdr.datatype);
                            end
                        end
                    end
                end  %% arbor
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                    end
                end
	 	if f < start_frame
	      	   continue;
	      	end
                data{f-start_frame+1} = data_tmp;
            end  %% num_frames
        case 4 % PVP_NONSPIKING_ACT_FILE_TYPE
            for f=1:numframes
                data_tmp = struct('time',0,'values',zeros(hdr.nxGlobal,hdr.nyGlobal,hdr.nf));
                data_tmp.time = fread(fid,1,'float64');
                for y=1:nyprocs
                    yidx = (1:hdr.ny)+(y-1)*hdr.ny;
                    for x=1:nxprocs
                        xidx = (1:hdr.nx)+(x-1)*hdr.nx;
                        Z = fread(fid,hdr.recordsize,precision);
                        data_tmp.values(xidx,yidx,:) = permute(reshape(Z,hdr.nf,hdr.nx,hdr.ny),[2 3 1]);
                    end
                end
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                    end
                end
	 	if f < start_frame
	      	   continue;
	      	end
                data{f-start_frame+1} = data_tmp;
	    end %% numframes
        case 5 % PVP_KERNEL_FILE_TYPE
            fseek(fid,0,'bof'); % there's a header in every frame, unlike other file types
            % So go back to the beginning and read the header in each frame.
            for f=1:numframes
                hdr = readpvpheader(fid,ftell(fid));
                hdr = rmfield(hdr,'additional');
                numextrabytes = hdr.headersize - 80;
                fseek(fid,-numextrabytes,'cof');
                wgt_extra = [fread(fid,3,'int32');fread(fid,2,'float32');fread(fid,1,'int32')];
                numextra = numextrabytes/4 - 6;
                hdr.nxp = wgt_extra(1);
                hdr.nyp = wgt_extra(2);
                hdr.nfp = wgt_extra(3);
                hdr.wMin = wgt_extra(4);
                hdr.wMax = wgt_extra(5);
                hdr.numPatches = wgt_extra(6);
                if numextra > 0
                    hdr.additional = fread(fid,numextra,'int32');
                end
                data_tmp = struct('time',hdr.time,'values',[]);
                data_tmp.values = cell(1,1,hdr.nbands);
                for arbor=1:hdr.nbands
                    cellindex = sub2ind([1 1 hdr.nbands],1,1,arbor);
                    % octave has trouble with multidim cell arrays
                    data_tmp.values{cellindex} = nan(hdr.nxp,hdr.nyp,hdr.nfp,hdr.numPatches);
                    for p=1:hdr.numPatches
                        fread(fid,1,'uint16'); % patch->nx
                        fread(fid,1,'uint16'); % patch->ny
                        fread(fid,1,'uint32'); % patch->offset
                        Z = fread(fid,hdr.nfp*hdr.nxp*hdr.nyp,precision);
                        tempdata = reshape(Z,hdr.nfp,hdr.nxp,hdr.nyp);
                        tempdata = permute(tempdata,[2 3 1]);
                        
                        data_tmp.values{cellindex}(:,:,:,p) = tempdata;
                    end
                    if hdr.datatype==1 % byte-type.  If float-type, no rescaling took place.
                        data_tmp.values{cellindex} = data_tmp.values{1}/255*(hdr.wMax-hdr.wMin)+hdr.wMin;
                    elseif hdr.datatype ~= 3
                        error('readpvpfile:baddatatype',...
                            'Weight file type requires hdr.datatype of 1 or 3; received %d',...
                            hdr.datatype);
                    end
                end
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                    end
                end
	 	if f < start_frame
	      	   continue;
	      	end
                data{f-start_frame+1} = data_tmp;
	    end %% num_frames
        otherwise
            assert(0); % This possibility should have been weeded out above
    end
end

fclose(fid);

if ~isempty(errorident)
    error(errorident,errorstring);
end
