function [data,hdr] = readpvpfile(filename,progressperiod, last_frame, start_frame, skip_frames)
% Usage:[data,hdr] = readpvpfile(filename,progressperiod, last_frame, start_frame)
% filename is a pvp file (any type)
% progressperiod is an optional integer argument.  A message is printed
%     to the screen every progressperiod frames.
% last_frame is the index of the last frame to read.  Default is all frames.
% start_frame is the starting frame.  Default is 1.
%
% data is a cell array containing the data.
%     In general, data has one element for each time step written.
%     Each element is a struct containing the fields 'time' and 'values'
%     For activities, values is an nx-by-ny-by-nf array.
%     For weights, values is a cell array, each element is an nxp-by-nyp-by-nfp array.
% hdr is a struct containing the information in the file's header

%% start_frame allows the user to only read frames from some starting point
if nargin < 4 || ~exist('start_frame','var') || isempty(start_frame)
    start_frame = 1;
end%if

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
    case 1 % PVP_FILE_TYPE, obsolete
        errorident = 'readpvpfile:obsoletefiletype';
        errorstring = sprintf('readpvpfile:File %s has obsolete file typd %d', filename, hdr.filetype);
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
    case 6 % PVP_ACT_SPARSEVALUES_FILE_TYPE
        numframes = hdr.nbands;
        % framesize is variable
    otherwise
        errorident = 'readpvpfile:badfiletype';
        errorstring = sprintf('readpvpfile:File %s has unrecognized file type %d',filename,hdr.filetype);
end
if (~exist('skip_frames','var') || isempty(skip_frames)) || skip_frames < 1
    skip_frames = 1;
end
%% allow user to override value of numframes
if (exist('last_frame','var') && ~isempty(last_frame))
    lastframe = min(last_frame, numframes);
else
    lastframe = numframes;
end%if
tot_frames = ceil((lastframe-start_frame+1)/skip_frames);

if isempty(errorstring)
    if(lastframe ~= round(lastframe) || lastframe <= 0)
        errorident = 'readpvpfile:badfilelength';
        errorstring = sprintf('readpvpfile:File %s has file length inconsistent with header',filename);
    end%if
end%if

if isempty(errorstring)
    %%data = cell(lastframe-start_frame+1,1);
    data = cell(tot_frames,1);
    switch hdr.datatype
        case 1 % PV_BYTE_TYPE
            precision = 'uchar';
            data_size = 1;
        case 2 % PV_INT_TYPE
            precision = 'int32';
            data_size = 4;
        case 3 % PV_FLOAT_TYPE
            precision = 'float32';
            data_size = 4;
        case 4 % PV_SPARSEVALUES_TYPE
            precision = 'int32'; % data alternates between int32 and float32 but we'll typecast to get the values
            data_size = 4;
    end
    switch hdr.filetype
        case 1 % PVP_FILE_TYPE, obsolete
            assert(0); % Error should be caught above.
        case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking
            data_tmp = struct('time',0,'values',[]);
            for f=1:lastframe
                data_tmp.time = fread(fid,1,'float64');
                numactive = fread(fid,1,'uint32');
                data_tmp.values = fread(fid,numactive,'uint32');
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, lastframe);
                        if exist('fflush')
                           fflush(1);
                        end%if
                    end%if
                end%if
                if f < start_frame || mod(f,skip_frames)~=0
                    continue;
                end%if
                data{ceil((f - start_frame + 1)/skip_frames)} = data_tmp;
            end%for %% last_frame
        case 3 % PVP_WGT_FILE_TYPE
            %fseek(fid,0,'bof');
            fseek(fid, (start_frame-1)*(data_size*hdr.recordsize + hdr.headersize + 8), 'bof');
            for f=start_frame:lastframe
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
                data_tmp = struct('time',hdr.time,'values',[],'nx',[],'ny',[],'offset',[]);
                data_tmp.values = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data_tmp.nx = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data_tmp.ny = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data_tmp.offset = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                for arbor=1:hdr.nbands
                    for y=1:hdr.nyprocs
                        for x=1:hdr.nxprocs
                            cellindex = sub2ind([hdr.nxprocs hdr.nyprocs hdr.nbands],x,y,arbor);
                            % octave has trouble with multidim cell arrays
                            patchesperproc = hdr.numPatches/(hdr.nxprocs*hdr.nyprocs);
                            data_tmp.values{cellindex} = nan(hdr.nxp,hdr.nyp,hdr.nfp,patchesperproc);
                            data_tmp.nx{cellindex} = zeros(patchesperproc,1);
                            data_tmp.ny{cellindex} = zeros(patchesperproc,1);
                            data_tmp.offset{cellindex} = zeros(patchesperproc,1);
                            for p=1:patchesperproc
                                patchnx = fread(fid,1,'uint16');
                                patchny = fread(fid,1,'uint16');
                                patchoffset = fread(fid,1,'uint32');
                                Z = fread(fid,hdr.nxp*hdr.nyp*hdr.nfp,precision);
                                tempdata = reshape(Z(1:hdr.nfp*hdr.nxp*hdr.nyp),hdr.nfp,hdr.nxp,hdr.nyp);
                                tempdata = permute(tempdata,[2 3 1]);
                                % Need to move shrunken patches
                                data_tmp.values{cellindex}(:,:,:,p) = tempdata;
                                data_tmp.nx{cellindex}(p,1) = patchnx;
                                data_tmp.ny{cellindex}(p,1) = patchny;
                                data_tmp.offset{cellindex}(p,1) = patchoffset;
                            end%for
                            if hdr.datatype==1 % byte-type.  If float-type, no rescaling took place.
                                data_tmp.values{cellindex} = data_tmp.values{cellindex}/255*(hdr.wMax-hdr.wMin)+hdr.wMin;
                            elseif hdr.datatype ~= 3
                                error('readpvpfile:baddatatype',...
                                    'Weight file type requires hdr.datatype of 1 or 3; received %d',...
                                    hdr.datatype);
                            end%if
                        end%for %% x
                    end%for %% y
                end  %% arbor
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, lastframe);
                        if exist('fflush')
                           fflush(1);
                        end%if
                    end%if
                end%if
                if f < start_frame || mod(f,skip_frames)~=0
                    continue;
                end%if
                data{ceil((f - start_frame + 1)/skip_frames)} = data_tmp;
            end  %% last_frame
        case 4 % PVP_NONSPIKING_ACT_FILE_TYPE
            fseek(fid,(start_frame-1)*(hdr.recordsize*4 + 8), 'cof');
            for f=start_frame:lastframe
                data_tmp = struct('time',0,'values',zeros(hdr.nxGlobal,hdr.nyGlobal,hdr.nf));
                data_tmp.time = fread(fid,1,'float64');
                for y=1:nyprocs
                    yidx = (1:hdr.ny)+(y-1)*hdr.ny;
                    for x=1:nxprocs
                        xidx = (1:hdr.nx)+(x-1)*hdr.nx;
                        Z = fread(fid,hdr.recordsize,precision);
                        data_tmp.values(xidx,yidx,:) = permute(reshape(Z,hdr.nf,hdr.nx,hdr.ny),[2 3 1]);
                    end%for
                end%for
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, lastframe);
                        if exist('fflush')
                           fflush(1);
                        end%if
                    end%if
                end%if
                if f < start_frame || mod(f,skip_frames)~=0
                    continue;
                end%if
                data{ceil((f - start_frame + 1)/skip_frames)} = data_tmp;
            end%for %% lastframe
        case 5 % PVP_KERNEL_FILE_TYPE
            %keyboard;
            fseek(fid, (start_frame-1)*framesize, 'bof');
            for f=start_frame:lastframe
                % fseek(fid,0,'bof'); % there's a header in every frame, unlike other file types
                % So go back to the beginning and read the header in each frame.
                % for f=1:lastframe
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
                end%if
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
                    end%for
                    if hdr.datatype==1 % byte-type.  If float-type, no rescaling took place.
                        data_tmp.values{cellindex} = data_tmp.values{cellindex}/255*(hdr.wMax-hdr.wMin)+hdr.wMin;
                    elseif hdr.datatype ~= 3
                        error('readpvpfile:baddatatype',...
                            'Weight file type requires hdr.datatype of 1 or 3; received %d',...
                            hdr.datatype);
                    end%if
                end%for
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, lastframe);
                        if exist('fflush')
                           fflush(1);
                        end%if
                    end%if
                end%if
                if f < start_frame || mod(f,skip_frames)~=0
                    continue;
                end%if
                data{ceil((f - start_frame + 1)/skip_frames)} = data_tmp;
            end%for %% last_frame
        case 6 % PVP_ACT_SPARSEVALUES_FILE_TYPE
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
                if f < start_frame || mod(f,skip_frames)~=0
                    continue;
                end%if
                data{ceil((f - start_frame + 1)/skip_frames)} = data_tmp;
            end%for %% last_frame
        otherwise
            assert(0); % This possibility should have been weeded out above
    end
end%if

fclose(fid);

if ~isempty(errorident)
    error(errorident,errorstring);
end%if
