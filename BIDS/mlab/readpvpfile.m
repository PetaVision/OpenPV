%function [data,hdr] = readpvpfile(filename,progressperiod)
% Usage:[data,hdr] = readpvpfile(filename)
% filename is a pvp file (any type)
% data is a cell array containing the data.
%     In general, data has one element for each time step written.
%     Each element is a struct containing the fields 'time' and 'values'
%     For activities, values is an nx-by-ny-by-nf array.
%     For weights, values is a cell array, each element is an nxp-by-nyp-by-nfp array.

%%%%%%%%
%%DELETE
%%%%%%%%
clear all; close all; more off;

global output_path;  output_path  = '/Users/bnowers/Documents/workspace/BIDS/output/';
global filename;     filename     = '/Users/bnowers/Documents/workspace/BIDS/output/BIDS_ADC_V_last.pvp';
global OUT_FILE_EXT; OUT_FILE_EXT = 'png';      %either png or jpg for now
global MOVIE_FLAG;   MOVIE_FLAG   = 1;          %set 1 to make a movie, set -1 to not make a movie
global FNUM_SPEC;    FNUM_SPEC    = '0:20:100'; %can be '-1', 'int(frame)', or 'start:int:end'
global PRINT_FLAG;   PRINT_FLAG   = 1;         %set to 1 to print all fiures to output dir, -1 to not
%%%%%%%%

%% Parse some flags
if ~strcmpi(FNUM_SPEC,'-1') %if flag is not set to -1
    fnum_flag = 1;

    disp_frames = str2num(FNUM_SPEC);
    if gt(length(disp_frames),1) %flag delimits start-frame:interval:end-frame
        frame_of_interest = -1; %will only be set if user gave a single frame as input to FNUM_SPEC
    elseif eq(length(disp_frames),1) %single number is given
        frame_of_interest = disp_frames+1;
    else
        error(['FNUM_SPEC is not correct. FNUM_SPEC = ',FNUM_SPEC])
    end%if gt(length(disp_frames),1)
else
    fnum_flag = -1;
    frame_of_interest = -1;
end%if strcmpi

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
end%if fid<0

errorident = '';
errorstring = '';

hdr = readpvpheader(fid);

%% Correct for max frame
if fnum_flag
    if gt(length(disp_frames),1)
        if gt(max(disp_frames(:)),hdr.nbands-1)
            disp_frames(end) = hdr.nbands-1;
        end%if gt(max...
    end%if gt(length...
end%if fnum_flag

switch hdr.filetype
    case 1 % PVP_FILE_TYPE
        framesize = hdr.recordsize*hdr.numrecords;
        numframes = (filedata(1).bytes - hdr.headersize)/framesize;
    case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking
        numframes = hdr.nbands;
        % framesize is variable
    case 3 % PVP_WGT_FILE_TYPE % HyPerConns that aren't KernelConns
        framesize = hdr.recordsize*hdr.numrecords*hdr.nbands+hdr.headersize;
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

if isempty(errorstring)
    if(numframes ~= round(numframes) || numframes <= 0)
        errorident = 'readpvpfile:badfilelength';
        errorstring = sprintf('readpvpfile:File %s has file length inconsistent with header',filename);
    end
end

if isempty(errorstring)
    data = cell(numframes,1);
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
                data{f} = struct('time',hdr.time,'values',[]);
                data{f}.time = hdr.time;
                Y = zeros(numvalues,hdr.numrecords);
                for r=1:hdr.numrecords
                    Y(:,r) = fread(fid, numvalues, precision);
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
                end
                data{f}.values = Z;
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                    end
                end
            end
        case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking
            status = fseek(fid,4*hdr.numparams,"bof"); %hdr.numparams = number of parameters specified in the header file
            if ne(status,0)
                error('readpvpfile: unable to seek to the end of the header for pvp file ',filedata)
            end%if ne(status,0)

            integrated_image = ones([hdr.nxGlobal,hdr.nyGlobal]);

            for frame=0:numframes-1
                if lt(frame,10)
                    frame_str = ['00',num2str(frame)];
                elseif ge(frame,10) && lt(frame,100)
                    frame_str = ['0', num2str(frame)];
                elseif gt(frame,99)
                    frame_str = num2str(frame);
                end%if lt(frame,10)

                [time, count] = fread(fid,1,"float64"); %count is the number of items read
                if ne(time,frame) %pvp files start at time_step = 0
                    error(['readpvpfile: Not reading the correct frame. frame = ',num2str(frame),...
                        'time = ',num2str(time)])
                end%if ne(time,frame)
                if eq(count,0)
                    error('readpvpfile: Did not read any elements into time from pvp file')
                end%if eq(count,0)
                if feof(fid) 
                  break;
                end%if feof

                [num_spikes, count] = fread(fid,1,"int32");
                if eq(count,0)
                    error('readpvpfile: Did not read any elements into num_spikes from pvp file')
                end%if eq(count,0)
                [spike_idx, count] = fread(fid,num_spikes,"int32"); %spike_idx is a vector of indices for each spike
                if ne(count,num_spikes)
                    error(['readpvpfile: The number of elements read in does not equal the number of spikes fired.',...
                        char(10),'numbewhile pvp_time < pvp_framer of elements = ',num2str(count),'number of spikes = ',num2str(num_spikes)])
                end%if ne(count,num_spikes)

                file_pos = ftell(fid);
                sparse_size = hdr.nf*hdr.nxGlobal*hdr.nyGlobal;
                activity = sparse(spike_idx+1, 1, 1, sparse_size, 1, num_spikes); %spike_idx+1 because pvp starts idx at 0

                features = full(activity);
                features = reshape(features,[hdr.nf,hdr.nxGlobal,hdr.nyGlobal]); %nf = num_features, nx = num_col, ny = num_rows

                pvp_image = zeros([hdr.nxGlobal,hdr.nyGlobal]);
                pvp_image(:,:) = sum(features,1);
                pvp_image = flipud(rot90(pvp_image));

                integrated_image += pvp_image;
               
                %% Generate Movie frames
                if eq(MOVIE_FLAG,1)
                    print_movie_path = [output_path, 'Movie/'];
                    if ne(exist(print_movie_path,'dir'),7) %if exists func doesn't return a 7, then print_movie_path is not a dir
                        mkdir(print_movie_path);
                    end%if ne(exist(),7)

                    inst_movie_path = [print_movie_path,'Instantaneous_Frames/'];
                    if ne(exist(inst_movie_path,'dir'),7) %if exists func doesn't return a 7, then inst_movie_path is not a dir
                        mkdir(inst_movie_path);
                    end%if ne(exist(),7)

                    print_movie_filename = [inst_movie_path,frame_str,'.',OUT_FILE_EXT];
                    try
                        imwrite(pvp_image,print_movie_filename,OUT_FILE_EXT)
                    catch
                        disp(['readpvpfile: WARNING. Could not print file(s):',...
                            char(10),print_movie_filename,])
                    end%_try_catch
                end%if eq(MOVIE_FLAG,1)

                %% Display figures
                if gt(fnum_flag,0)
                    if eq(frame,frame_of_interest) || find(disp_frames==frame)
                        figs_disp_flag = 1;

                        tim_fig_id = figure;
                            imshow(pvp_image)
                            axis image
                            axis off

                        int_fig_id = figure;
                            imagesc(integrated_image)
                            axis image
                            axis off
                            colorbar

                        fig_ids = [2];
                        fig_ids(1) = tim_fig_id;
                        fig_ids(2) = int_fig_id;
                    else
                        figs_disp_flag = 0;
                    end%if eq(frame,frame_of_interest)
                end %gt(fnum_flag,0)

                %% Print out figures
                if gt(PRINT_FLAG,0)
                    if figs_disp_flag
                        print_fig_path = [output_path, 'Figures/'];
                        if ne(exist(print_fig_path,'dir'),7) %if exists func doesn't return a 7, then print_fig_path is not a dir
                            mkdir(print_fig_path);
                        end%if ne(exist(),7)

                        print_fig_filename = [print_fig_path,'frame',frame_str,'_fig'];
                        try
                            for fig_idx = 1:length(fig_ids)
                                curr_fig_id = fig_ids(fig_idx);
                                print(curr_fig_id,['-d',OUT_FILE_EXT],[print_fig_filename,num2str(curr_fig_id),'.',OUT_FILE_EXT]);
                                close(curr_fig_id)
                            end%for fig_idx
                        catch
                            disp(['readpvpfile: Couldn''t print figure ',print_fig_filename])
                        end%_try_catch
                    end%if fig_id
                end%if PRINT_FLAG
            end%for frame

            %% Create movie from frames generated
            if eq(MOVIE_FLAG,1)
                system(['ffmpeg -r 12 -f image2 -i ',inst_movie_path,'%03d.png -sameq -y ',print_movie_path,'pvp_instantaneous_movie.mp4']);
                system(['rm -rf ',inst_movie_path]); %Comment this line to keep movie frames
            end%if eq(MOVIE_FLAG,1)

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
                data{f} = struct('time',hdr.time,'values',[],'nx',[],'ny',[],'offset',[]);
                data{f}.values = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data{f}.nx = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data{f}.ny = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);
                data{f}.offset = cell(hdr.nxprocs,hdr.nyprocs,hdr.nbands);                
                for arbor=1:hdr.nbands
                    for y=1:hdr.nyprocs
                        for x=1:hdr.nxprocs
                            cellindex = sub2ind([hdr.nxprocs hdr.nyprocs hdr.nbands],x,y,arbor);
                            % octave has trouble with multidim cell arrays
                            patchesperproc = hdr.numPatches/(hdr.nxprocs*hdr.nyprocs);
                            data{f}.values{cellindex} = nan(hdr.nxp,hdr.nyp,hdr.nfp,patchesperproc);
                            for p=1:patchesperproc
                                patchnx = fread(fid,1,'uint16');
                                patchny = fread(fid,1,'uint16');
                                patchoffset = fread(fid,1,'uint32');
                                Z = fread(fid,hdr.nxp*hdr.nyp*hdr.nfp,precision);
                                tempdata = reshape(Z(1:hdr.nfp*hdr.nxp*hdr.nyp),hdr.nfp,hdr.nxp,hdr.nyp);
                                tempdata = permute(tempdata,[2 3 1]);
                                % Need to move shrunken patches
                                data{f}.values{cellindex}(:,:,:,p) = tempdata;
                                data{f}.nx{cellindex} = patchnx;
                                data{f}.ny{cellindex} = patchny;
                                data{f}.offset{cellindex} = patchoffset;
                            end
                        end
                    end
                end
            end
        case 4 % PVP_NONSPIKING_ACT_FILE_TYPE
            for f=1:numframes
                data{f} = struct('time',0,'values',zeros(hdr.nxGlobal,hdr.nyGlobal,hdr.nf));
                data{f}.time = fread(fid,1,'float64');
                for y=1:nyprocs
                    yidx = (1:hdr.ny)+(y-1)*hdr.ny;
                    for x=1:nxprocs
                        xidx = (1:hdr.nx)+(x-1)*hdr.nx;
                        Z = fread(fid,hdr.recordsize,precision);
                        data{f}.values(xidx,yidx,:) = permute(reshape(Z,hdr.nf,hdr.nx,hdr.ny),[2 3 1]);
                    end
                end
                if exist('progressperiod','var')
                    if ~mod(f,progressperiod)
                        fprintf(1,'File %s: frame %d of %d\n',filename, f, numframes);
                        fflush(1);
                    end
                end
            end
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
                data{f} = struct('time',hdr.time,'values',[]);
                data{f}.values = cell(1,1,hdr.nbands);
                for arbor=1:hdr.nbands
                    cellindex = sub2ind([1 1 hdr.nbands],1,1,arbor);
                    % octave has trouble with multidim cell arrays
                    data{f}.values{cellindex} = nan(hdr.nxp,hdr.nyp,hdr.nfp,hdr.numPatches);
                    for p=1:hdr.numPatches
                        fread(fid,1,'uint16'); % patch->nx
                        fread(fid,1,'uint16'); % patch->ny
                        fread(fid,1,'uint32'); % patch->offset
                        Z = fread(fid,hdr.nfp*hdr.nxp*hdr.nyp,precision);
                        tempdata = reshape(Z,hdr.nfp,hdr.nxp,hdr.nyp);
                        tempdata = permute(tempdata,[2 3 1]);
                                                
                        data{f}.values{cellindex}(:,:,:,p) = tempdata;
                    end
                    if hdr.datatype==1 % byte-type.  If float-type, no rescaling took place.
                        data{f}.values{cellindex} = data{f}.values{1}/255*(hdr.wMax-hdr.wMin)+hdr.wMin;
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
            end
        otherwise
            assert(0); % This possibility should have been weeded out above
    end
end

fclose(fid);

if ~isempty(errorident)
    error(errorident,errorstring);
end
