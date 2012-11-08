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

global wrkspc_path;  wrkspc_path  = '~/NeoVision2';
global output_path;  output_path  = [wrkspc_path,'/neovision-programs-petavision/Heli/Challenge/026/p0/ns14850'];
global filename;     filename     = [output_path,'/GanglionON.pvp'];
global rootname;     rootname     = '00';

global OUT_FILE_EXT; OUT_FILE_EXT = 'png';             %either png or jpg for now
global IN_FILE_EXT; IN_FILE_EXT = 'png';
%%%%%%%%

global recon_output_path;  recon_output_path  = [output_path,'/movON'];
global sourceDir;    sourceDir    = ['~/NeoVision2/neovision-data-challenge-heli/026'];
gray2rgb = @(Image) double(cat(3,Image,Image,Image))./255;
fps = 33;



filedata = dir(filename);
if length(filedata) ~= 1
  error('readpvpfile:notonefile',...
        'Path %s should expand to exactly one file; in this case there are %d',...
        filename,length(filedata));
end

if filedata(1).bytes < 1
  error(['readpvpfile:fileempty',...
         'File', filname, 'is empty']);
endif %%filedata(1).bytes

fid = fopen(filename);

errorident = '';
errorstring = '';

hdr = readpvpheader(fid);


switch hdr.filetype
  case 1 % PVP_FILE_TYPE
    framesize = hdr.recordsize*hdr.numrecords;
    numframes = (filedata(1).bytes - hdr.headersize)/framesize;
  case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking
    disp(['PVP File Type: Spiking'])
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
    errorstring = sprintf(["readpvpfile:File ", filename, " has unrecognized file type ", hdr.filetype]);
endswitch
			  
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
        case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking; 
            status = fseek(fid,4*hdr.numparams,"bof"); %hdr.numparams = number of parameters specified in the header file
            if ne(status,0)
                error('readpvpfile: unable to seek to the end of the header for pvp file ',filedata)
            end%if ne(status,0)while pvp_time < pvp_frame

	    inFiles = glob([sourceDir,'/*.', IN_FILE_EXT]);
	    if ne(length(inFiles)*fps,numframes)
		error(["length(inFiles)*fps != numframes in:", sourceDir]);
            endif

            inFrame = 1;
            for frame=0:numframes-1
                if feof(fid) 
                  break;
                end%if feof

                %%%%%%%
                %% Verify time step
                %%%%%%%
                [time, count] = fread(fid,1,"float64"); %count is the number of items read
                if eq(count,0)
                    error('readpvpfile: Did not read any elements into time from pvp file')
                end%if eq(count,0)
                if ne(time,frame) %pvp files start at time_step = 0
                    error(['readpvpfile: Not reading the correct frame. frame = ',num2str(frame),...
                        'time = ',num2str(time)])
                end%if ne(time,frame)

                %%%%%%%
                %% Get the total number of spikes in this time step
                %%%%%%%
                [num_spikes, count] = fread(fid,1,"int32");
                if eq(count,0)
                    error('readpvpfile: Did not read any elements into num_spikes from pvp file')
                end%if eq(count,0)

                %%%%%%%
                %% Get the index for each spike for this time step
                %%%%%%%
                [spike_idx, count] = fread(fid,num_spikes,"int32"); %spike_idx is a vector of indices for each spike
                if ne(count,num_spikes)
                    error(['readpvpfile: The number of elements read in does not equal the number of spikes fired.',...
                        char(10),'numbewhile pvp_time < pvp_framer of elements = ',num2str(count),'number of spikes = ',num2str(num_spikes)])
                end%if ne(count,num_spikes)

                %%%%%%%
                %% Get sparse activity
                %%%%%%%
                sparse_size = hdr.nf*hdr.nxGlobal*hdr.nyGlobal;
                activity = sparse(spike_idx+1, 1, 1, sparse_size, 1, num_spikes); %spike_idx+1 because pvp starts idx at 0

                features = full(activity);
                features = reshape(features,[hdr.nf,hdr.nxGlobal,hdr.nyGlobal]); %nf = num_features, nx = num_col, ny = num_rows

                %%%%%%%
                %% Create PVP image & integrated image(s)
                %%%%%%%
                pvp_image = squeeze(sum(features,1));
                pvp_image = flipud(rot90(pvp_image));


                %%%%%%%
                %% Overlay onto input image
                %%%%%%%

                if lt(frame,10)
                    inFrame_str = ['0000',num2str(frame)];
                elseif ge(frame,10) && lt(frame,100)
                    inFrame_str = ['000', num2str(frame)];
                elseif ge(frame,100) && lt(frame,1000)
                    inFrame_str = ['00', num2str(frame)];
                elseif ge(frame,1000) && lt(frame,10000)
                    inFrame_str = ['0', num2str(frame)];
                elseif ge(frame,10000)
                    inFrame_str = num2str(frame);
                end%if lt(inFrame,10)

                if ne(exist(recon_output_path,'dir'),7) %if exists func doesn't return a 7, then movie_path is not a dir
                    mkdir(recon_output_path);
                end%if ne(exist(),7)

                orig_img = imread(inFiles{inFrame});
                idx = find(orig_img<64);
                orig_img(idx) = 64;
                idx = find(orig_img>192);
                orig_img(idx) = 192;
		if (~isrgb(orig_img))
                  orig_img = gray2rgb(orig_img);
	        endif                
		resized_pvp_image = logical(imresize(pvp_image,2));
                %resized_pvp_image += abs(min(resized_pvp_image(:)));
                orig_img(:,:,1) += 100.*resized_pvp_image;
                idx = find(orig_img>255);
                orig_img(idx) = 255;
                imwrite(orig_img,[recon_output_path,'/Image_',inFrame_str,'.jpg'])
                if eq(mod(frame+1,fps),0)
                    inFrame += 1;
                end

            end%for frame


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

