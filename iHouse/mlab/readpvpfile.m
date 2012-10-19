function [data,hdr] = readpvpfile(filename,output_path,MOVIE_FLAG,progressperiod)
% Usage:[data,hdr] = readpvpfile(filename)
% filename is a pvp file (any type)
% data is a cell array containing the data.
%     In general, data has one element for each time step written.
%     Each element is a struct containing the fields 'time' and 'values'
%     For activities, values is an nx-by-ny-by-nf array.
%     For weights, values is a cell array, each element is an nxp-by-nyp-by-nfp array.

rootname     = '00';
OUT_FILE_EXT = 'jpg';             %either png or jpg for now
%MOVIE_FLAG   = 1;
global FNUM_ALL;
global FNUM_SPEC;              %Can be -1 for all or specify multiple start:int:end frames

inst_movie_path = [output_path,'Instantaneous_Frames/'];
if (exist(inst_movie_path, 'dir') ~= 7)
   mkdir(inst_movie_path);
end

%% Parse FNUM_SPEC
if (FNUM_ALL <= 0)
   fnum_flag = 1;
   frame_of_interest = [FNUM_SPEC{:}];
else
   fnum_flag = -1;
   frame_of_interest = [];
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
       %%Assert for max frame
       if fnum_flag > 0
           assert(max(frame_of_interest(:)) <= numframes-1);
       end
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
        case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking; I'm not using yet
            %Only one feature for now
            assert(hdr.nf == 1);
            movieFrame = 0;
            buffer = fread(fid, Inf, '*uint8');
            bufPos = 1;
            for frame=1:numframes
                data{frame} = struct('time',0,'values',[]);
                data{frame}.time = typecast(buffer(bufPos:bufPos + 7), 'double');
                bufPos += 8;
                numactive = typecast(buffer(bufPos:bufPos + 3), 'uint32'); 
                bufPos += 4;
                data{frame}.values = typecast(buffer(bufPos:bufPos + (4 * numactive) - 1), 'uint32');
                bufPos += 4 * numactive;
               % data{frame}.time = fread(fid,1,'float64');
               % numactive = fread(fid,1,'uint32');
               % data{frame}.values = fread(fid,numactive,'uint32');
                %If this is a frame of intrest
                if((~isempty(find(frame_of_interest == frame)) || FNUM_ALL > 0) && MOVIE_FLAG > 0)
                   %%%%%%%
                   %% Set up frame string for printing
                   %%%%%%%
                   if lt(movieFrame,10)
                       frame_str = ['00',num2str(movieFrame)];
                   elseif ge(movieFrame,10) && lt(movieFrame,100)
                       frame_str = ['0', num2str(movieFrame)];
                   elseif gt(movieFrame,99)
                       frame_str = num2str(movieFrame);
                   end%if lt(frame,10)
                   outImg = zeros([hdr.nyGlobal, hdr.nxGlobal]);
                   tempOutImg = outImg';
                   tempOutImg(data{frame}.values + 1) = 1;
                   outImg = tempOutImg';
                   print_movie_filename = [inst_movie_path,rootname,'_',frame_str,'.',OUT_FILE_EXT];
                   try
                       imwrite(outImg,print_movie_filename,OUT_FILE_EXT)
                       movieFrame += 1;
                   catch
                       disp(['readpvpfile: WARNING. Could not print file: ',char(10),print_movie_filename])
                   end%_try_catch
                 end%End frame_of_interest printing
            end%End num_frames
            if eq(MOVIE_FLAG,1)
                system(['ffmpeg -loglevel 0 -v 0 -r 20 -f image2 -i ',inst_movie_path,rootname,'_%03d.',OUT_FILE_EXT,' -sameq -y ',output_path,'pvp_instantaneous_movie.mp4 &']);
            end%if eq(MOVIE_FLAG,1)
            clear buffer;
        case 3 % PVP_WGT_FILE_TYPE
            fseek(fid,0,'bof');
            for f=1:numframes
                %Read header
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
                            if hdr.datatype==1 % byte-type.  If float-type, no rescaling took place.
                                data{f}.values{cellindex} = data{f}.values{1}/255*(hdr.wMax-hdr.wMin)+hdr.wMin;
                            elseif hdr.datatype ~= 3
                                error('readpvpfile:baddatatype',...
                                'Weight file type requires hdr.datatype of 1 or 3; received %d',...
                                hdr.datatype);
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
end
