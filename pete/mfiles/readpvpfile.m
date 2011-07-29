function [data,hdr] = readpvpfile(filename)
% [data,hdr] = readpvpfile(filename)

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
    case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking; I'm not using yet
        errorident = 'readpvpfile:unimplementedfiletype';
        errorstring = sprintf('readpvpfile:File %s has unimplemented file type %d',filename,hdr.filetype);
    case 3 % PVP_WEIGHT_FILE_TYPE
        framesize = hdr.recordsize*hdr.numrecords+hdr.headersize;
        numframes = filedata(1).bytes/framesize;
    case 4 % PVP_NONSPIKING_ACT_FILE_TYPE
        % who defined this stupid format?
        nxprocs = hdr.nxGlobal/hdr.nx;
        nyprocs = hdr.nyGlobal/hdr.ny; % I mean, what is the "contiguous" flag supposed to indicate?
        framesize = hdr.recordsize*hdr.datasize*nxprocs*nyprocs+8;
        numframes = (filedata(1).bytes - hdr.headersize)/framesize;
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
                        for band=1:hdr.nf
                            Z(xidx,yidx,band) = reshape(Y(band:hdr.nf:hdr.nx*hdr.ny*hdr.nf,r),hdr.nx,hdr.ny);
                        end
                    end
                end
                data{f}.values = Z;
            end
            % case 2 % PVP_ACT_FILE_TYPE % Compressed for spiking; I'm not using yet
        case 3 % PVP_WGT_FILE_TYPE
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
                patchesperproc = hdr.numPatches/hdr.nxprocs/hdr.nyprocs;
                data{f} = struct('time',hdr.time,'values',[]);
                data{f}.values = cell(hdr.nyprocs,hdr.nxprocs);
                %TODO scales by wMin and wMax if datatype is byte-type
                for y=1:hdr.nyprocs
                    for x=1:hdr.nxprocs
                        data{f}.values{y,x} = zeros(hdr.nxp,hdr.nyp,hdr.nfp,patchesperproc);
                        for p=1:patchesperproc
                            patchnx = fread(fid,1,'int16');
                            patchny = fread(fid,1,'int16');
                            Z = fread(fid,hdr.nfp*hdr.nxp*hdr.nyp,precision);
                            data{f}.values{y,x}(:,:,:,p) = permute(reshape(Z,[hdr.nfp,hdr.nxp,hdr.nyp]),[2 3 1]);
                            if patchnx~=hdr.nxp || patchny~=hdr.nyp
                                warning('readpvpfile:shrunkenpatch',...
                                    'Readpvpfile:Warning:data{%f}{%f}{%f}, patch %d: shrunken patches not implemented',f,x,y,p);
                            end
                        end
                        if hdr.datatype==1 % byte-type.  If float-type, no rescaling took place.
                            data{f}.values{y,x} = data{f}.values{y,x}/255*(hdr.wMax-hdr.wMin)+hdr.wMin;
                        elseif hdr.datatype ~= 3
                            error('readpvpfile:baddatatype',...
                                  'Weight file type requires hdr.datatype of 1 or 3; received %d',...
                                  hdr.datatype);
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
            end
        otherwise
            assert(0); % This possibility should have been weeded out above
    end
end

fclose(fid);

if ~isempty(errorident)
    error(errorident,errorstring);
end