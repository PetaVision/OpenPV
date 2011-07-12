function W = readweights(weightfile, progressupdateperiod)
% W = readweights(weightfile)
%
% Reads the weights from a PetaVision-generated weight file
% weightfile is a filename.
% progressupdateperiod is a scalar that indicates how often to output progress.
%              Default is 0 (no output).
% W is a 4-dimensional array.
% W(i,j,m,n) is the weight of the (i,j)th location for feature m
% and kernel patch n.

if ~exist('progressupdateperiod','var')
    progressupdateperiod = 0;
end

if ~isscalar(progressupdateperiod)
    error('readweights:progressupdateperiodscalar',...
          'progressupdateperiod must be a scalar');
end

if progressupdateperiod < 0
    progressupdateperiod = 0;
end

if progressupdateperiod ~= round(progressupdateperiod)
    error('readweights:progressupdateperiodinteger',...
          'progressupdateperiod must be a nonnegative integer');
end

filedata = dir(weightfile);
if length(filedata) ~= 1
    error('readweights:notonefile',...
          'Path %s should expand to exactly one file; in this case there are %d',...
          weightfile,length(filedata));
end

if filedata(1).bytes < 1
    error('readweights:fileempty',...
          'File %s is empty',weightfile);
end%if filedata(1).bytes

fid = fopen(weightfile);
if fid == -1
    error('readweights:cantopenfile','Can''t open %s',weightfile);
end

errormsg = '';
rec=1;
W = cell(1,1);
while filedata(1).bytes > ftell(fid);
    headersize = fread(fid,1,'int32');
    if headersize < 104
        errormsg = 'readweights:headertooshort';
        errorstr = sprintf(...
            'File %s: header size must be at least 104.  Value is %d',...
            weightfile, headersize);
        break;
    end%if headersize<
    if headersize > filedata(1).bytes - ftell(fid) + 4;
        errormsg = 'readweights:headertoolong';
        errorstr = sprintf(...
            'File %s: header size is %d but remaining file length is only %d',...
            weightfile, headersize, filedata(1).bytes);
        break;
    end%if headersize>
    hdr = [headersize; fread(fid,17,'int32'); fread(fid,1,'float64')];
    % hdr(1): header size
    % hdr(2): number of parameters
    % hdr(3): file type
    % hdr(4): nx (presynaptic layer's nx when called by HyPerConn::writeWeights)
    % hdr(5): ny (    "    )
    % hdr(6): nf (    "    )
    % hdr(7): number of records = nxprocs * nyprocs (1 for non-mpi runs?)
    % hdr(8): index record size
    % hdr(9): index data size
    % hdr(10): index data type
    % hdr(11): index nx procs
    % hdr(12): index ny procs
    % hdr(13): index nx global
    % hdr(14): index ny global
    % hdr(15): index kx0
    % hdr(16): index ky0
    % hdr(17): nb (presynaptic layer's nb when called by HyPerConn::writeWeights)
    % hdr(18): nbands (presynaptic layer's nf when called by HyPerConn::writeWeights
    % hdr(19): time stored as double (takes two 32-bit words)
    if hdr(3) ~= 3
        errormsg = 'readweights:badfiletype';
        errorstr = sprintf(...
            'File %s: file type is %d; readweights expects 3',...
            weightfile,hdr(3));
        break;
    end% if hdr(3)
    
    numparams = hdr(2);
    if numparams * 4 ~= headersize
        errormsg = 'readweights:badnumparams';
        errorstr = sprintf(...
            'File %s: number of parameters %d; expected %d from headersize',...
            weightfile,numparams,headersize/4);
        break;
    end%if numparams*4
    
    if numparams < 26
        errormsg = 'readweights:badnumparams';
        errorstr = sprintf(...
            'File %s: weight files need numparams at least 26; value was %d\n',...
            weightfile,numparams);
        break;
    end%if numparams<26

    datasize = hdr(9);
    datatype = hdr(10);
    dtype = '';
    if datasize == 1 && datatype == 1
        dtype = 'char*1';
    end%if hdr(9)==1
    if datasize == 4 && datatype == 3
        dtype = 'float32';
    end%if hdr(9)==4
    if isempty(dtype)
        errormsg = 'readweights:baddatatype';
        errorstr = sprintf(...
            'File %s: data type %d and data size %d are not supported',...
            weightfile, hdr(10), hdr(9) );
    end%if ~datatype

    if filedata(1).bytes < ftell(fid) + 24
        errormsg = 'readweights:badweightheader';
        errorstr = sprintf(...
            'File %s: weights must have an additional header of at least 6 32-bit words',...
            weightfile);
        break;
    end%if filedata(1).bytes
    
    wgthdr = [fread(fid,3,'int32'); fread(fid,2,'float32'); fread(fid,1,'int32')];
    % wgthdr(1): nxp
    % wgthdr(2): nyp
    % wgthdr(3): nfp
    % wgthdr(4): wMin
    % wgthdr(5): wMax
    % wgthdr(6): numPatches
    nxp=wgthdr(1);
    nyp=wgthdr(2);
    nfp=wgthdr(3);
    numpatches = wgthdr(6);
    
    patchsize = nxp*nyp*nfp;
    numprocs = hdr(11)*hdr(12);
    expectedrecordsize = numpatches*(4+datasize*patchsize)/numprocs;
    % this may be wrong if marginwidth is nonzero
    if expectedrecordsize ~= hdr(8)
        errormsg = 'readweights:badrecordsize';
        errorstr = sprintf(...
            'File %s: record size was %d; size expected from patch dimensions %d',...
            weightfile, hdr(8), expectedrecordsize);
        break;
    end%if expectedrecordsize
    
    if filedata(1).bytes < ftell(fid) + hdr(8)
        errormsg = 'readweights:badrecord';
        errorstr = sprintf(...
            'File %s: record size in header is %d, but only %d bytes remain in file',...
            weightfile, hdr(8), filedata(1).bytes-ftell(fid));
        break;
    end%if filedata(1).bytes
    W{rec,1} = NaN(nxp,nyp,nfp,numpatches);
    for p=1:numpatches
        patchnx = fread(fid,1,'int16'); % usually nx and ny, but there might be
        patchny = fread(fid,1,'int16'); % shrunken patches
        A = fread(fid,nxp*nyp*nfp,dtype);
        A = A(1:patchnx*patchny*nfp);
        % I think it's correct that shrunken patches are padded with 0
        A = reshape(A,nfp,patchnx,patchny);
        if stringsidentical(dtype,'char*1')
            wMin = wgthdr(4);
            wMax = wgthdr(5);
            A = wMin + A/255*(wMax-wMin);
        end%if dtype
        for f=1:nfp
            W{rec,1}(1:patchnx,1:patchny,f,p) = squeeze(A(f,:,:));
            % If there are shrunken patches, W will have NaNs.
            % The file format doesn't say how the shrunken patch is offset.
        end%for f
    end%for p
    
    if progressupdateperiod > 0 && ~mod(rec,progressupdateperiod);
        fprintf(1,'Weight record %d\n',rec);
    end    
    rec = rec+1;
end%while

fclose(fid);
if ~isempty(errormsg)
    error(errormsg,errorstr);
end%if ~isempty(errormsg)