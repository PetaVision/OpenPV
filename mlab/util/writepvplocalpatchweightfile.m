function writepvplocalpatchweightfile(filename, data, nxRestricted, nyRestricted, nfPre, nxExtended, nyExtended)
    % Usage: writepvplocalpatchweightfile(filename, data, nxRestricted, nyRestricted, nfPre, nxExtended, nyExtended)
    % filename is the path to the pvp file to be created.  If the file already exists it will be clobbered.
    %
    % data is a cell array representing the weights, in the same format returned by readpvpfile:
    %     data{k} is a structure with two fields, time and values.
    %     data{k}.time is the timestamp of frame k.
    %     data{k}.values is a cell structure, one element for each arbor.
    %     data{k}.values{a} contains patches for arbor a.
    %     data{k}.values{a}(x,y,f,p) is the weight at location (x,y,f)
    %         of data patch p, where k-1 is the local presynaptic extended neuron index.
    %         (note non-matlab/octave ordering of column and row.  Also note that the
    %         ordering x,y,f is different from PetaVision's f,x,y).
    %     Note that even if the patch is shrunken, it occupies a full-sized patch in
    %     data and in the pvpfile.
    %
    %     nxRestricted and nyRestricted are the dimensions of the presynaptic
    %         restricted layer.
    %     nfPre is the number of features in the presynaptic layer.
    %     nxExtended and nyExtended are the dimensions of the presynaptic
    %         extended layer.
    
    if ~ischar(filename) || ~isvector(filename) || size(filename,1)~=1
        error('writepvpweightfile:filenamenotstring', 'filename must be a string');
    end%if
    
    if ~iscell(data)
        error('writepvpweightfile:datanotcell', 'data must be a cell array, one element per frame');
    end%if
   
    if ~isvector(data)
        error('writepvpweightfile:datanotcellvector', 'data cell array must be a vector; either number of rows or number of columns must be one');
    end%if
    
    fid = fopen(filename, 'w');
    if fid < 0
        error('writepvpweightfile:fopenerror', 'unable to open %s', filename);
    end%if
    
    errorpresent = 0;
    hdr = zeros(26,1);
    for frameno=1:length(data)   % allows either row vector or column vector.  isvector(data) was verified above
        numarbors = numel(data{frameno}.values);
        arbor1size = size(data{frameno}.values{1});
        % Note: calculation of wmin and wmax is off because it runs over the full patch instead of the shrunken patch
        wmax = max(data{frameno}.values{1}(:)); % The other cells in the values cell array are handled below.
        wmin = min(data{frameno}.values{1}(:));
        for arbor=2:numarbors
            arborsize = size(data{frameno}.values{arbor});
            if ~isequal(arborsize,arbor1size)
                msgid='valuescellsunequalsizes';
                errmsg=sprintf('Frame %d has different sizes for cells and %d', frameno, arbor);
                errorpresent=1;
                break;
            end%if
            wmaxcell = max(data{frameno}.values{arbor}(:));
            wmax = max(wmax,wmaxcell);
            wmincell = min(data{frameno}.values{arbor}(:));
            wmin = min(wmin,wmincell);
        end%for
        if errorpresent, break; end%if
        if numel(arbor1size)<4
            arbor1size=[arbor1size ones(1,4-numel(arbor1size))];
        end%if
        nxp = arbor1size(1);
        nyp = arbor1size(2);
        nfp = arbor1size(3);
        numpatches = arbor1size(4);
        if numpatches ~= nxExtended*nyExtended*nfPre
            msgid='numpatcheserror';
            errormsg=sprintf('Frame %d number of patches %d is inconsistent with presynaptic extended dimensions %d-by-%d-by-%d\n',
                frameno,numpatches,nxExtended,nyExtended,nfPre);
            errorpresent=1;
            break;
        end%if
        % Each frame has its own header
        hdr(1) = 104; % Number of bytes in header
        hdr(2) = 26;  % Number of 4-byte values in header
        hdr(3) = 3;   % File type for non
        hdr(4) = nxRestricted; % Presynaptic nx
        hdr(5) = nyRestricted; % Presynaptic ny
        hdr(6) = nfPre; % Presynaptic nf
        hdr(7) = numarbors; % Number of records, for weight files one arbor is a record.
        hdr(8) = 0;   % Weight pvp files do not use the record size header field
        hdr(9) = 4;   % Data size: floats are four bytes
        hdr(10) = 3;  % Types are 1=byte, 2=int, 3=float.  Type 1 is for compressed weights; type 2 isn't used for weight files
        hdr(11) = 1;  % Number of processes in x-direction. nxprocs is no longer used.
        hdr(12) = 1;  % Number of processes in y-direction. nyprocs is no longer used.
        hdr(13) = nxExtended;  % Presynaptic nxGlobal
        hdr(14) = nyExtended;  % Presynaptic nyGlobal
        hdr(15) = 0;  % kx0, no longer used
        hdr(16) = 0;  % ky0, no longer used
        hdr(17) = 1;  % Number of batch elements is 1 for weight files
        hdr(18) = numarbors; % number of arbors
        hdr(19:20) = typecast(double(data{frameno}.time),'uint32'); % timestamp
        % hdr(21:26) is for extra header values used by weights
        hdr(21) = nxp;% Size of full patches in the x-direction
        hdr(22) = nyp;% Size of full patches in the y-direction
        hdr(23) = nfp;% Number of features in the connection
        hdr(24) = typecast(single(wmin),'uint32'); % Minimum of all weights over all arbors in this frame.
        hdr(25) = typecast(single(wmax),'uint32'); % Maximum of all weights over all arbors in this frame.
        hdr(26) = numpatches; % Number of weight patches in one arbor
        fwrite(fid,hdr,'uint32');
        for arbor=1:numarbors
            patchno=0;
            for y=1:nyExtended
                for x=1:nxExtended
                    for f=1:nfPre
                        patchno=patchno+1;
                        fwrite(fid,nxp,'uint16');
                        fwrite(fid,nyp,'uint16');
                        fwrite(fid,0,'uint32');
                        fwrite(fid,permute(data{frameno}.values{arbor}(:,:,:,patchno),[3 1 2]),'single');                       
                    end%for
                end%for
            end%for
            assert(patchno==numpatches);
        end%for
    end%for
    
    fclose(fid); clear fid;
    
    if errorpresent
        error(errormsg); % Why doesn't octave's error command handle msgid, while matlab gives a warning if you don't use it?
    end%if
    
end%function
