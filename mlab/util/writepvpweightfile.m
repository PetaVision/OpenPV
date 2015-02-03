function writepvpweightfile(filename, data, nxGlobalPre, nyGlobalPre, nfPre, nbPre, nxGlobalPost, nyGlobalPost, postweightsflag)
    % Usage: writepvpweightfile(filename, data, nxGlobalPre, nyGlobalPre, nfPre, nbPre, nxGlobalPost, nyGlobalPost, postweightsflag)
    % filename is the pvp file to be created.  If the file already
    % exists it will be clobbered.
    %
    % data is a cell array representing the weights, in the same
    % format returned by readpvpfile:
    %
    % data{k} is a structure with two fields, time and values.
    % data{k}.time is the timestamp of frame k.
    % data{k}.values is a 3-D cell structure.
    % data{k}.values{p,q,r} contains patches for the mpi process with commColumn()==p-1
    % and commRow()==q-1, and arbor r (note non-matlab/octave ordering of column and row)
    % data{k}.values{p,q,r}(x,y,f,p) is the weight at location (x,y,f)
    % of data patch p, where k-1 is the local presynaptic extended neuron index.
    % (again, note non-matlab/octave ordering of column and row.  Also note that the
    % ordering x,y,f is different from PetaVision's f,x,y.
    % (the minus ones are because matlab/octave is 1-indexed and PetaVision is zero-indexed.
    % Note that even if the patch is shrunken, it occupies a full-sized patch in data and in the pvpfile.
    % readpvpfile() fills field names data{k}.nx, data{k}.ny, and data{k}.offset.  However,
    % writepvpweightfile does not use those fields if they exist, but instead computes them
    % the way petavision does.
    %
    % nxGlobalPre, nyGlobalPre, nfPre, nbPre are the dimensions of the presynaptic layer
    % in global coordinates.  nxGlobalPre must be an integer multiple of commColumns() and
    % and nyGlobalPre must be an integer multiple of commRows().
    % nxGlobalPost and nyGlobalPost are the dimensions of the postsynaptic layer in global
    % coordinates.  nfGlobalPost is not needed since it's the same as the connection's nfp,
    % the size of data{k}.values{p,q,r} in the third dimension. nbGlobalPost is not needed
    % because weights map onto the post-synaptic restricted layer.
    %
    % postweightsflag is a boolean argument that tells whether the weights are postweights, i.e. the weight patches are nonshrunken.
    % If postweights is true, nbPre must be zero.
    %  
    
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
        numxprocs = size(data{frameno}.values,1);
        numyprocs = size(data{frameno}.values,2);
        numarbors = size(data{frameno}.values,3);
        numcells = numxprocs*numyprocs*numarbors;
        assert(numcells==numel(data{frameno}.values)); % will fail if data{frameno}.values is N-dimensional, N>=4
        if mod(nxGlobalPre,numxprocs) ~= 0
            msgid='nxglobalprenotmultiple';
            errmsg=sprintf('Frame %d: number of columns of processes %d is not a divisor of nxGlobalPre %d', numxprocs, nxGlobalPre);
            errorpresent=1;
        end%if
        if mod(nyGlobalPre,numyprocs) ~= 0
            msgid='nyglobalprenotmultiple';
            errmsg=sprintf('Frame %d: number of rows of processes %d is not a divisor of nyGlobalPre %d', numyprocs, nyGlobalPre);
            errorpresent=1;
        end%if
        if mod(nxGlobalPost,numxprocs) ~= 0
            msgid='nxglobalpostnotmultiple';
            errmsg=sprintf('Frame %d: number of columns of processes %d is not a divisor of nxGlobalPost %d', numxprocs, nxGlobalPost);
            errorpresent=1;
        end%if
        if mod(nyGlobalPre,numyprocs) ~= 0
            msgid='nyglobalpostnotmultiple';
            errmsg=sprintf('Frame %d: number of rows of processes %d is not a divisor of nyGlobalPre %d', numyprocs, nyGlobalPost);
            errorpresent=1;
        end%if
        if errorpresent
            break;
        end%if
        nxLocalPre = nxGlobalPre/numxprocs;
        nyLocalPre = nyGlobalPre/numyprocs;
        nxLocalPreExt = nxLocalPre+2*nbPre;
        nyLocalPreExt = nyLocalPre+2*nbPre;
        nxLocalPost = nxGlobalPost/numxprocs;
        nyLocalPost = nyGlobalPost/numyprocs;
        cell1size = size(data{frameno}.values{1,1,1});
        wmax = max(data{frameno}.values{1,1,1}(:)); % The other cells in the values cell array are handled below.
        wmin = min(data{frameno}.values{1,1,1}(:));
        for cell=2:numcells
            cellsize = size(data{frameno}.values{cell});
            if ~isequal(cellsize,cell1size)
                msgid='valuescellsunequalsizes';
                [p,q,r] = ind2sub([numxprocs,numyprocs,numarbors],cell);
                errmsg=sprintf('Frame %d has different sizes for cells {1,1,1} and {p,q,r}', frameno, arbor);
                errorpresent=1;
                break;
            end%if
            wmaxcell = max(data{frameno}.values{cell}(:));
            wmax = max(wmax,wmaxcell);
            wmincell = min(data{frameno}.values{cell}(:));
            wmin = min(wmin,wmincell);
        end%for
        if errorpresent, break; end%if
        if numel(cell1size<4)
            cell1size=[cell1size ones(1,4-numel(cell1size))];
        end%if
        nxp = cell1size(1);
        nyp = cell1size(2);
        nfp = cell1size(3);
        numpatches = cell1size(4);
        if numpatches ~= (nxLocalPreExt)*(nyLocalPreExt)*nfPre
            msgid='numpatcheserror';
            errormsg=sprintf('Frame %d number of patches %d is inconsistent with local presynaptic extended dimensions %d,%d,%d,%d\n',
                frameno,numpatches,nxLocalPre,nyLocalPre,nfPre,nbPre);
            errorpresent=1;
            break;
        end%if
        % Each frame has its own header
        hdr(1) = 104; % Number of bytes in header
        hdr(2) = 26;  % Number of 4-byte values in header
        hdr(3) = 3;   % File type for non
        hdr(4) = nxLocalPre;   % Presynaptic nx
        hdr(5) = nyLocalPre;   % Presynaptic ny
        hdr(6) = nfPre; % Presynaptic nf
        hdr(7) = numcells;   % Number of records, for weight files one cell is a record.
        hdr(8) = numpatches*(8+4*nxp*nyp*nfp);   % Record size, the data for the arbor is preceded by nx(2 bytes), ny(2 bytes) and offset(4 bytes)
        hdr(9) = 4;   % Data size: floats are four bytes
        hdr(10) = 3;  % Types are 1=byte, 2=int, 3=float.  Type 1 is for compressed weights; type 2 isn't used for weight files
        hdr(11) = numxprocs;  % Number of processes in x-direction
        hdr(12) = numyprocs;  % Number of processes in y-direction
        hdr(13) = nxGlobalPre;  % Presynaptic nxGlobal
        hdr(14) = nyGlobalPre;  % Presynaptic nyGlobal
        hdr(15) = 0;  % kx0, no longer used
        hdr(16) = 0;  % ky0, no longer used
        hdr(17) = nbPre;  % Presynaptic nb
        hdr(18) = numarbors; % number of arbors
        hdr(19:20) = typecast(double(data{frameno}.time),'uint32'); % timestamp
        % hdr(21:26) is for extra header values used by weights
        hdr(21) = nxp;% Size of full patches in the x-direction
        hdr(22) = nyp;% Size of full patches in the y-direction
        hdr(23) = nfp;% Number of features in the connection
        hdr(24) = typecast(single(wmin),'uint32'); % Minimum of all weights over all arbors in this frame.
        hdr(25) = typecast(single(wmax),'uint32'); % Maximum of all weights over all arbors in this frame.
        hdr(26) = numpatches*numcells; % Number of weight patches
        fwrite(fid,hdr,'uint32');
        if exist('postweightsflag','var') && postweightsflag
            if nbPre ~= 0
                msgid='postweightswithnb';
                errmsg=sprintf('If postweightsflag is set, nbPre must be zero (was %d)', nbPre);
                break;
            end%if
            nx=repmat(nxp,1,nxLocalPreExt);
            ny=repmat(nyp,1,nyLocalPreExt);
            offset=zeros(nxLocalPreExt,nyLocalPreExt);
        else
            [nx,ny,offset] = patchgeometry(nxLocalPre, nyLocalPre, nbPre, nxLocalPost, nyLocalPost, nxp, nyp, nfp);
        end%if
        for cell=1:numcells
            patchno=0;
            for y=1:nyLocalPre+2*nbPre
                for x=1:nxLocalPre+2*nbPre
                    for f=1:nfPre
                        patchno=patchno+1;
                        fwrite(fid,nx(x),'uint16');
                        fwrite(fid,ny(y),'uint16');
                        fwrite(fid,offset(x,y),'uint32');
                        fwrite(fid,permute(data{frameno}.values{cell}(:,:,:,patchno),[3 1 2]),'single');                       
                    end%for
                end%for
            end%for
            assert(patchno==numpatches);
            for patchno=1:numpatches
            end%for
        end%for
    end%for
    
    fclose(fid); clear fid;
    
    if errorpresent
        error(errormsg); % Why doesn't octave's error command handle msgid, while matlab gives a warning if you don't use it?
    end%if
    
end%function

function [nx,ny,offset] = patchgeometry(nxLocalPre,nyLocalPre,nbPre,nxLocalPost,nyLocalPost,nxp,nyp,nfp)
    nxLocalPreExt = nxLocalPre+2*nbPre;
    nyLocalPreExt = nyLocalPre+2*nbPre;
    offset = repmat(0, nxLocalPreExt, nyLocalPreExt);
    manytoonefactorx = max(nxLocalPre/nxLocalPost,1); % should do error checking since this should be an integer
    manytoonefactory = max(nyLocalPre/nyLocalPost,1);
    onetomanyfactorx = max(nxLocalPost/nxLocalPre,1);
    onetomanyfactory = max(nyLocalPost/nyLocalPre,1);

    [offsety,ny] = patchgeometryonedimension(nyLocalPreExt, nbPre, nyLocalPost, nyp);
    [offsetx,nx] = patchgeometryonedimension(nxLocalPreExt, nbPre, nxLocalPost, nxp);
    for y=1:nyLocalPreExt
        for x=1:nxLocalPreExt
            offset(x,y) = (offsety(y)*nxp+offsetx(x))*nfp; % can't use sub2ind because offsetx(x) could be 0 through nxp
        end%for
    end%for
end%function

function [start,width] = patchgeometryonedimension(nLocalPreExt,nb,nLocalPostRes,patchwidth)
    nLocalPreRes = nLocalPreExt-2*nb;
    points = (-nb:nLocalPreRes+nb-1); % zero-indexed coordinates in restricted space of extended neurons
    
    postFactor = max(nLocalPostRes/nLocalPreRes,1); % number of post-synaptic neurons in a unit cell
    preFactor = max(nLocalPreRes/nLocalPostRes,1); % number of pre-synaptic neurons in a unit cell
    assert(postFactor==round(postFactor));
    assert(preFactor==round(preFactor));
    radius = (patchwidth/postFactor - 1)/2; % measured in unit cells
    assert(radius==round(radius));
    start = postFactor*((-points-preFactor+1)/preFactor+radius);
    start = max(start, 0);
    start = min(start, nLocalPostRes);
    start = floor(start);
    width = points/preFactor+radius+1;
    width = min(width,width(end:-1:1));
    width = max(width,0);
    width = min(width, patchwidth);
    width = min(width, nLocalPostRes);
    width = postFactor*floor(width);
    assert(all(start+width>=0 & start+width<=patchwidth));
end%function
