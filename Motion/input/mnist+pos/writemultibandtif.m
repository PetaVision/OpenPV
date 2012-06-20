function filesize = writemultibandtif(data,filename)
% filesize = writemultibandtiff(data,filename)
% data is an 3-D array with features in the third dimension
% If data is of class single or double, it should be in the range [0,1].
% It is then stored as a 16-bit unsigned tiff file, with 0 and below
% mapping to 0, and 1 and above to 65535.
% If data is integral, it is converted to a 16-bit unsigned tiff file
% Values below zero are mapped to zero and values above 65535 are
% mapped to 65535.
%
% filename is--surprise!--the filename to write to.
%
% filesize is the size of the resulting file in bytes

if ~isnumeric(data)
    error(sprintf('%s:notnumeric',mfilename),...
          '%s:data must be numeric',mfilename);
end

if all(data(:)==round(data(:)))
    data = max(0,min(65535,data));
else
    data = max(0,min(65535,round(data*65535)));
end

[m,n,f] = size(data);
datapermute = permute(data,[3,2,1]);

if f==1
    numifds = 10;
else
    numifds = 11;
end

ifds = struct('tag',cell(numifds,1),'type',cell(numifds,1),...
              'count',cell(numifds,1),'usesoffset',cell(numifds,1),...
              'value',cell(numifds,1),'offsetdata',cell(numifds,1));
dataaddress = 8;
ifd0address = dataaddress + 2*m*n*f;
currentoffset = ifd0address+2+numifds*12+4;

thisifd = 0;

% ImageWidth
thisifd = thisifd + 1;
ifds(thisifd).tag = 256;
ifds(thisifd).type = 4; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = 1;
ifds(thisifd).usesoffset = false;
ifds(thisifd).value = n; % Since this is matlab, size(data,1) is height

% ImageLength
thisifd = thisifd + 1;
ifds(thisifd).tag = 257;
ifds(thisifd).type = 4; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = 1;
ifds(thisifd).usesoffset = false;
ifds(thisifd).value = m; % Since this is matlab, size(data,2) is width

% BitsPerSample
thisifd = thisifd + 1;
ifds(thisifd).tag = 258;
ifds(thisifd).type = 3; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = f;
if f>2
ifds(thisifd).usesoffset = true;
    ifds(thisifd).value = currentoffset;
    ifds(thisifd).offsetdata = repmat(16,f,1);
    currentoffset = currentoffset + 2*f;
else
    ifds(thisifd).usesoffset = false;
    ifds(thisifd).value = repmat(16,f,1);
end

% Compression
thisifd = thisifd + 1;
ifds(thisifd).tag = 259;
ifds(thisifd).type = 3; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = 1;
ifds(thisifd).usesoffset = false;
ifds(thisifd).value = 1; % Uncompressed

% PhotometricInterpretation
thisifd = thisifd + 1;
ifds(thisifd).tag = 262;
ifds(thisifd).type = 3; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = 1;
ifds(thisifd).usesoffset = false;
ifds(thisifd).value = 1; % BlackIsZero

% StripOffsets
thisifd = thisifd + 1;
ifds(thisifd).tag = 273;
ifds(thisifd).type = 4; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = m; % For simplicity, use one row per strip
if m<=1
    ifds(thisifd).usesoffset = false;
    ifds(thisifd).value = dataaddress;
    
else
    ifds(thisifd).usesoffset = true;
    ifds(thisifd).value = currentoffset;
    ifds(thisifd).offsetdata = dataaddress+2*n*f*(0:m-1)';
    currentoffset = currentoffset + 4*m;
end

% SamplesPerPixel
thisifd = thisifd + 1;
ifds(thisifd).tag = 277;
ifds(thisifd).type = 3; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = 1;
ifds(thisifd).usesoffset = false;
ifds(thisifd).value = f;

% RowsPerStrip
thisifd = thisifd + 1;
ifds(thisifd).tag = 278;
ifds(thisifd).type = 4; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = 1;
ifds(thisifd).usesoffset = false;
ifds(thisifd).value = 1;

% StripByteCounts
thisifd = thisifd + 1;
ifds(thisifd).tag = 279;
ifds(thisifd).type = 4; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = m;
if m <= 1
    ifds(thisifd).usesoffset = false;
    ifds(thisifd).value = 2*n*f;
else
    ifds(thisifd).usesoffset = true;
    ifds(thisifd).value = currentoffset;
    ifds(thisifd).offsetdata = repmat(2*f*n,m,1);
    currentoffset = currentoffset + 4*m;
end

% ExtraSamples
if f>1
    thisifd = thisifd + 1;
    ifds(thisifd).tag = 338;
    ifds(thisifd).type = 3; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
    ifds(thisifd).count = f-1;
    if f-1 <= 2
        ifds(thisifd).usesoffset = false;
        ifds(thisifd).value = zeros(f-1,1); % 0=unspecified data
    else
        ifds(thisifd).usesoffset = true;
        ifds(thisifd).value = currentoffset;
        ifds(thisifd).offsetdata = zeros(f-1,1); % 0=unspecified data
        currentoffset = currentoffset + 2*(f-1);
    end
end

% SampleFormat
thisifd = thisifd + 1;
ifds(thisifd).tag = 339;
ifds(thisifd).type = 3; % 1=byte,2=ascii,3=uint16,4=uint32,5=rational
ifds(thisifd).count = f;
if f <= 2
    ifds(thisifd).usesoffset = false;
    ifds(thisifd).value = ones(f,1); % 1=unsigned integer data
else
    ifds(thisifd).usesoffset = true;
    ifds(thisifd).value = currentoffset;
    ifds(thisifd).offsetdata = ones(f,1); % 1=unsigned integer data
    currentoffset = currentoffset + 2*f;
end

assert(thisifd == numifds);
fid = fopen(filename,'w');
if fid < 0
    error(sprintf('%s:badfile',mfilename),...
          '%s:File %s could not be created for writing',...
          mfilename,filename);
end
fwrite(fid,'II','char*1'); % Use little-endian encoding
fwrite(fid,42,'uint16',0,'l'); % Tiff version identifier
fwrite(fid,ifd0address,'uint32',0,'l');  % Offset of first IFD
fwrite(fid,datapermute,'uint16',0,'l'); % The actual data
fwrite(fid,numifds,'uint16',0,'l'); % Number of IFDs
for thisifd=1:numifds
    fwrite(fid,ifds(thisifd).tag,'uint16',0,'l'); % directory entry tag
    fwrite(fid,ifds(thisifd).type,'uint16',0,'l'); % directory entry type
    fwrite(fid,ifds(thisifd).count,'uint32',0,'l'); % directory enttry count
    if ifds(thisifd).usesoffset
        fwrite(fid,ifds(thisifd).value,'uint32',0,'l');
    else
        n = numel(ifds(thisifd).count);
        switch ifds(thisifd).type
            case 1 % byte
                capacity=4; dtype='uint8';
            case 2 % ascii
                capacity=4; dtype='char*1';
            case 3 % uint16
                capacity=2; dtype='uint16';
            case 4 % uint32
                capacity=1; dtype='uint32';
            case 5 % rational
                capacity=-1; % rationals are 8 bytes and can't fit
            otherwise
                capacity=-1; % other datatypes might be added someday
        end
        assert(n<=capacity);
        excess = capacity-n;
        v = [ifds(thisifd).value(:);zeros(excess,1)];
        fwrite(fid,v,dtype,0,'l');
    end
end

fwrite(fid,[0;0;0;0],'uint8',0,'l');

for thisifd=1:numifds
    if ifds(thisifd).usesoffset
        switch ifds(thisifd).type
            case 1 % byte
                dtype='uint8';
            case 2 % ascii
                dtype='char*1';
            case 3 % uint16
                dtype='uint16';
            case 4 % uint32
                dtype='uint32';
            case 5 % rational
                dtype='uint32'; % Rationals should have been stored as pairs of uint32s
            otherwise
                assert(0); % other datatypes might be added someday
        end
        fwrite(fid,ifds(thisifd).offsetdata,dtype,0,'l');
    end
end

filesize = currentoffset;
fclose(fid); clear fid;
