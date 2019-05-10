function hdr = readpvpheader(file,pos)
% hdr = readpvpheader(file,pos)
% file specifies the pvp file.
%     If fid is an integer, it is the file id of a file open for reading.
%     If file is a string, it refers to a path to the pvpfile. 
% pos gives the file position.  If absent, use the current position.
%     If a nonnegative integer, use that position, measured forward from the
%     beginning of the file. If a negative integer, use that position, measured
%     backward from the end of the file.
%
% If the file was specified by file id, the file remains open and when the
% routine exits, the position (as returned by ftell) is the end of the header.
% If the file was specified by path, the file is opened at the start of the
% routine, and closed at the end of the routine.
%
% It is an error to use a position close enough to the end of the file that
% there isn't room for the header.
%
% hdr is a struct whose fields are:
%     headersize
%     numparams
%     filetype
%     nx
%     ny
%     nf
%     numrecords
%     recordsize
%     datasize
%     datatype
%     nxprocs
%     nyprocs
%     nxExtended
%     nyExtended
%     kx0
%     ky0
%     nb
%     nbands
%     time
%
% If hdr.numparams is bigger than 20, there is a field 'additional'
% containing an vector of hdr.numparams-20 elements.

if ~exist('pos','var')
   pos = 0;
end%if

% Is file a file id (integer) or path (string)?
if ischar(file)
    fid = fopen(file, 'r');
elseif isnumeric(file) && isscalar(file) && round(file)==file
    fid = double(file);
else
    error('readpvpheader:filebad', 'readpvpheader error: file must be either a path or a file id.');
end%if

if !is_valid_file_id(fid), error('readpvpheader:badfid', 'readpvpheader error: bad file id.'); end;
% This checks if file id is valid, but not whether the mode allows reading.

status = fseek(fid, 0, 'eof');
if status ~= 0, error('readpvpheader:seekeof', 'readpvpheader error: seeking to end of file failed.'); end
if (pos < 0)
    status = fseek(fid, pos, 'eof');
else
    status = fseek(fid, pos, 'bof');
end%if
if status ~= 0, error('readpvpheader:seekeof', 'readpvpheader error: seeking to position %d failed.', pos); end

headerwords = fread(fid,20,'int32');
if numel(headerwords) < 20
    error('readpvpheader:toomanyparams', 'readpvpheader error: numparams is %d but end of file reached before %d parameters could be read.', hdr.numparams, hdr.numparams);
end%if
hdr.headersize = headerwords(1);
hdr.numparams = headerwords(2);
hdr.filetype = headerwords(3);
hdr.nx = headerwords(4);
hdr.ny = headerwords(5);
hdr.nf = headerwords(6);
hdr.numrecords = headerwords(7);
hdr.recordsize = headerwords(8);
hdr.datasize = headerwords(9);
hdr.datatype = headerwords(10);
hdr.nxprocs = headerwords(11);
hdr.nyprocs = headerwords(12);
hdr.nxExtended = headerwords(13);
hdr.nyExtended = headerwords(14);
hdr.kx0 = headerwords(15);
hdr.ky0 = headerwords(16);
hdr.nbatch = headerwords(17);
hdr.nbands = headerwords(18);
hdr.time = typecast(int32(headerwords(19:20)), 'double');

if hdr.numparams>20
    hdr.additional = fread(fid,hdr.numparams-20,'int32');
    if numel(hdr.additional) < hdr.numparams-20
        error('readpvpheader:toomanyparams', 'readpvpheader error: numparams is %d but end of file reached before %d parameters could be read.', hdr.numparams, hdr.numparams);
    end%if

    if hdr.filetype == 3 || hdr.filetype == 5 % weight file type or shared-weight file type
        additional = hdr.additional;
        if (numel(additional) < 6)
           error('readpvpheader:toofewwgtparams','readpvpheader error: filetype %d indicates a weight file but numparams %d is fewer than 26.', hdr.filetype, hdr.numparams);
        end%if
        hdr = rmfield(hdr, 'additional');
        hdr.nxp = additional(1);
        hdr.nyp = additional(2);
        hdr.nfp = additional(3);
        hdr.min = double(typecast(int32(additional(4)),'single'));
        hdr.max = double(typecast(int32(additional(5)),'single'));
        hdr.numpatches = additional(6);
        if numel(additional) > 6
            hdr.additional = additional(7:end);
        end%if
    end%if
end

if ischar(file) % If we opened the file, close it; otherwise, leave it alone.
    fclose(fid);
end%if
