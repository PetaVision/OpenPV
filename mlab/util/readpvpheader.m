function hdr = readpvpheader(file,pos)
% hdr = readpvpheader(file,pos)
% file specifies the pvp file.
%     If fid is an integer, it is the file id of a file open for reading.
%     If file is a string, it refers to a path to the pvpfile. 
% pos gives the file position.  If absent, use the current position.
%     If a nonnegative integer, use that position, measured forward from the
%     beginning of the file. It is an error to use a negative position.
%
% If the file was specified by file id, the file remains open and when the
% routine exits, the position (as returned by ftell) is the end of the header.
% If the file was specified by path, the file is opened at the start of the
% routine, and closed at the end of the routine.
%
% It is an error to use a negative position or a position close enough to
% the end of the file that there isn't room for the header, although the
% routine is too lazy to check.
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

if pos < 0
    error('readpvpheader:negpos', 'readpvpheader error: pos argument must be nonnegative.');
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
fileend = ftell(fid);
status = fseek(fid, pos, 'bof');
if status ~= 0, error('readpvpheader:seekeof', 'readpvpheader error: seeking to position %d failed.', pos); end

if 80 > fileend-pos
    error('readpvpheader:toofewparams', 'readpvpheader error: File is not long enough to contain a pvp header starting at position %d.', pos);
end%if

headerwords = fread(fid,18,'int32');
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
hdr.time = fread(fid,1,'float64');

if hdr.numparams>20
    if 4*hdr.numparams > fileend-pos
        error('readpvpheader:toomanyparams', 'readpvpheader error: numparams %d is too large for the given file.', hdr.numparams);
    end%if
    hdr.additional = fread(fid,hdr.numparams-20,'int32');
end

if ischar(file) % If we opened the file, close it; otherwise, leave it alone.
    fclose(fid);
end%if
