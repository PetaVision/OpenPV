function hdr = readpvpheader(file,pos)
% hdr = readpvpheader(file,pos)
% file specifies the pvp file.
%     If fid is an integer, it is the file id of a file open for reading.
%     If file is a string, it refers to a path to the pvpfile. 
% pos gives the file position.  If absent, use the current position.
%     If pos >= 0, count pos bytes forward from the beginning of the file.
%     If pos < 0, count -pos bytes backward from the end of the file.
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
% For weight files (filetype 3 or 5), hdr also has fields
%     nxp
%     nyp
%     nfp
%     wMin
%     wMax
%     numPatches
%
% If hdr.numparams is bigger than 20 for activity files or 26 for
% weight files, there is a field 'additional' containing a vector of
% the remaining parameters.

if ~exist('pos','var')
   pos = 0;
end%if

openedfile = false; % keep track of whether we opened the file and therefore need to close it.
% Is file a file id (integer) or path (string)?
if ischar(file)
    fid = fopen(file, 'r');
    if fid < 0, error('readpvpheader:openerror', 'readpvpheader error: unable to open %s', file); end
    openedfile = true;
elseif isnumeric(file) && isscalar(file) && round(file)==file
    fid = double(file);
    if !is_valid_file_id(fid), error('readpvpheader:badfid', 'readpvpheader error: bad file id.'); end;
    % This checks if file id is valid, but not whether the mode allows reading.
else
    error('readpvpheader:filebad', 'readpvpheader error: file must be either a path or a file id.');
end%if

errordetected = false;
try
    if (pos < 0)
        status = fseek(fid, pos, 'eof');
    else
        status = fseek(fid, pos, 'bof');
    end%if
    if status ~= 0
        error('readpvpheader:seekeof', 'readpvpheader error: seeking to position %d failed.', pos);
    end%endif

    headerwords = fread(fid,20,'int32');
    if numel(headerwords) < 20
        error('readpvpheader:fileshort', 'readpvpheader error: end of file reached before minimum header size of 80 bytes could be read.');
    end%if
    hdr.headersize = headerwords(1);
    hdr.numparams = headerwords(2);
    if hdr.headersize < 80 || hdr.headersize != 4 * hdr.numparams
        error('readpvpheader:badheadersize', 'readpvpheader error: headersize (%d) must be at least 80 and must be exactly 4 times numparams (%d).', hdr.headersize, hdr.numparams);
    end%if
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
            hdr.wMin = double(typecast(int32(additional(4)),'single'));
            hdr.wMax = double(typecast(int32(additional(5)),'single'));
            hdr.numPatches = additional(6);
            if numel(additional) > 6
                hdr.additional = additional(7:end);
            end%if
        end%if
    end
catch
    errordetected = true;
end_try_catch

if openedfile, fclose(fid); end % Close file if we opened it but not if we didn't

if errordetected
    error(lasterror.identifier, lasterror.message);
end%if
