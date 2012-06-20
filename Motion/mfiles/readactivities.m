function [A,T] = readactivities(activityfile, progressupdateperiod)
% [A,T] = readactivities(activityfile, progressflag)
%
% Reads a PetaVision-generated activity file (a*.pvp)
% activityfile is the filename
% progressupdateperiod is a scalar that indicates how often to output progress.
%              Default is 0 (no output).
% A is an i-by-j-by-n array.  A(:,:,n) gives the activity at timestep n
% T is an n-by-1 vector giving the time corresponding to timestep n

if ~exist('progressupdateperiod','var')
    progressupdateperiod = 0;
end

if ~isscalar(progressupdateperiod)
    error('readactivities:progressupdateperiodscalar',...
          'progressupdateperiod must be a scalar');
end

if progressupdateperiod < 0
    progressupdateperiod = 0;
end

if progressupdateperiod ~= round(progressupdateperiod)
    error('readactivities:progressupdateperiodinteger',...
          'progressupdateperiod must be a nonnegative integer');
end

fid = fopen(activityfile);
if fid == -1
    error('readactivities:cantopenfile','Can''t open %s',activityfile);
end

hdr = fread(fid,20,'int32');

nx = hdr(4);
ny = hdr(5);
nf = hdr(6);

nxglob = hdr(13);
nyglob = hdr(14);

nxprocs = nxglob/nx;
nyprocs = nyglob/ny;

localsize = hdr(8);
sizeofframe = nxglob*nyglob*nf*4+8;
% each frame contains an nx-by-ny-by-nf array of 4-byte floats, with an
% 8-byte header indicating the time as a double-precision value.

fseek(fid,0,'eof');
eof = ftell(fid);

numberofframes = (eof-80)/sizeofframe;

% if numberofframes ~= round(numberofframes)
%     fclose(fid);
%     error('readactivity:weirdsize',...
%           ['File %s does not have the right file length for ' ...
%            'an integral number of time steps'],...
%           activityfile);
% end%if numberofframes

A = zeros(nxglob,nyglob,nf,numberofframes,'single');
T = zeros(numberofframes,1);

for t=1:numberofframes
    startofframe = 80+(t-1)*sizeofframe;
    fseek(fid,startofframe,'bof');
    T(t) = fread(fid,1,'float64');
    for y_proc = 1 : nyprocs
        for x_proc = 1 : nxprocs
            R = fread(fid,nx*ny*nf,'float32');
            A(nx*(x_proc-1)+1:nx*x_proc, ny*(y_proc-1)+1:ny*y_proc,:,t) = permute(reshape(R,[nf,nx,ny]), [2 3 1]);
        end
    end
    if progressupdateperiod > 0 && ~mod(t,progressupdateperiod)
        fprintf(1,'%d of %d\n',t,numberofframes);
    end
end%for t

fclose(fid);

