function [A,T] = readactivities(activityfile)
% [A,T] = readactivities(activityfile)
%
% Reads a PetaVision-generated activity file (a*.pvp)
% activityfile is the filename
% A is an i-by-j-by-n array.  A(:,:,n) gives the activity at timestep n
% T is an n-by-1 vector giving the time corresponding to timestep n

fid = fopen(activityfile);
if fid == -1
    error('readactivities:cantopenfile','Can''t open %s',activityfile);
end

hdr = fread(fid,20,'int32');

nx = hdr(4);
ny = hdr(5);
nf = hdr(6);

sizeofframe = nx*ny*nf*4+8;
% each frame contains an nx-by-ny-by-nf array of 4-byte floats, with an
% 8-byte header indicating the time as a double-precision value.

fseek(fid,0,'eof');
eof = ftell(fid);

numberofframes = (eof-80)/sizeofframe;

if numberofframes ~= round(numberofframes)
    fclose(fid);
    error('readactivity:weirdsize',...
          ['File %s does not have the right file length for ' ...
           'an integral number of time steps'],...
          activityfile);
end

A = zeros(nx,ny,nf,numberofframes,'single');
T = zeros(numberofframes,1);

for t=1:numberofframes
    startofframe = 80+(t-1)*sizeofframe;
    fseek(fid,startofframe,'bof');
    T(t) = fread(fid,1,'float64');
    R = fread(fid,nx*ny*nf,'float32');
    A(:,:,:,t) = permute(reshape(R,[nf,nx,ny]), [2 3 1]);
    disp(sprintf('%d of %d',t,numberofframes));
end

fclose(fid);

