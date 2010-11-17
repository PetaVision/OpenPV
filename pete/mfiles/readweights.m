function W = readweights(weightfile)
% W = readweights(weightfile)
%
% Reads the weights from a PetaVision-generated weight file
% weightfile is a filename.
% W is a 4-dimensional array.
% W(i,j,m,n) is the weight of the (i,j)th location for feature m
% and kernel patch n.

filedata = dir(weightfile);
if length(filedata) ~= 1
    error('readweights:notonefile',...
          'Path %s should expand to one file; in this case there are %d',...
          length(filedata));
end

if filedata(1).bytes < 104
    error('readweights:filetooshort',...
          'File %s is too short to contain a valid header',weightfile);
end%if numel

fid = fopen(weightfile);
if fid == -1
    error('readweights:cantopenfile','Can''t open %s',weightfile);
end

hdr = fread(fid, 20, 'int32');
nxp = fread(fid, 1, 'int32');
nyp = fread(fid, 1, 'int32');
nfp = fread(fid, 1, 'int32');
weight_min = fread(fid, 1, 'float32');
weight_max = fread(fid, 1, 'float32');
numpatches = fread(fid, 1, 'int32');
A = fread(fid);
fclose(fid);

if isempty(nxp) || isempty(nyp) || isempty(nfp) || ...
   isempty(weight_min) || isempty(weight_max) || isempty(numpatches)
    error('readweights:badheader',...
          'Unable to read header of file %s',weightfile);
end

if numpatches*(nxp*nyp*nfp+4) ~= numel(A);
    error('readweights:badlength',...
          'The length of %s is inconsistent with its header',weightfile);
end%if numpatches

W = zeros(nxp, nyp, nfp, numpatches);

for p=1:numpatches
    baseindex = (p-1)*(4+nxp*nyp*nfp)+4;
    R = reshape(A(baseindex+1:baseindex+nxp*nyp*nfp),[nfp,nxp,nyp]);
    for f=1:nfp
        W(:,:,f,p) = squeeze(R(f,:,:));
    end%for f
end%for p

W = W/255*(weight_max-weight_min)+weight_min;
