function W = readweights(weightfile)
% W = readweights(weightfile)
%
% Reads the weights from a PetaVision-generated weight file
% weightfile is a filename.
% W is a 4-dimensional array.
% W(i,j,m,n) is the weight of the (i,j)th location for feature m
% and kernel patch n.

fid = fopen(weightfile);
if fid == -1
    error('readweights:cantopenfile','Can''t open %s',weightfile);
end

A = fread(fid);
fclose(fid);

if numel(A)<104
    error('readweights:filetooshort',...
          'File %s is too short to contain a valid header',weightfile);
end%if numel

nxp = [1 2^8 2^16 2^24]*A(81:84);
nyp = [1 2^8 2^16 2^24]*A(85:88);
nfp = [1 2^8 2^16 2^24]*A(89:92);

numpatches = [1 2^8 2^16 2^24]*A(69:72);

if numpatches*(nxp*nyp*nfp+4)+104 ~= numel(A);
    error('readweights:badlength',...
          'The length of %s is inconsistent with its header',weightfile);
end%if numpatches

W = zeros(nxp, nyp, nfp, numpatches);

for p=1:numpatches
    baseindex = 104+(p-1)*(4+nxp*nyp*nfp)+4;
    R = reshape(A(baseindex+1:baseindex+nxp*nyp*nfp),[nfp,nxp,nyp]);
    for f=1:nfp
        W(:,:,f,p) = squeeze(R(f,:,:));
    end%for f
end%for p
