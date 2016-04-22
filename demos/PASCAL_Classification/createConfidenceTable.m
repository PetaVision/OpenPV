function [confidenceTable, truesTable, falsesTable] = createConfidenceTable(groundTruthPvpFile, reconstructionPvpFile, pv_dir, numPoints)
% confidenceTable = createConfidenceTable(groundTruthPvpFile, reconstructionPvpFile, numPoints)
%
% Inputs:
%     groundTruthPvpFile is a pvp file containing ground truth (ones or zeros).
%         Right now it's hardcoded to expect a sparse binary file (filetype=2)
%         This should change to be more flexible.
%     reconstructionPvpFile is a pvp file containing reconstructions of the
%         ground truth.  It should be a nonsparse file.
%     pv_dir is a directory containing readpvpfile.m.  It is only used if
%         readpvpfile is not already in the path.
%     numPoints is a positive integer, giving the number of reconstruction values.
%
% Output:
%     confidenceTable is a (numPoints+1)-by-(nf+1) array.
%         The last column is a vector of increasing values, whose first value is the
%         minimum of reconstructionPvpFile activities, and whose last value is the
%         maximum.  That is,
%             confidenceTable(:,end) = linspace(minimum,maximum,numPoints+1)
%         Each of the first nf columns are increasing vectors, where
%             confidenceTable(j,feature) is the confidence, expressed as a
%         percentage, that an image tile whose reconstruction of the given
%         feature activity is confidenceTable(j,end) actually has the feature.

if isempty(which('readpvpfile'))
   addpath(pv_dir);
end%if
if isempty(which('readpvpfile'))
   error('createConfidenceTable:missingreadpvpfile','createConfidenceTable error: no readpvpfile in either the path or the pv_dir input argument');
end%if

[groundTruthPvp,gthdr] = readpvpfile(groundTruthPvpFile);
[reconstructionPvp,reconhdr] = readpvpfile(reconstructionPvpFile);

if (numel(groundTruthPvp)==0)
    error('createConfidenceTable:emptygroundtruth','createConfidenceTable error: groundTruthPvpFile ''%s'' is empty', groundTruthPvpFile);
end%if
if (numel(reconstructionPvp)==0)
    error('createConfidenceTable:emptyreconstruction','createConfidenceTable error: reconstructionPvpFile ''%s'' is empty', reconstructionPvpFile);
end%if

if (numel(groundTruthPvp)~=numel(reconstructionPvp))
   warning('createConfidenceTable:pvpfilesincompatible','createConfidenceTable warning: groundTruthPvpFile ''%s'' and reconstructionPvpFile ''%s'' do not have the same number of frames', groundTruthPvpFile, reconstructionPvpFile);
   warning('createConfidenceTable will attempt to process up to the last congruent frame. This might not be what was intended.');
end%if

if (gthdr.filetype ~= 2)
    error('createConfidenceTable:badfiletype','createConfidenceTable error: groundTruthPvpFile ''%s'' must be a sparse binary file', groundTruthPvpFile);
end%if

if (reconhdr.filetype ~= 4)
    error('createConfidenceTable:badfiletype','createConfidenceTable error: reconstructionPvpFile ''%s'' must be a nonsparse file', reconstructionPvpFile);
end%if

if (gthdr.nx ~= reconhdr.nx || gthdr.ny ~= reconhdr.ny || gthdr.nf ~= reconhdr.nf)
    error('createConfidenceTable:badfiletype','createConfidenceTable error: groundTruthPvpFile ''%s'' and reconstructionPvpFile ''%s'' must have the same dimensions', groundTruthPvpFile, reconstructionPvpFile);
end%if

nx = gthdr.nx;
ny = gthdr.ny;
nf = gthdr.nf;

assert(isequal(size(reconstructionPvp{1}.values),[reconhdr.nx,reconhdr.ny,reconhdr.nf]));
lastframe = min(numel(reconstructionPvp), numel(groundTruthPvp));
reconfull = zeros(reconhdr.nx,reconhdr.ny,reconhdr.nf,lastframe);
groundtruthfull = zeros(gthdr.nx,gthdr.ny,gthdr.nf,lastframe);

for k=1:lastframe
    reconfull(:,:,:,k) = reconstructionPvp{k}.values;
    Z = zeros(gthdr.nf,gthdr.nx,gthdr.ny);
    Z(groundTruthPvp{k}.values+1) = 1;
    Zp = permute(Z,[2 3 1]);
    assert(isequal(size(Zp),[nx,ny,nf]));
    groundtruthfull(:,:,:,k)=Zp;
end%for

% groundtruthfull and reconfull are now directly comparable.

reconmin = min(reconfull(:));
reconmax = max(reconfull(:));
confidenceTable = zeros(numPoints+1, nf+1);
confidenceTable(:,nf+1) = linspace(reconmin, reconmax, numPoints+1);
truesTable = zeros(numPoints+1, nf+1);
truesTable(:,end) = confidenceTable(:,end);
falsesTable = zeros(numPoints+1, nf+1);
falsesTable(:,end) = confidenceTable(:,end);
maxConfidence = zeros(1,nf);

for feature=1:gthdr.nf
    trues = reconfull(:,:,feature,:)(groundtruthfull(:,:,feature,:)~=0);
    falses = reconfull(:,:,feature,:)(groundtruthfull(:,:,feature,:)==0);
    for k=1:numPoints+1;
        m=confidenceTable(k,end);
        T = sum(trues>=m);
        F = sum(falses>=m);
        truesTable(k,feature) = T;
        falsesTable(k,feature) = F;
        if (T==0 && F==0) || T/(T+F) < maxConfidence(feature);
            confidenceTable(k,feature) = maxConfidence(feature);
        else%if
            confpct = T/(T+F);
            maxConfidence(feature) = confpct;
            confidenceTable(k,feature) = confpct;
        end%if
    end%for
    fprintf('%d\n', feature); fflush(1);
end%for
