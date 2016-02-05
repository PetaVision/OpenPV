function [confidenceTable, truesTable, falsesTable] = createConfidenceTable(groundTruth, reconstruction, numPoints)
% confidenceTable = createConfidenceTable(groundTruth, reconstruction, numPoints)
%
% Inputs:
%     groundTruth is a 4-D array containing ground truth (ones or zeros).
%     reconstruction is a 4-D array containing reconstructions of the
%         ground truth.  It should be the same size as groundTruth.
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

szg = size(groundTruth); szg(end+1:4)=1;
szr = size(reconstruction); szr(end+1:4)=1;
if szg(4)~=szr(4)
   warning('createConfidenceTable:arraysincompatible','createConfidenceTable warning: groundTruth array and reconstruction array do not have the same number of frames (%d vs. %d).', szg(4), szr(4));
   warning('createConfidenceTable will attempt to process up to the last congruent frame. This might not be what was intended.');
end%if

if ~isequal(szg(1:3), szr(1:3))
    error('createConfidenceTable:baddimensions','createConfidenceTable error: groundTruth and reconstruction must have the same dimensions (%d-by-%d-by-%d versus %d-by-%d-by-%d)', szg(1), szg(2), szg(3), szr(1), szr(2), szr(3));
end%if

nx = szg(1);
ny = szg(2);
nf = szg(3);

reconmin = min(reconstruction(:));
reconmax = max(reconstruction(:));
confidenceTable = zeros(numPoints+1, nf+1);
confidenceTable(:,nf+1) = linspace(reconmin, reconmax, numPoints+1);
truesTable = zeros(numPoints+1, nf+1);
truesTable(:,end) = confidenceTable(:,end);
falsesTable = zeros(numPoints+1, nf+1);
falsesTable(:,end) = confidenceTable(:,end);
maxConfidence = zeros(1,nf);

for feature=1:nf
    trues = reconstruction(:,:,feature,:)(groundTruth(:,:,feature,:)~=0);
    falses = reconstruction(:,:,feature,:)(groundTruth(:,:,feature,:)==0);
    for k=1:numPoints+1;
        m=confidenceTable(k,end);
        T = sum(trues>=m)/(numel(trues)+isempty(trues));
        F = sum(falses>=m)/(numel(falses)+isempty(falses));
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
