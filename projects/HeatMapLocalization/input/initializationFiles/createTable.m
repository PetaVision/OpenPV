% This script m-file creates GroundTruth, ReconS1
% as full 4X3X21X7958 arrays;
% and confidenceTableS1 as a 101-by-22 array
% To change the value of 101, change numintervals.
% To change the value of 22, change numfeatures.
% (The confidence tables are (numintervals+1)-by-(numfeatures+1) arrays.
% See createConfidenceTable.m for an explanation of the extra feature and extra column.

numintervals = 100;
numfeatures = 21;
mask = [0; ones(20,1)]; % mask out background

if isempty(which('readpvpfile'))
   addpath(pv_dir);
end%if
if isempty(which('readpvpfile'))
   error('createThreeTables:missingreadpvpfile','createThreeTables error: no readpvpfile in either the path or the pv_dir variable');
end%if

if ~exist('binfiles', 'dir')
    mkdir binfiles;
end%if

if ~exist('binfiles', 'dir')
    error('createThreeTables:no_binfiles', 'createThreeTables error: unable to create binfiles directory');
end%if

pvpraw = readpvpfile('pvpFiles/GroundTruth.pvp');
N = numel(pvpraw);
GroundTruth = zeros(4,3,numfeatures,N);
for n=1:N
    if ~isempty(pvpraw{n})
        Z = zeros(numfeatures,4,3);
        Z(pvpraw{n}.values+1) = 1;
        GroundTruth(:,:,:,n) = permute(Z,[2 3 1]);
    end%if
end%for

clear Z;

pvpraw = readpvpfile('pvpFiles/GroundTruthReconS1.pvp');
N = numel(pvpraw);
Recon1X1 = zeros(4,3,numfeatures,N);
for n=1:N
    ReconS1(:,:,:,n) = pvpraw{n}.values;
end%for

confidenceTableS1 = createConfidenceTable(GroundTruth,ReconS1,numintervals);
writeConfidenceTable(confidenceTableS1(:,1:end-1), confidenceTableS1(1,end), confidenceTableS1(end, end), mask, 'pvpFiles/confidenceTableS1.bin');

clear n N pvpraw numfeatures numintervals;
