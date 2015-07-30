%Script to find average of depth maps
addpath("~/workspace/PetaVision/mlab/util");

datafile = '~/mountData/geoint/depthAve/a1_DepthDownsample.pvp';

[data, hdr] = readpvpfile(datafile);

[nx, ny] = size(data{1}.values)

sumVals = zeros(nx, ny);
counts = zeros(nx, ny);

for t = 1:length(data)
   sumVals = sumVals + data{t}.values;
   counts(find(data{t}.values > 0)) += 1;
end

meanVals = sumVals ./ counts;

meanVals(find(isnan(meanVals))) = 0;

keyboard

   
