addpath('~/workspace/pv-core/mlab/util');
tuningFile = '~/mountData/benchmark/featuremap/depth_tuning_rank.txt';
outFile = '~/mountData/benchmark/featuremap/depth_tuning_rank.pvp';
numFeatures = 512;
nx = 300;
ny = 90;

data{1}.time = 0;

outVals = ones(nx, ny, numFeatures);

%Read tuning file to grab values of each feature
fp = fopen(tuningFile , 'r');
%Throw away first line
line = fgetl(fp);

while(~feof(fp))
   out = fscanf(fp, '%f: %f\n', 2);
   outVals(:, :, out(1)) = out(2);
end

data{1}.values = outVals;

writepvpactivityfile(outFile, data);

