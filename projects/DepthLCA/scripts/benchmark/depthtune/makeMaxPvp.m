addpath('~/workspace/pv-core/mlab/util');
encodingFile = '~/mountData2/benchmark/icaweights_LCA/a12_V1.pvp'
outFile = '~/mountData/benchmark/featuremap/maxWeighting.pvp';
numFeatures = 512;
nx = 300;
ny = 90;

threshold = .5;


[data, hdr] = readpvpfile(encodingFile);
numFrames = length(data);

maxVals = zeros(numFeatures, 1);
N = nx * ny * numFeatures;

for f = 1:numFrames
   active_ndx = data{f}.values(:, 1);
   active_vals = data{f}.values(:, 2);
   tmp_v1 = full(sparse(active_ndx+1, 1, active_vals, N, 1, N));
   %pv does [nf, nx, ny] ordering
   tmp_v1 = reshape(tmp_v1, [numFeatures, nx, ny]);

   frameMax = squeeze(max(max(tmp_v1, [], 3), [], 2));
   assert(size(maxVals) == size(frameMax));
   maxVals = max(maxVals, frameMax);
end

%Threshold each neuron
for f = 1:numFrames
   active_ndx = data{f}.values(:, 1); %These indices are 0 indexed
   %Start from the back to remove values as needed
   for i = length(active_ndx):-1:1
      featureIdx = mod(active_ndx(i), numFeatures);
      if(data{f}.values(i, 2) <= maxVals(featureIdx+1) * threshold) 
         data{f}.values(i, :) = [];
      end
   end
end
      
writepvpsparsevaluesfile(outFile, data, nx, ny, numFeatures);

