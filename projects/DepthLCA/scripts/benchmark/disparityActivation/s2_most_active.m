addpath("~/workspace/PetaVision/mlab/util")
outdir = '/nh/compneuro/Data/Depth/LCA/benchmark/stereo_response_batch_all/';
plotdir = [outdir 'activation/'];
mkdir(plotdir);
checkpointDir = 'Last/';
weightFiles = {'DepthGTToV1S2';};
preFiles = {'V1S2';};

numDepth = 128;

sumPostNf = zeros(length(weightFiles), numDepth);
for wi = 1:length(weightFiles)
   weightFile = [outdir checkpointDir weightFiles{wi} '_W.pvp'];
   preFile = "/nh/compneuro/Data/Depth/LCA/benchmark/stereo_train/a16_V1S2.pvp";

   [data, hdr] = readpvpfile(weightFile);

   [preData, preHdr] = readpvpfile(preFile);

   numActivations = zeros(preHdr.nf, 1);
   [numFrames, drop] = size(preData)
   for(iframe = 1:numFrames)
      indices = preData{iframe}.values(:, 1);
      indices = indices - 1;
      indices = mod(indices, preHdr.nf);
      indices = indices + 1;
      for(fi = 1:preHdr.nf);
         numActivations(fi) = numActivations(fi) + length(find(indices == fi));
      end
   end

   [drop, idx] = max(numActivations);


   keyboard

   weights = data{1}.values{1};
   weights = squeeze(max(squeeze(max(weights))));

   [numPostNf, numPreNf] = size(weights);

   weights = reshape(weights, numPostNf, numDepth, []);
   weights = squeeze(mean(weights, 3));
   %weights are now [postNf, preNf]

   %For every postNf (V1 neuron)
   for ni = 1:numPostNf
      %Plot vector
      activation = weights(ni, :);
      sumPostNf(wi, :) = sumPostNf(wi, :) + activation;
      [drop, idx] = max(activation);

      %figure;
      %x = numDepth:-1:1;
      %handle = plot(x, activation/max(activation(:)));
      %outFilename = [plotdir, weightFiles{wi}, '_', num2str(ni), '.png'];
      %title("Depth vs Activation for single neuron");
      %xlabel("Depth (Low values are close, High values are far)");
      %ylabel("Normalized activation");
      %saveas(handle, outFilename);
   end
end

for wi = 1:length(weightFiles)
   sumPostNf(wi, :) = sumPostNf(wi, :) / max(sumPostNf(wi, :));
end
x = numDepth:-1:1;

handle = figure;
plot(x, (sumPostNf(1, :)), x, (sumPostNf(2, :)), x, (sumPostNf(3, :)));
legend(weightFiles{1}, weightFiles{2}, weightFiles{3});
title("Depth vs Activation")
xlabel("Depth (Low values are close, High values are far)");
ylabel("Normalized activation");

outFilename = strcat(plotdir, "mean.png");
print(handle, outFilename);


