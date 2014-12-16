addpath("~/workspace/PetaVision/mlab/util")
outdir = '/nh/compneuro/Data/Depth/LCA/benchmark/stereo_response_batch_all/';
plotdir = [outdir 'activation/'];
mkdir(plotdir);
checkpointDir = 'Last/';
weightFiles = {'DepthGTToV1S2'; ...
           'DepthGTToV1S4'; ...
           'DepthGTToV1S8'; ...
          };

numDepth = 128;

sumPostNf = zeros(length(weightFiles), numDepth);
hists = {}
for wi = 1:length(weightFiles)
   weightFile = [outdir checkpointDir weightFiles{wi} '_W.pvp'];

   [data, hdr] = readpvpfile(weightFile);


   weights = data{1}.values{1};
   weights = squeeze(max(squeeze(max(weights))));

   [numPostNf, numPreNf] = size(weights);

   histIdxs = zeros(length(weightFiles), numPostNf);

   weights = reshape(weights, numPostNf, numDepth, []);
   weights = squeeze(mean(weights, 3));
   %weights are now [postNf, preNf]

   %For every postNf (V1 neuron)
   for ni = 1:numPostNf
      %Plot vector
      activation = weights(ni, :);
      [drop, idx] = max(activation);
      histIdxs(wi, ni) = numDepth - idx;

      figure;
      x = numDepth:-1:1;
      handle = plot(x, activation/max(activation(:)));
      outFilename = [plotdir, weightFiles{wi}, '_', num2str(ni), '.png'];
      title("Depth vs Activation for single neuron");
      xlabel("Depth (Low values are close, High values are far)");
      ylabel("Normalized activation");
      saveas(handle, outFilename);
   end

   [yHist, xHist] = hist(histIdxs(wi, :), [0, 10, 20, 30, 40, 50, 60, 70 ,80, 90, 100, 110, 120]);
   hists{wi}.yHist = yHist;
   hists{wi}.xHist = xHist;
end

handle = figure;
hold on
bar(hists{3}.xHist, hists{3}.yHist, .7, 'FaceColor', [.2, .2, .5])
bar(hists{2}.xHist, hists{2}.yHist, .5, 'FaceColor', [.2, .5, .2])
bar(hists{1}.xHist, hists{1}.yHist, .3, 'FaceColor', [.5, .2, .2])
hold off
legend('V1S8', 'V1S4', 'V1S2');
xlim([1, 128]);
ylim([0, 70]);
title("Max Depth Histogram");
xlabel("Depth (Low values are close, High values are far)");
ylabel("Number Activations");
outFilename = [plotdir, "hist.png"];
print(handle, outFilename);
   
   

%for wi = 1:length(weightFiles)
%   sumPostNf(wi, :) = sumPostNf(wi, :) / max(sumPostNf(wi, :));
%end
%x = numDepth:-1:1;
%
%handle = figure;
%plot(x, (sumPostNf(1, :)), x, (sumPostNf(2, :)), x, (sumPostNf(3, :)));
%legend(weightFiles{1}, weightFiles{2}, weightFiles{3});
%title("Depth vs Activation")
%xlabel("Depth (Low values are close, High values are far)");
%ylabel("Normalized activation");
%
%outFilename = strcat(plotdir, "mean.png");
%print(handle, outFilename);
%
%
