clear all; close all; dbstop error;

%addpath('devkit/matlab/')
addpath('~/workspace/PetaVision/mlab/util')
outdir =  '~/output/depthInference/'
mkdir(outdir);
LCAdir =  '~/saved_output/depthInference/saved_ATA_LCA/';
RELUdir = '~/saved_output/depthInference/saved_ATA_RELU/';

%For rcorr patches
targetNeurons = [239, 3, 308]; %1 indexed
LCADictFilename = [LCAdir, 'V1ToDepthGT_W.pvp'];
RELUDictFilename = [RELUdir, 'V1ToDepthGT_W.pvp'];
%For rcorr patch plots
ySampleSkip = 6;
whiteRange = [0, .75];
borderWidth = 1;

patchDir = [outdir 'patches/'];
mkdir(patchDir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function outI = toColor(I, maxDisp)

outI = double(I(:))/maxDisp;
map = [0 0 0 114; 0 0 1 185; 1 0 0 114; 1 0 1 174; ...
       0 1 0 114; 0 1 1 185; 1 1 0 114; 1 1 1 0];

bins  = map(1:end-1,4);
cbins = cumsum(bins);
bins  = bins./cbins(end);
cbins = cbins(1:end-1) ./ cbins(end);
ind   = min(sum(repmat(outI(:)', [6 1]) > repmat(cbins(:), [1 numel(outI)])),6) + 1;
bins  = 1 ./ bins;
cbins = [0; cbins];

outI = (outI-cbins(ind)) .* bins(ind);
outI = min(max(map(ind,1:3) .* repmat(1-outI, [1 3]) + map(ind+1,1:3) .* repmat(outI, [1 3]),0),1);

outI = reshape(outI, [size(I, 1) size(I, 2) 3]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%RCorr patches for target neurons
[LCAPatches, LCAPatchesHdr] = readpvpfile(LCADictFilename);
[RELUPatches, RELUPatchesHdr] = readpvpfile(RELUDictFilename);

LCADict = LCAPatches{1}.values{1};
RELUDict = RELUPatches{1}.values{1};
%Matrix in the form of [nxp, nyp, nfp, kernelNum];

[nxp, nyp, nfp, kernelNum] = size(LCADict);
assert(nxp == size(RELUDict, 1));
assert(nyp == size(RELUDict, 2));
assert(nfp == size(RELUDict, 3));
assert(kernelNum == size(RELUDict, 4));

%1 plot per target neuron
%neuronImg = zeros(3*borderWidth + 2*nyp, (length(targetNeurons)+1)*borderWidth + length(targetNeurons)*nxp, 3);

LCAOutPatches = zeros(length(targetNeurons), nyp, nxp);
RELUOutPatches = zeros(length(targetNeurons), nyp, nxp);

maxPatchVal = -inf;
for ni = 1:length(targetNeurons)
   target_ni = targetNeurons(ni);
   LCA_targetPatch = LCADict(:, :, :, target_ni);
   RELU_targetPatch = RELUDict(:, :, :, target_ni);
   %Now in the form [nxp, nyp, nfp]
   %Do winner take all to reduce depth bins, 0 index it to keep it the same as PV
   [LCA_vals, LCA_patch] = max(LCA_targetPatch, [], 3);
   [RELU_vals, RELU_patch] = max(RELU_targetPatch, [], 3);

   %Change to 0 indexing
   LCA_patch = ((LCA_patch-1)/127)';
   RELU_patch = ((RELU_patch-1)/127)';

   LCAOutPatches(ni, :, :) = LCA_patch;
   RELUOutPatches(ni, :, :) = RELU_patch;
   
   maxLCA = max(LCA_patch(:));
   maxRELU = max(RELU_patch(:));

   if maxLCA > maxPatchVal
      maxPatchVal = maxLCA;
   end
   if maxRELU > maxPatchVal
      maxPatchVal = maxRELU;
   end
end

currX = borderWidth+1;
patches_errorfile= fopen([patchDir, 'patch_error.txt'], 'w');
for ni = 1:length(targetNeurons)
   LCA_patch = squeeze(LCAOutPatches(ni, :, :));
   RELU_patch = squeeze(RELUOutPatches(ni, :, :));

   %Put to same color scale, based on previously found maxGT
   LCA_color_patch = toColor(LCA_patch, maxPatchVal);
   RELU_color_patch = toColor(RELU_patch, maxPatchVal);

   outFilename = [patchDir, num2str(targetNeurons(ni)), '_LCA_rcorr_patches.png'];
   imwrite(LCA_color_patch, outFilename);

   outFilename = [patchDir, num2str(targetNeurons(ni)), '_RELU_rcorr_patches.png'];
   imwrite(RELU_color_patch, outFilename);

   %%Add to neuronImg
   %neuronImg(borderWidth+1:nyp+borderWidth, currX:currX+nxp-1, :) = LCA_color_patch;
   %neuronImg(borderWidth+nyp+borderWidth+1:end-borderWidth, currX:currX+nxp-1, :) = RELU_color_patch;
   %currX += nxp+borderWidth;

   fprintf(patches_errorfile, 'LCA_%d: %f\n', ni, mean(LCA_patch(:)));
   fprintf(patches_errorfile, 'RELU_%d: %f\n', ni, mean(RELU_patch(:)));
end

fclose(patches_errorfile)


%Set defaults
%Set plot default sizes
set(0, 'DefaultTextFontSize', 30);
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultTextFontName', 'Times');
set(0, 'DefaultAxesFontSize', 20);
set(0, 'DefaultAxesFontName', 'Times');
set(0, 'DefaultLineLineWidth', 20);

%Sum over x, make one depth plot per every # neurons, goes from heavy red (top) to light red (bot)
for ni = 1:length(targetNeurons)
   target_ni = targetNeurons(ni);
   LCA_targetPatch = LCADict(:, :, :, target_ni);
   RELU_targetPatch = RELUDict(:, :, :, target_ni);
   plot_LCA = squeeze(sum(LCA_targetPatch, 1));
   plot_RELU= squeeze(sum(RELU_targetPatch, 1));

   handle = figure;
   hold on

   numSteps = floor(nyp/ySampleSkip)
   whitenStepVal = (whiteRange(2)-whiteRange(1))/numSteps;
   whitenVals = whiteRange(1):whitenStepVal:(whiteRange(2)); %Values of other channels to make light red/blue
   sampleIdxs = 1:ySampleSkip:nyp;
   assert(length(whitenVals) == length(sampleIdxs));
   for(i = 1:length(whitenVals))

      yi = sampleIdxs(i);
      whitenVal = whitenVals(i);
      
      if(i == 1)
         h_LCA = plot(squeeze(plot_LCA(yi, :)), 'color', [1, whitenVal, whitenVal]);
         axis([0, 128, -.12, -.04])
         h_RELU = plot(squeeze(plot_RELU(yi, :)), 'color', [whitenVal, whitenVal, 1]);
         axis([0, 128, -.12, -.04])
      else
         plot(squeeze(plot_LCA(yi, :)), 'color', [1, whitenVal, whitenVal]);
         axis([0, 128, -.12, -.04])
         plot(squeeze(plot_RELU(yi, :)), 'color', [whitenVal, whitenVal, 1]);
         axis([0, 128, -.12, -.04])
      end
   end
   hold off
   L = legend([h_LCA, h_RELU], 'SCANN', 'ReLU');
   legend left
   legend boxoff
   set(L, 'FontSize', 30);
   set(gca, 'xticklabel', []);
   %xlabel('Far                                     Near', 'FontSize', 30);
   %ylabel('STA Activation', 'FontSize', 30);
   
   outFilename = [patchDir, num2str(target_ni), '_lineplot.png'];
   saveas(handle, outFilename);
end

%Sum over x only, make heatmap
for ni = 1:length(targetNeurons)
   target_ni = targetNeurons(ni);
   LCA_targetPatch = LCADict(:, :, :, target_ni);
   RELU_targetPatch = RELUDict(:, :, :, target_ni);
   %PV axis, goes x, y, f, kernel
   plot_LCA = squeeze(sum(LCA_targetPatch, 1));
   plot_RELU= squeeze(sum(RELU_targetPatch, 1));
   
   maxVal = -inf;
   minVal = inf;

   if(max(plot_LCA(:)) > maxVal)
      maxVal = max(plot_LCA(:));
   end
   if(max(plot_RELU(:)) > maxVal)
      maxVal = max(plot_RELU(:));
   end
   if(min(plot_LCA(:)) < minVal)
      minVal = min(plot_LCA(:));
   end
   if(min(plot_RELU(:)) < minVal)
      minVal = min(plot_RELU(:));
   end

   %Normalize to be between 0 and 1
   norm_plotLCA = (plot_LCA - minVal)/(maxVal-minVal);
   norm_plotRELU = (plot_RELU - minVal)/(maxVal-minVal);

   [ny, ndepth] = size(norm_plotLCA);
   assert(ny == size(norm_plotRELU, 1));
   assert(ndepth == size(norm_plotRELU, 2));

   outFilename = [patchDir, num2str(target_ni), '_LCA_heatplot.png'];
   imwrite(norm_plotLCA, outFilename);

   outFilename = [patchDir, num2str(target_ni), '_RELU_heatplot.png'];
   imwrite(norm_plotRELU, outFilename);
end

