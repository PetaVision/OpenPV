%A script to make depth tuning curves
addpath('~/workspace/PetaVision/mlab/util');

%To avoid losing focus when plotting
setenv("GNUTERM","dumb");

%outDir = '/nh/compneuro/Data/Depth/LCA/benchmark/depth_tune/';
outDir = '~/output/';
%dataDir = '/nh/compneuro/Data/Depth/';
dataDir = '~/saved_output/benchmarkEncoding/';
loadData = false;

LCA_v1ActFile = [dataDir, 'V1_LCA.pvp'];
RELU_v1ActFile = [dataDir, 'V1_RELU.pvp'];

depthFile = ['~/saved_output/depthTuning/DepthDownsample.pvp'];
plotOutDir = [outDir, '/depthTuning/'];

mkdir(plotOutDir)

dictPvpDir = ['~/saved_output/dictTrain/saved_binoc_512_white/recons_run/Last/'];
dictPvpFiles = {[dictPvpDir, 'LCA_V1ToLeftRecon_W.pvp'];...
                [dictPvpDir, 'LCA_V1ToRightRecon_W.pvp']};

%Given a depth and a neuron, these values define how big of an x/y patch to look for that neuron at
sampleDim = 5;
numDepthBins = 64;

targetNeurons = [239, 3, 308];


%Create output directory in outDir
mkdir(plotOutDir);

%Get all relevent info 
saveFilename = [outDir, 'tuningData.mat']
if(loadData)
   load(saveFilename);
else
   [LCA_outVals, LCA_kurtVals, LCA_peakMean] = calcDepthTuning(LCA_v1ActFile, depthFile, sampleDim, numDepthBins, 11);
   [RELU_outVals, RELU_kurtVals, RELU_peakMean] = calcDepthTuning(RELU_v1ActFile, depthFile, sampleDim, numDepthBins, 1);
   save(saveFilename, 'LCA_outVals', 'LCA_kurtVals', 'LCA_peakMean', 'RELU_outVals', 'RELU_kurtVals', 'RELU_peakMean');
end

disp('Loading done, reading dictionary pvpfiles');

%Get dictionary elements
[left_w_data, left_hdr] = readpvpfile(dictPvpFiles{1});
[right_w_data, right_hdr] = readpvpfile(dictPvpFiles{2});

disp('Done')

%Check sizes
[numNeurons, numDepths, numLines] = size(LCA_outVals);
assert(numNeurons, size(RELU_outVals, 1));
assert(numDepths, size(RELU_outVals, 2));
assert(numLines, size(RELU_outVals, 3));


%Set plot default sizes
set(0, ...
'DefaultTextFontSize', 20, ...
'DefaultTextFontWeight', 'bold', ...
'DefaultAxesFontSize', 14, ...
'DefaultAxesFontName', 'Times New Roman', ...
'DefaultLineLineWidth', 4)

%1 figure per neuron
for i = 1:length(targetNeurons);
   ni = targetNeurons(i);
   handle = figure;
   if(size(left_w_data{1}.values{1}, 3) == 1)
      colormap(gray);
   end

   %Left and right data
   SL = subplot(3, 2, 1, 'align');
   leftImg = permute(left_w_data{1}.values{1}(:, :, :, ni), [2, 1, 3]);
   norm_leftImg = (leftImg - min(leftImg(:))) / (max(leftImg(:))-min(leftImg(:)));
   imshow(norm_leftImg);
   axis off;

   ax = get(SL, 'Position');
   ax(3) += .1;
   ax(4) += .1;
   ax(1) -= .05;
   ax(2) -= .05;
   set(SL, 'Position', ax);
   

   SR = subplot(3, 2, 2, 'align');
   rightImg = permute(right_w_data{1}.values{1}(:, :, :, ni), [2, 1, 3]);
   norm_rightImg = (rightImg - min(rightImg(:))) / (max(rightImg(:))-min(rightImg(:)));
   imshow(norm_rightImg);
   axis off;
   ax = get(SR, 'Position');
   ax(3) += .1;
   ax(4) += .1;
   ax(1) -= .05;
   ax(2) -= .05;
   set(SR, 'Position', ax);

   SLCA = subplot(3, 2, [3, 4]);
   hold on;
   %One plot per numLines
   for(plotIdx = 1:numLines)
      %LCA in red
      hLCA = plot(LCA_outVals(ni, :, plotIdx), 'color', 'r');
   end
   hold off;

   %ylabel('T(u)', 'FontSize', 16);

   L = legend(hLCA, 'SCANN');
   set(L, 'FontSize', 24);
   legend left
   legend boxoff

   set(gca, 'xticklabel', []);

   ax = get(SLCA, 'Position');
   ax(4) += .05;
   ax(2) -= .05;
   set(SLCA, 'Position', ax);

   SRELU = subplot(3, 2, [5, 6]);
   hold on;
   for(plotIdx = 1:numLines)
      %RELU in blue 
      hRELU = plot(RELU_outVals(ni, :, plotIdx), 'color', 'b');
   end
   hold off;

   L = legend(hRELU, 'ReLU');
   set(L, 'FontSize', 24);
   legend left
   legend boxoff

   %ylabel('T(u)', 'FontSize', 16);
   set(gca, 'xticklabel', []);

   ax = get(SRELU, 'Position');
   ax(4) += .05;
   ax(2) -= .06;
   set(SRELU, 'Position', ax);

   print(handle, [plotOutDir, num2str(ni), '.png']);
   close(handle)
end

%%%Kurtosis
%%Write mean and std of kurtosis in file
%LCA_kurtFile = fopen([plotOutDir, 'LCA_kurtosis.txt'], 'w');
%fprintf(LCA_kurtFile, 'kurtosis: %f +- %f\n', mean(LCA_kurtVals(:)), std(LCA_kurtVals(:)));
%[LCA_sortedKurt, LCA_sortedKurtIdxs] = sort(LCA_kurtVals, 'descend');
%
%RELU_kurtFile = fopen([plotOutDir, 'RELU_kurtosis.txt'], 'w');
%fprintf(RELU_kurtFile, 'kurtosis: %f +- %f\n', mean(RELU_kurtVals(:)), std(RELU_kurtVals(:)));
%[RELU_sortedKurt, RELU_sortedKurtIdxs] = sort(RELU_kurtVals, 'descend');
%
%%Write ranking by kurtosis
%for(ni = 1:numNeurons)
%   fprintf(LCA_kurtFile, '%d: %f\n', LCA_sortedKurtIdxs(ni), LCA_sortedKurt(ni));
%   fprintf(RELU_kurtFile, '%d: %f\n', RELU_sortedKurtIdxs(ni), RELU_sortedKurt(ni));
%end
%
%fclose(LCA_kurtFile);
%fclose(RELU_kurtFile);
%%fclose(ICA_kurtFile);

%%Histogram of all kurt vals
%handle = figure;
%hold on;
%%hist(ICA_kurtVals, 'r', 'BarWidth',.9);
%hist(RELU_kurtVals, 'b', 'BarWidth', .9);
%hist(LCA_kurtVals, 'g', 'BarWidth', .7);
%L = legend('Feedforward + RELU', 'LCA');
%FL = findall(L, '-property','FontSize');
%set(FL, 'FontSize', 16);
%title('Kurtosis Histogram', 'FontSize', 28);
%xlabel({'Kurtosis Value','Less Selective               More Selective'}, 'FontSize', 16);
%ylabel('Count', 'FontSize', 16);
%hold off;
%outFilename = [plotOutDir, 'Kurtosis_Hist.png'];
%print(handle, outFilename);
%close(handle);

%%%Peak mean
%%Write mean and std of peakmean in file
%LCA_pmFile = fopen([plotOutDir, 'LCA_peakmean.txt'], 'w');
%fprintf(LCA_pmFile, 'peak-mean: %f +- %f\n', mean(LCA_peakMean(:)), std(LCA_peakMean(:)));
%[LCA_sortedPm, LCA_sortedPmIdxs] = sort(LCA_peakMean, 'descend');
%
%RELU_pmFile = fopen([plotOutDir, 'RELU_peakmean.txt'], 'w');
%fprintf(RELU_pmFile, 'peak-mean: %f +- %f\n', mean(RELU_peakMean(:)), std(RELU_peakMean(:)));
%[RELU_sortedPm, RELU_sortedPmIdxs] = sort(RELU_peakMean, 'descend');
%
%%Write ranking by peakmean
%for(ni = 1:numNeurons)
%   fprintf(LCA_pmFile, '%d: %f\n', LCA_sortedPmIdxs(ni), LCA_sortedPm(ni));
%   fprintf(RELU_pmFile, '%d: %f\n', RELU_sortedPmIdxs(ni), RELU_sortedPm(ni));
%end

%Set plot default sizes
set(0, ...
'DefaultTextFontSize', 20, ...
'DefaultTextFontWeight', 'bold', ...
'DefaultAxesFontSize', 20, ...
'DefaultAxesFontName', 'Times New Roman', ...
'DefaultLineLineWidth', 3)

%Histogram of all peakMeans
[RELUf, RELUx] = hist(RELU_peakMean,[.4: .05: .85], 'b', 'BarWidth', .9);
[LCAf, LCAx] = hist(LCA_peakMean, [.4: .05: .85], 'r', 'BarWidth',.7);

handle = figure;
set(handle, 'Position', [1 1 1 .5])

hold on;
bar(RELUx, RELUf/max(RELUf(:)), 'b', 'BarWidth', .9);
bar(LCAx, LCAf/max(LCAf(:)), 'r', 'BarWidth', .7);
hold off;

L = legend('ReLU', 'SCANN');
%ax = get(L, 'Position');
%ax(1) -= .01;
%
%set(L, 'Position', ax);
%set(L, 'FontSize', 24);


legend boxoff
legend left


xlabel('<- Less Selective   Peak-Mean Value   More Selective ->');
ylabel('Normalized Count', 'FontSize', 24);
outFilename = [plotOutDir, 'PeakMean_Hist.png'];


print(handle, outFilename);
close(handle);

