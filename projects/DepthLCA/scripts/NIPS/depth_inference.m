clear all; close all; dbstop error;

%addpath('devkit/matlab/')
addpath('~/workspace/OpenPV/pv-core/mlab/util')

%outdir =  '/nh/compneuro/Data/Depth/LCA/benchmark/validate/plots/';
outdir =  '~/mountData/outplots/LCA_vs_RELU_sparse/'

mkdir(outdir);

LCAdir = '~/mountData/benchmark/validate/rcorr/aws_icaweights_binoc_LCA/'
RELUdir =  '~/mountData/benchmark/validate/rcorr/aws_icapatch_binoc_RELU_sparse/'

%These should be equivelent for LCA or RELU
timestamp = [LCAdir '/timestamps/DepthImage.txt'];
gtPvpFile = [LCAdir 'a4_DepthDownsample.pvp'];

%imageDir = 's3://kitti/stereo_flow/multiview/training/image_2/';
%imageDir = '/nh/compneuro/Data/KITTI/stereo_flow/multiview/training/image_2/'
imageDir = '/home/sheng/mountData/datasets/kitti/stereo_flow/multiview/training/image_2/'

%Estimate pvps
LCAFilename = [LCAdir 'a7_RCorrRecon.pvp'];
RELUFilename = [RELUdir 'a8_RCorrRecon.pvp'];

% error threshold for error calculation
tau = 3;

scoreDir = [outdir];
mkdir(scoreDir);


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

function d_err = disp_error (D_gt,D_est,tau)
   %Change to 0-255 px values
   if(max(D_gt(:)) <= 1)
      D_gt = D_gt * 255;
      D_est = D_est * 255;
   end
   E = abs(D_gt-D_est);
   E(D_gt<=0) = 0;
   d_err = length(find(E>tau))/length(find(D_gt>0));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[data_gt, hdr_gt] = readpvpfile(gtPvpFile);
[data_LCA, hdr_LCA] = readpvpfile(LCAFilename);
[data_RELU, hdr_RELU] = readpvpfile(RELUFilename);

%%Reading less frames for debugging
%[data_gt, hdr_gt] = readpvpfile(gtPvpFile, 0, 10, 0, 0);
%[data_LCA, hdr_LCA] = readpvpfile(LCAFilename, 0, 10, 0, 0);
%[data_RELU, hdr_RELU] = readpvpfile(RELUFilename, 0, 10, 0, 0);

numFrames = length(data_gt);
numFrames = min(numFrames, length(data_LCA));
numFrames = min(numFrames, length(data_RELU));
numFrames

%Build timestamp matrix
time = zeros(numFrames, 1);
gtFilenames = cell(numFrames, 1);

timeFile = fopen(timestamp, 'r');
for(i = 1:numFrames)
   line = fgetl(timeFile);
   split = strsplit(line, ',');
   time(i, 1) = str2num(split{2});
   frameName = strsplit(split{end}, '/'){end};
   gtFilenames(i, 1) = frameName;
end
%time is the corresponding time of the run, gtFilenames contains the suffix filename of the actual image
fclose(timeFile)

%List to keep track of error values
LCA_errList = zeros(numFrames, 1);
RELU_errList = zeros(numFrames, 1);
for(i = 2:numFrames)
   %Get all data
   GTData = data_gt{i}.values';
   LCAData = data_LCA{i}.values';
   RELUData = data_RELU{i}.values';

   %Get max data for current frame
   curr_maxGT = max(GTData(:));

   LCA_errList(i) = disp_error(GTData, LCAData, tau);
   RELU_errList(i) = disp_error(GTData, RELUData, tau);

   %Calc target nx and ny
   targetNx = size(GTData, 2) * 4;
   targetNy = size(GTData, 1) * 4;

   %Mask out estdata with mask
   LCAData(find(GTData == 0)) = 0;
   RELUData(find(GTData == 0)) = 0;
   %Image data
   targetTime = data_gt{i}.time;
   targetFrame = gtFilenames{i, 1};

   
   imageFilename = [imageDir, targetFrame];

   if(imageFilename(1:3) == 's3:')
      system(['aws s3 cp ', imageFilename, ' tmpImg.png']);
      image = imread('tmpImg.png');
   else
      image = imread(imageFilename);
   end


   [imageNy, imageNx, nf] = size(image);
   assert(imageNy >= targetNy);
   assert(imageNx >= targetNx);
   offsetX = imageNx - targetNx + 1;
   offsetY = imageNy - targetNy + 1;

   %Crop from left and top
   cropImg = double(image(offsetY:end, offsetX:end, :))/255;

   %Resize to size of image
   GTData =   imresize(GTData, [targetNy, targetNx], 'nearest');
   LCAData =  imresize(LCAData, [targetNy, targetNx], 'nearest');
   RELUData = imresize(RELUData, [targetNy, targetNx], 'nearest');


   %To color
   GTData = toColor(GTData, curr_maxGT);
   LCAData = toColor(LCAData, curr_maxGT);
   RELUData = toColor(RELUData, curr_maxGT);

   %White margin of 20 px in between images
   horMargin = ones(20, targetNx, 3);

   %Cat all images together
   outImg = [cropImg; horMargin; GTData; horMargin; LCAData; horMargin; RELUData];

   outFilename = [scoreDir targetFrame(1:end-4) '_depth_inference.png']
   imwrite(outImg, outFilename);
end

%Inference error score
LCA_ErrorFile = fopen([scoreDir, 'LCA_error.txt'], 'w');
fprintf(LCA_ErrorFile, 'Error: %f +- %f\n', mean(LCA_errList(:)), std(LCA_errList(:)));
[LCA_sortedError, LCA_sortedErrorIdx] = sort(LCA_errList);

RELU_ErrorFile= fopen([scoreDir, 'RELU_error.txt'], 'w');
fprintf(RELU_ErrorFile, 'Error: %f +- %f\n', mean(RELU_errList(:)), std(RELU_errList(:)));
[RELU_sortedError, RELU_sortedErrorIdx] = sort(RELU_errList);

%Write ranking
for(ni = 1:numFrames-1)
   fprintf(LCA_ErrorFile, '%s: %f\n', gtFilenames{LCA_sortedErrorIdx(ni), 1}(1:end-4), LCA_sortedError(ni));
   fprintf(RELU_ErrorFile, '%s: %f\n', gtFilenames{RELU_sortedErrorIdx(ni), 1}(1:end-4), RELU_sortedError(ni));
end

fclose(LCA_ErrorFile);
fclose(RELU_ErrorFile);
