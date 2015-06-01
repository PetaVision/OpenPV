clear all; close all; dbstop error;

addpath('devkit/matlab/')
addpath('~/workspace/PetaVision/mlab/util')

outdir = '/home/ec2-user/mountData/plots/'
LCAdir = '/home/ec2-user/mountData/benchmark/validate/aws_white_rcorr_LCA/';
RELUdir = '/home/ec2-user/mountData/benchmark/validate/aws_white_rcorr_RELU/'

%These should be equivelent for LCA or RELU
timestamp = [LCAdir '/timestamps/DepthImage.txt'];
gtPvpFile = [LCAdir 'a3_DepthDownsample.pvp'];
imageDir = 's3://kitti/stereo_flow/multiview/training/image_2/'

%Estimate pvps
LCAFilename = [LCAdir 'a6_RCorrRecon.pvp'];
RELUFilename = [RELUdir 'a5_RCorrRecon.pvp'];

% error threshold for error calculation
tau = 3;

scoreDir = [outdir 'scores/']
mkdir(scoreDir);

%[data_gt, hdr_gt] = readpvpfile(gtPvpFile);
%[data_LCA, hdr_LCA] = readpvpfile(LCAFilename);
%[data_RELU, hdr_RELU] = readpvpfile(RELUFilename);

%numFrames = hdr_gt.nbands;
%%Sanity checks
%assert(numFrames == hdr_LCA.nbands);
%assert(numFrames == hdr_RELU.nbands);

%Reading smaller for debugging
[data_gt, hdr_gt] = readpvpfile(gtPvpFile, 0, 10, 0, 0);
[data_LCA, hdr_LCA] = readpvpfile(LCAFilename, 0, 10, 0, 0);
[data_RELU, hdr_RELU] = readpvpfile(RELUFilename, 0, 10, 0, 0);

numFrames = length(data_gt);
assert(numFrames == length(data_LCA));
assert(numFrames == length(data_RELU));

%Build timestamp matrix
time = zeros(numFrames, 1);
gtFilenames = cell(numFrames, 1);

timeFile = fopen(timestamp, 'r');
for(i = 1:numFrames)
   line = fgetl(timeFile);
   split = strsplit(line, ',');
   time(i, 1) = str2num(split{2});
   frameName = strsplit(split{3}, '/'){end};
   gtFilenames(i, 1) = frameName;
end
%time is the corresponding time of the run, gtFilenames contains the suffix filename of the actual image
fclose(timeFile)

%List to keep track of error values
LCA_errList = zeros(numFrames, 1);
RELU_errList = zeros(numFrames, 1);

%%Find absolute gt scale first
maxGT = -inf;
minGT = inf;
for(i = 1:numFrames)
   gtData = data_gt{i}.values' * 256; %Do we need to round here?
   curr_maxGT = max(gtData(:));
   %Do not include 0 px value DNC area in finding min
   tmpGTData = gtData;
   tmpGTData(find(gtData == 0)) = inf;
   curr_minGT = min(tmpGTData(:));

   if(curr_maxGT > maxGT)
      maxGT = curr_maxGT;
   end
   if(curr_minGT < minGT)
      minGT = curr_minGT;
   end
end
minGT = floor(minGT);
maxGT = ceil(maxGT);

for(i = 1:numFrames)
   %Get all data
   GTData = data_gt{i}.values' * 256;
   LCAData = data_LCA{i}.values' * 256;
   RELUData = data_RELU{i}.values' * 256;
   %Mask out estdata with mask
   LCAData(find(GTData == 0)) = 0;
   RELUData(find(GTData == 0)) = 0;

   %Image data
   targetTime = data_gt{i}.time;
   targetFrame = gtFilenames{i, 1};
   imageFilename = [imageDir, targetFrame]
   system(['aws s3 cp ', imageFilename, ' tmpImg.png']);
   image = imread('tmpImg.png');

   %Make figure
   h = figure;
   %Image
   subplot(4, 1, 1);
   imagesc(image);
   axis off;
   %GT
   subplot(4, 1, 2);
   imagesc(GTData, [minGT maxGT]);
   colormap(jet);
   axis off;
   %LCA
   subplot(4, 1, 3);
   imagesc(LCAData, [minGT maxGT]);
   colormap(jet);
   axis off;
   %RELU
   subplot(4, 1, 4);
   imagesc(RELUData, [minGT maxGT]);
   colormap(jet);
   axis off;

   keyboard
end











%   %TODO reszie other way
%   im = imresize(im, [nx, ny]);
%   subplot(2, 1, 1);
%   imshow(im);
%   subplot(2, 1, 2);
%   imshow(disp_to_color(estData));
%   %handle = imshow([disp_to_color(estData); im]);
%   saveas(handle, outFilename);
%
%   d_err = disp_error(gtData,estData,tau);
%   errList(i) = d_err;
%
%   outFilename = [scoreDir num2str(targetTime) '_gtVsEst.png']
%
%   figure;
%   handle = imshow(disp_to_color([estData;gtData]));
%   title(sprintf('Error: %.2f %%',d_err*100));
%   saveas(handle, outFilename);
%end
%
%aveFile = fopen([scoreDir 'aveError.txt'], 'w');
%fprintf(aveFile, '%f', mean(errList(:)));
%fclose(aveFile);
