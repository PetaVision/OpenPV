clear all; close all; dbstop error;

addpath('devkit/matlab/')
addpath('~/workspace/PetaVision/mlab/util')


outDir = '/home/ec2-user/mountData/plots/'
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

%Reading smaller for debugging
[data_gt, hdr_gt] = readpvpfile(gtPvpFile, 1, 10, 0, 0);
[data_LCA, hdr_LCA] = readpvpfile(LCAFilename, 1, 10, 0, 0);
[data_RELU, hdr_RELU] = readpvpfile(RELUFilename, 1, 10, 0, 0);

numFrames = hdr_gt.nbands;
%Sanity checks
assert(numFrames == hdr_LCA.nbands);
assert(numFrames == hdr_RELU.nbands);

%Build timestamp matrix
time = zeros(numFrames, 1);
gtFilenames = cell(numFrames, 1);

%Build timestamp matrix
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
maxGT = 0;
minGT = 256;
for(i = 1:numFrames)
   gtData = data_gt{i}.values' * 256; %Do we need to round here?
   curr_maxGT = max(gtData(:));
   curr_minGT = min(gtData(:));
   if(curr_maxGT > maxGT)
      maxGT = curr_maxGT;
   end
   if(curr_minGT < minGT)
      minGT = curr_minGT;
   end
end

keyboard
      


%for(i = 1:numFrames)
%   estData = data_est{i}.values' * 256;
%   gtData = data_gt{i}.values' * 256;
%
%   handle = figure;
%   targetTime = data_est{i}.time;
%   targetFrame = gtFilenames{1, i};
%   imageFilename = [imageDir, targetFrame]
%
%   system(['aws s3 cp ', imageFilename, ' tmpImg.png']);
%
%   outFilename = [scoreDir num2str(targetTime) '_EstVsImage.png']
%   im = imread('tmpImg.png');
%   [nx, ny, nf] = size(estData);
%
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
%   %Mask out estdata with mask
%   estData(find(gtData == 0)) = 0;
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
