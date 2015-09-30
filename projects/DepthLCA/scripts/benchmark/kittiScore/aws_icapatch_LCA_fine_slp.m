disp('======= KITTI DevKit Demo =======');
clear all; close all; dbstop error;

addpath('devkit/matlab/')
addpath('~/workspace/pv-core/mlab/util')

% error threshold
tau = 3;

outdir = '/home/ec2-user/mountData/benchmark/train/slp/aws_icapatch_LCA_fine_bias/';
timestamp = [outdir '/timestamps/DepthImage.txt'];
outPvpFile = [outdir 'a8_SLP_Recon.pvp'];
gtPvpFile = [outdir 'a5_DepthDownsample.pvp'];
reconFile = [outdir 'a2_LeftRecon.pvp'];
scoreDir = [outdir 'scores/']

mkdir(scoreDir);

[data_est, hdr_est] = readpvpfile(outPvpFile);
[data_gt, hdr_gt] = readpvpfile(gtPvpFile);
[data_recon, hdr_gt] = readpvpfile(reconFile);

numFrames = hdr_est.nbands;
errList = zeros(1, numFrames);

%%Build timestamp matrix
%time = zeros(1, numFrames);
%gtFilenames = cell(1, numFrames);
%
%%Build timestamp matrix
%timeFile = fopen(timestamp, 'r');
%for(i = 1:numFrames)
%   line = fgetl(timeFile);
%   split = strsplit(line, ',');
%   time(1,i) = str2num(split{2});
%   frameName = strsplit(split{3}, '/'){end};
%   gtFilenames(1,i) = frameName;
%end
%fclose(timeFile)

for(i = 1:numFrames)
   estData = data_est{i}.values' * 256;
   gtData = data_gt{i}.values' * 256;

   handle = figure;
   targetTime = data_est{i}.time;
   %targetFrame = gtFilenames{1, i};
   %imageFilename = [imageDir, targetFrame]
   %system(['aws s3 cp ', imageFilename, ' tmpImg.png']);

   outFilename = [scoreDir num2str(targetTime) '.png']
   %im = imread('tmpImg.png');
   im = data_recon{i}.values';
   im = (im - min(im(:)))/(max(im(:)) - min(im(:)));
   [nx, ny, nf] = size(estData);

   d_err = disp_error(gtData,estData,tau);
   errList(i) = d_err;

   estData(find(gtData == 0)) = 0;

   subplot(2, 1, 1);
   imshow(im);
   subplot(2, 1, 2);
   imshow(disp_to_color([estData;gtData]));

   title(sprintf('Error: %.2f %%',d_err*100));
   saveas(handle, outFilename);


   %Mask out estdata with mask
   %outFilename = [scoreDir num2str(targetTime) '_gtVsEst.png']

   %figure;
   %handle = imshow(disp_to_color([estData;gtData]));
   %saveas(handle, outFilename);
end

aveFile = fopen([scoreDir 'aveError.txt'], 'w');
fprintf(aveFile, '%f', mean(errList(:)));
fclose(aveFile);
