disp('======= KITTI DevKit Demo =======');
clear all; close all; dbstop error;

addpath('devkit/matlab/')
addpath('~/workspace/PetaVision/mlab/util')

% error threshold
tau = 3;

outdir = '/nh/compneuro/Data/Depth/LCA/benchmark/validate/rcorr_nf_512_ICA/';
%timestamp = [outdir '/timestamps/DepthImage.txt'];
outPvpFile = [outdir 'a5_RCorrRecon.pvp'];
gtPvpFile = [outdir 'a2_DepthDownsample.pvp'];
scoreDir = [outdir 'scores/']

mkdir(scoreDir);

[data_est, hdr_est] = readpvpfile(outPvpFile);
[data_gt, hdr_gt] = readpvpfile(gtPvpFile);

numFrames = hdr_est.nbands;

%%Build timestamp matrix
%time = zeros(1, numFrames);
%gtFilenames = cell(1, numFrames);
%
%%Build timestamp matrix
%%timeFile = fopen(timestamp, 'r');
%
%for(i = 1:numFrames)
%   line = fgetl(timeFile);
%   split = strsplit(line, ',');
%   time(1,i) = str2num(split{2});
%   gtFilenames(1,i) = split(3);
%end

%fclose(timeFile)

for(i = 1:numFrames)
   estData = data_est{i}.values' * 256;
   targetTime = data_est{i}.time;
   gtData = data_gt{i}.values' * 256;
   outFilename = [scoreDir num2str(targetTime) '.png']

   d_err = disp_error(gtData,estData,tau);
   figure;
   handle = imshow(disp_to_color([estData;gtData]));
   title(sprintf('Error: %.2f %%',d_err*100));
   saveas(handle, outFilename);
end
   



%% flow demo
%disp('Load and show optical flow field ... ');
%F_est = flow_read('flow_est.png');
%F_gt  = flow_read('flow_gt.png');
%f_err = flow_error(F_gt,F_est,tau);
%F_err = flow_error_image(F_gt,F_est);
%figure,imshow([flow_to_color([F_est;F_gt]);F_err]);
%title(sprintf('Error: %.2f %%',f_err*100));
%figure,flow_error_histogram(F_gt,F_est);
