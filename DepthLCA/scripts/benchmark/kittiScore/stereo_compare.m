disp('======= KITTI DevKit Demo =======');
clear all; close all; dbstop error;

addpath('devkit/matlab/')
addpath('~/workspace/PetaVision/mlab/util')

% error threshold
tau = 3;

outdir = '/nh/compneuro/Data/Depth/LCA/benchmark/stereo_validate_rcorr_np_batch/';
timestamp = [outdir '/timestamps/DepthImage.txt'];
outPvpFile = [outdir 'a19_RCorrReconAll.pvp'];
scoreDir = [outdir 'scores/']

mkdir(scoreDir);

[data, hdr] = readpvpfile(outPvpFile);

numFrames = hdr.nbands;

%Build timestamp matrix
time = zeros(1, numFrames);
gtFilenames = cell(1, numFrames);

%Build timestamp matrix
timeFile = fopen(timestamp, 'r');

for(i = 1:numFrames)
   line = fgetl(timeFile);
   split = strsplit(line, ',');
   time(1,i) = str2num(split{2});
   gtFilenames(1,i) = split(3);
end

fclose(timeFile)

for(i = 1:numFrames)
   estData = data{i}.values' * 256;
   targetTime = data{i}.time;
   idx = find(time == targetTime);
   gtfile = gtFilenames{idx};
   outFilename = [scoreDir, strsplit(gtfile, '/'){end}]

   D_gt  = disp_read(gtfile);

   [actY, actX] = size(D_gt)
   [y, x] = size(estData);

   assert(actY >= y && actX >= x)

   offsetY = actY - y;
   offsetX = actX - x;

   grownData = zeros(actY, actX);
   %data is set to anchor br
   grownData(offsetY+1:end, offsetX+1:end) = estData;

   d_err = disp_error(D_gt,grownData,tau);
   figure;
   handle = imshow(disp_to_color([grownData;D_gt]));
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
