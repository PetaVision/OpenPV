addpath(genpath('devkit/'));
disp('======= KITTI Depth Generator =======');

clips = [22, 23, 27, 28, 29, 32, 35, 36, 39, 46, 48, 51, 52, 56, 57, 59, ...
   60, 61, 64, 70, 79, 84, 86, 87, 91, 93, 95, 96, 101, 104, 106, 113, 117];
base_dirs = cell(length(clips), 1);
out_dirs = cell(length(clips), 1);
for i = 1:length(clips)
   base_dirs{i} = sprintf('/nh/compneuro/Data/KITTI/2011_09_26/2011_09_26_drive_%04d_sync', clips(i));
   out_dirs{i} = sprintf('/nh/compneuro/Data/newDepth/%04d', clips(i));
end

for i = 1:numel(base_dirs)
    dirname = strcat(base_dirs(i),'/image_02/data/*.png');
    num_frames = numel(dir(char(dirname))) - 1;
    velodyne(char(base_dirs(i)), char(out_dirs(i)), num_frames);
end

send_text_message('505-804-4673', 'Verizon', 'makeDepthImages.m', 'Run is done');
