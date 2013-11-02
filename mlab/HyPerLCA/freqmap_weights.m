%%%%%%%%%%%%%%%%%%%%%%
%freqmap_weights.m
%
% Dylan Paiton
% Nov 1, 2013
%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;
more off;

run_name      = 'Deep_V1';
max_patches   = 256; %Max number of patches to plot
num_hist_bins = 100;
weight_path   = '/nh/compneuro/Data/vine/LCA/2013_02_01/output_2013_02_01_12x12x128_lambda_05X2_deep';
weight_file   = [weight_path,filesep,'w4_V1ToError.pvp'];
workspace_dir = '~/workspace';
output_dir    = [weight_path,filesep,'frequency_analysis'];

addpath([workspace_dir,filesep,'PetaVision/mlab/util']);

mkdir([output_dir]);

weight_fid = fopen(weight_file);
weight_hdr = readpvpheader(weight_fid);
fclose(weight_fid);

weight_filedata   = dir(weight_file);
weight_framesize  = weight_hdr.recordsize * weight_hdr.numrecords+weight_hdr.headersize;
tot_weight_frames = weight_filedata(1).bytes/weight_framesize;

progressperiod = ceil(tot_weight_frames/10);
num_frames     = tot_weight_frames;
start_frame    = tot_weight_frames;
skip_frame     = 1;

if exist(weight_file,'file')
   [data,hdr] = readpvpfile(weight_file,progressperiod,num_frames,start_frame,skip_frame);
else
    error(['freqmap_weights: ~exist(pvp_filename,"file") in pvp file: ', weight_file]);
end

i_frame = 1;
i_arbor = 1;

weight_vals = squeeze(data{i_frame}.values{i_arbor});
weight_time = squeeze(data{i_frame}.time);

num_weight_dims = ndims(weight_vals);
num_patches     = size(weight_vals, num_weight_dims);
num_patches     = min(num_patches, max_patches);

%num_patches_rows  = floor(sqrt(num_patches));
%num_patches_cols  = ceil(num_patches / num_patches_rows);
num_weight_colors = 1;
if num_weight_dims == 4
    num_weight_colors = size(weight_vals,3);
else
    num_weight_colors = 1;
end

sum_weight_colors = true;

bin_vals = zeros(num_patches,size(weight_vals,1));

for i_patch = 1:5%num_patches
    if num_weight_colors == 1
        patch_tmp = squeeze(weight_vals(:,:,i_patch));
    else
        patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
        if sum_weight_colors
            patch_tmp = sum(patch_tmp,3);
        end
    end

    patch_fft = fft2(patch_tmp);

    center = floor(size(patch_tmp,1)/2);
    [Y,X] = meshgrid(1:size(patch_tmp,1));

    i=1;
    for radius = 0:size(patch_tmp,1)
        mask1 = sqrt((Y-center).^2+(X-center).^2)<=radius;
        mask2 = sqrt((Y-center).^2+(X-center).^2)<=(radius-1);
        mask  = mask1 - mask2;

        masked_fft = patch_fft .* mask;

        bin_vals(i_patch,i) = sum(real(masked_fft(:)).^2);

        i = i + 1;
    end
end

