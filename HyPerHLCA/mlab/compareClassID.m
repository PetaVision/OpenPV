
%% quick and dirty visualization harness for ground truth pvp files
%% each classID is assigned a different color
%% where bounding boxes overlapp, the color is a mixture
%% use this script to visualize ground truth sparse pvp files and for comparison with 
%% original images to verify that the bounding box annotations are reasonable
%% hit any key to advance to the next image
close all
pkg load all
setenv("GNUTERM","X11")
addpath("~/workspace/PetaVision/mlab/imgProc");
addpath("~/workspace/PetaVision/mlab/util");
addpath("~/workspace/PetaVision/mlab/HyPerLCA");

plot_flag = true;
output_dir = "~/workspace/PASCAL_VOC/PASCAL_S1_96_S2_384_MLP/VOC2007_landscape";

%%draw reconstructed image
DoG_weights = [];
Recon_list = {["a3_"],  ["ImageReconS1"]; ["a7_"],  ["ImageReconS2"]};
%% list of layers to unwhiten
num_Recon_list = size(Recon_list,1);
Recon_unwhiten_list = zeros(num_Recon_list,1);
%% list of layers to use as a normalization reference for unwhitening
Recon_normalize_list = 1:num_Recon_list;
%% list of (previous) layers to sum with current layer
Recon_sum_list = cell(num_Recon_list,1);
num_Recon_frames_per_layer = 4;
Recon_LIFO_flag = true;
[Recon_hdr, Recon_fig,  Recon_fig_name, Recon_vals,  Recon_time, Recon_mean,  Recon_std] = analyzeUnwhitenedReconPVP(Recon_list, num_Recon_frames_per_layer, output_dir, plot_flag, Recon_sum_list, Recon_LIFO_flag);
drawnow;

%% sparse activity
Sparse_list ={["a2_"],  ["S1"]; ["a5_"], ["S2"]}; %%; ["a5_"], ["GroundTruth"]}; 
fraction_Sparse_frames_read = 1;
min_Sparse_skip = 1;
fraction_Sparse_progress = 10;
num_epochs = 1;
num_procs = 1;
Sparse_frames_list = [];
load_Sparse_flag = false;
[Sparse_hdr, Sparse_hist_rank_array, Sparse_times_array, Sparse_percent_active_array, Sparse_percent_change_array, Sparse_std_array, Sparse_struct_array] = analyzeSparseEpochsPVP2(Sparse_list, output_dir, load_Sparse_flag, plot_flag, fraction_Sparse_frames_read, min_Sparse_skip, fraction_Sparse_progress, Sparse_frames_list, num_procs, num_epochs);
drawnow;
%pause;

%% Error vs time
nonSparse_list = {["a1_"], ["ImageReconS1Error"]; ["a8_"], ["ImageReconS2Error"]; ["a4_"], ["S1ReconS2Error"]}; %%; ["a9_"], ["GroundTruthReconS2Error"]; ["a10_"], ["GroundTruthS2ReconS1Error"]};
num_nonSparse_list = size(nonSparse_list,1);
nonSparse_skip = repmat(1, num_nonSparse_list, 1);
nonSparse_skip(1) = 1;
nonSparse_norm_list = {["a0_"], ["Image"]; ["a2_"], ["S1"]; ["a7_"], ["S2"]; ["a2_"], ["S1"]};
nonSparse_norm_strength = ones(num_nonSparse_list,1);
nonSparse_norm_strength(1) = 1/18;
Sparse_std_ndx = [0 0 1 3 3]; %% 
fraction_nonSparse_frames_read = 1;
min_nonSparse_skip = 1;
fraction_nonSparse_progress = 10;
[nonSparse_times_array, nonSparse_RMS_array, nonSparse_norm_RMS_array, nonSparse_RMS_fig] = analyzeNonSparsePVP(nonSparse_list, nonSparse_skip, nonSparse_norm_list, nonSparse_norm_strength, Sparse_times_array, Sparse_std_array, Sparse_std_ndx, output_dir, plot_flag, fraction_nonSparse_frames_read, min_nonSparse_skip, fraction_nonSparse_progress);
drawnow;
%pause;




if plot_flag
  %true_fig = figure;
  pred_fig = figure;
  gt_fig = figure;
endif
%%imageRecon_fig = figure;
%true_classID_file = fullfile("~/workspace/PASCAL_VOC/VOC2007/VOC2007_padded0_landscape_classID.pvp")
pred_classID_file = fullfile("~/workspace/PASCAL_VOC/PASCAL_S1_96_S2_384_MLP/VOC2007_landscape/a8_GroundTruthReconS2.pvp")
gt_classID_file = fullfile("~/workspace/PASCAL_VOC/PASCAL_S1_96_S2_384_MLP/VOC2007_landscape/a5_GroundTruth.pvp")
%[true_data,true_hdr] = readpvpfile(true_classID_file); 
[pred_data,pred_hdr] = readpvpfile(pred_classID_file); 
[gt_data,gt_hdr] = readpvpfile(gt_classID_file); 
%%[imageRecon_data,imageRecon_hdr] = readpvpfile(imageRecon_file); 
%true_num_neurons = true_hdr.nf * true_hdr.nx * true_hdr.ny;
%true_num_frames = length(true_data);
pred_num_neurons = pred_hdr.nf * pred_hdr.nx * pred_hdr.ny;
pred_num_frames = length(pred_data);
gt_num_neurons = gt_hdr.nf * gt_hdr.nx * gt_hdr.ny;
gt_num_frames = length(gt_data);
%%imageRecon_num_neurons = imageRecon_hdr.nf * imageRecon_hdr.nx * imageRecon_hdr.ny;
%%imageRecon_num_frames = length(imageRecon_data);
num_colors = 2^24;
accuracy_vs_time = zeros(gt_num_frames,1);
confusion_matrix = zeros(gt_hdr.nf);
for i_frame = min(pred_num_frames, gt_num_frames): -1: min(pred_num_frames, gt_num_frames)-4
    display(["i_frame = ", num2str(i_frame)]);
    %%true_num_active = length(true_data{i_frame}.values);
    %%true_active_ndx = true_data{i_frame}.values+1;
    %%true_active_sparse = sparse(true_active_ndx,1,1,true_num_neurons,1,true_num_active);
    %%true_classID_cube = full(true_active_sparse);
    %%true_classID_cube = reshape(true_classID_cube, [true_hdr.nf, true_hdr.nx, true_hdr.ny]);
    %%true_classID_cube = permute(true_classID_cube, [3,2,1]);
    %%true_classID_heatmap = zeros(true_hdr.ny, true_hdr.nx, 3);
    %%for i_true_classID = 1 : true_hdr.nf
    %%	if ~any(true_classID_cube(:,:,i_true_classID))
    %%	   continue;
    %%	endif
    %%	true_class_color_code = i_true_classID * num_colors / true_hdr.nf;
    %%	true_class_color = getClassColor(true_class_color_code);
    %%	true_classID_band = repmat(true_classID_cube(:,:,i_true_classID), [1,1,3]);
    %%	true_classID_band(:,:,1) = true_classID_band(:,:,1) * true_class_color(1);
    %%	true_classID_band(:,:,2) = true_classID_band(:,:,2) * true_class_color(2);
    %%	true_classID_band(:,:,3) = true_classID_band(:,:,3) * true_class_color(3);
    %%	true_classID_heatmap = true_classID_heatmap + true_classID_band;
    %%endfor
    %%true_classID_heatmap = mod(true_classID_heatmap, 255);
    %%if plot_flag
    %%  figure(true_fig);
    %%  image(uint8(true_classID_heatmap)); axis off; axis image, box off;
    %%  drawnow
    %%endif

    %% ground truth layer is sparse
    %%gt_classID_cube = gt_data{i_frame}.values;
    gt_num_active = length(gt_data{i_frame}.values);
    gt_active_ndx = gt_data{i_frame}.values+1;
    gt_active_sparse = sparse(gt_active_ndx,1,1,gt_num_neurons,1,gt_num_active);
    gt_classID_cube = full(gt_active_sparse);
    gt_classID_cube = reshape(gt_classID_cube, [gt_hdr.nf, gt_hdr.nx, gt_hdr.ny]);
    gt_classID_cube = permute(gt_classID_cube, [3,2,1]);

    %%gt_classID_cube = permute(gt_classID_cube, [2,1,3]);
    [gt_classID_val, gt_classID_ndx] = max(gt_classID_cube, [], 3);
    min_gt_classID = min(gt_classID_val(:))
    gt_classID_cube = gt_classID_cube .* (gt_classID_cube >= min_gt_classID);
    gt_classID_heatmap = zeros(gt_hdr.ny, gt_hdr.nx, 3);
    for i_gt_classID = 1 : gt_hdr.nf
	if ~any(gt_classID_cube(:,:,i_gt_classID))
	   continue;
	endif
	gt_class_color_code = i_gt_classID * num_colors / gt_hdr.nf;
	gt_class_color = getClassColor(gt_class_color_code);
	gt_classID_band = repmat(gt_classID_cube(:,:,i_gt_classID), [1,1,3]);
	gt_classID_band(:,:,1) = gt_classID_band(:,:,1) * gt_class_color(1);
	gt_classID_band(:,:,2) = gt_classID_band(:,:,2) * gt_class_color(2);
	gt_classID_band(:,:,3) = gt_classID_band(:,:,3) * gt_class_color(3);
	gt_classID_heatmap = gt_classID_heatmap + gt_classID_band;
    endfor
    gt_classID_heatmap = mod(gt_classID_heatmap, 255);
    if plot_flag
      figure(gt_fig);
      image(uint8(gt_classID_heatmap)); axis off; axis image, box off;
      drawnow
    endif

    %% recon layer is not sparse
    pred_classID_cube = pred_data{i_frame}.values;
    pred_classID_cube = permute(pred_classID_cube, [2,1,3]);
    [pred_classID_val, pred_classID_ndx] = max(pred_classID_cube, [], 3);
    %%min_pred_classID = min(pred_classID_val(:))
    max_pred_classID = max(pred_classID_val(:))
    %%mean_pred_classID = mean(pred_classID_val(:));
    %%std_pred_classID = std(pred_classID_val(:));
    pred_classID_cube = double(pred_classID_cube >= (max_pred_classID*(.999)));
    %%pred_classID_cube = double(pred_classID_cube >= (mean_pred_classID+std_pred_classID));
    pred_classID_heatmap = zeros(pred_hdr.ny, pred_hdr.nx, 3);
    for i_pred_classID = 1 : pred_hdr.nf
	if ~any(pred_classID_cube(:,:,i_pred_classID))
	   continue;
	endif
	pred_class_color_code = i_pred_classID * num_colors / pred_hdr.nf;
	pred_class_color = getClassColor(pred_class_color_code);
	pred_classID_band = repmat(pred_classID_cube(:,:,i_pred_classID), [1,1,3]);
	pred_classID_band(:,:,1) = pred_classID_band(:,:,1) * pred_class_color(1);
	pred_classID_band(:,:,2) = pred_classID_band(:,:,2) * pred_class_color(2);
	pred_classID_band(:,:,3) = pred_classID_band(:,:,3) * pred_class_color(3);
	pred_classID_heatmap = pred_classID_heatmap + pred_classID_band;
	%keyboard;
    endfor
    pred_classID_heatmap = mod(pred_classID_heatmap, 255);
    %%min_pred_heatmap = min(pred_classID_heatmap(:));
    %%max_pred_heatmap = max(pred_classID_heatmap(:));
    %pred_classID_heatmap = 255 * (pred_classID_heatmap - min_pred_heatmap) / ((max_pred_heatmap - min_pred_heatmap) + (max_pred_heatmap == min_pred_heatmap));
    if plot_flag
      figure(pred_fig);
      image(uint8(pred_classID_heatmap)); axis off; axis image, box off;
      drawnow
    endif

    %% reconImage layer
    %%imageRecon_cube = imageRecon_data{i_frame}.values;
    %%imageRecon_cube = permute(imageRecon_cube, [2,1,3]);
    %%min_imageRecon = min(imageRecon_cube(:));
    %%max_imageRecon = max(imageRecon_cube(:));
    %%imageRecon_cube = (imageRecon_cube - min_imageRecon) / ((max_imageRecon - min_imageRecon) + (max_imageRecon == min_imageRecon));
    %%figure(imageRecon_fig);
    %%imagesc(imageRecon_cube); axis off; axis image, box off;
    %%drawnow

   %% keyboard;
    if plot_flag
      pause;
    endif
    

endfor
