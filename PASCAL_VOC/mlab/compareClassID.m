
%% Quick and dirty visualization harness for ground truth pvp files
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
output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_96_S2_1536_MLP/VOC2007_landscape5";

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
num_Recon_frames_per_layer = 1;
Recon_LIFO_flag = true;
[Recon_hdr, Recon_fig,  Recon_fig_name, Recon_vals,  Recon_time, Recon_mean,  Recon_std] = analyzeUnwhitenedReconPVP(Recon_list, num_Recon_frames_per_layer, output_dir, plot_flag, Recon_sum_list, Recon_LIFO_flag);
drawnow;

%% sparse activity
%Sparse_list ={["a2_"],  ["S1"]; ["a5_"], ["S2"]; ["a10_"], ["GroundTruth"]}; 
Sparse_list ={["a10_"], ["GroundTruth"]}; 
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
%nonSparse_list = {["a1_"], ["ImageReconS1Error"]; ["a8_"], ["ImageReconS2Error"]; ["a4_"], ["S1ReconS2Error"]; ["a12_"], ["GroundTruthError"]};%; ["a10_"], ["GroundTruthS2ReconS1Error"]};
nonSparse_list = {["a12_"], ["GroundTruthError"]};
num_nonSparse_list = size(nonSparse_list,1);
nonSparse_skip = repmat(1, num_nonSparse_list, 1);
nonSparse_skip(1) = 1;
nonSparse_norm_list = {["a0_"], ["Image"]; ["a0_"], ["Image"]; ["a2_"], ["S1"]; ["a10_"], ["GroundTruth"]};
nonSparse_norm_strength = ones(num_nonSparse_list,1);
%%nonSparse_norm_strength(1) = 1/18;
%%nonSparse_norm_strength(2) = 1/18;
%%Sparse_std_ndx = [0 0 1 3]; %% 
Sparse_std_ndx = [1]; %% 
fraction_nonSparse_frames_read = 1;
min_nonSparse_skip = 1;
fraction_nonSparse_progress = 10;
[nonSparse_times_array, nonSparse_RMS_array, nonSparse_norm_RMS_array, nonSparse_RMS_fig] = analyzeNonSparsePVP(nonSparse_list, nonSparse_skip, nonSparse_norm_list, nonSparse_norm_strength, Sparse_times_array, Sparse_std_array, Sparse_std_ndx, output_dir, plot_flag, fraction_nonSparse_frames_read, min_nonSparse_skip, fraction_nonSparse_progress);
drawnow;
				%pause;


classes={...
         'aeroplane'
         'bicycle'
         'bird'
         'boat'
         'bottle'
         'bus'
         'car'
         'cat'
         'chair'
         'cow'
         'diningtable'
         'dog'
         'horse'
         'motorbike'
         'person'
         'pottedplant'
         'sheep'
         'sofa'
         'train'
         'tvmonitor'};


%%imageRecon_fig = figure;
				%true_classID_file = fullfile("~/workspace/PASCAL_VOC/VOC2007/VOC2007_padded0_landscape_classID.pvp")
pred_classID_file = fullfile("/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_96_S2_1536_MLP/VOC2007_landscape5/a11_GroundTruthReconS2.pvp")
gt_classID_file = fullfile("/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_96_S2_1536_MLP/VOC2007_landscape5/a10_GroundTruth.pvp")
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
classID_hist_bins = -1:0.01:2;
num_classID_bins = length(classID_hist_bins);
pred_classID_hist = zeros(num_classID_bins, pred_hdr.nf,2);
%%pred_classID_sum = zeros(pred_hdr.nf, 1);
%%pred_classID_sum2 = zeros(pred_hdr.nf, 1);
for i_frame = 1 : min(pred_num_frames, gt_num_frames) 

  %% ground truth layer is sparse
  gt_time = gt_data{i_frame}.time;
  gt_num_active = length(gt_data{i_frame}.values);
  gt_active_ndx = gt_data{i_frame}.values+1;
  gt_active_sparse = sparse(gt_active_ndx,1,1,gt_num_neurons,1,gt_num_active);
  gt_classID_cube = full(gt_active_sparse);
  gt_classID_cube = reshape(gt_classID_cube, [gt_hdr.nf, gt_hdr.nx, gt_hdr.ny]);
  gt_classID_cube = permute(gt_classID_cube, [3,2,1]);

  %% only display predictions for these frames
  if i_frame >= min(pred_num_frames, gt_num_frames)-num_Recon_frames_per_layer+1 && i_frame <=  min(pred_num_frames, gt_num_frames)
    display(["i_frame = ", num2str(i_frame)]);

    [gt_classID_val, gt_classID_ndx] = max(gt_classID_cube, [], 3);
    min_gt_classID = min(gt_classID_val(:))
    %%gt_classID_cube = gt_classID_cube .* (gt_classID_cube >= min_gt_classID);
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
      gt_classID_heatmap = gt_classID_heatmap + gt_classID_band .* (gt_classID_heatmap == 0);
    endfor
    gt_classID_heatmap = mod(gt_classID_heatmap, 255);
    if plot_flag
      gt_fig = figure("name", ["Ground Truth: ", num2str(gt_time, "%i")]);
      image(uint8(gt_classID_heatmap)); axis off; axis image, box off;
      drawnow
    endif
    imwrite(uint8(gt_classID_heatmap), [output_dir, filesep, 'Recon', filesep, "gt_", num2str(gt_time, "%i"), '.png'], 'png');
  endif

  %% recon layer is not sparse
  pred_time = pred_data{i_frame}.time;
  pred_classID_cube = pred_data{i_frame}.values;
  pred_classID_cube = permute(pred_classID_cube, [2,1,3]);
  for i_classID = 1 : pred_hdr.nf
    pred_classID_tmp = squeeze(pred_classID_cube(:,:,i_classID));
    gt_classID_tmp = squeeze(gt_classID_cube(:,:,i_classID));
    pos_pred_tmp = pred_classID_tmp(gt_classID_tmp(:)~=0);
    neg_pred_tmp = pred_classID_tmp(gt_classID_tmp(:)==0);
    if any(pos_pred_tmp)
      pred_classID_hist(:,i_classID,1) = squeeze(pred_classID_hist(:,i_classID,1)) + hist(pos_pred_tmp(:), classID_hist_bins)';
    endif
    if any(neg_pred_tmp)
      pred_classID_hist(:,i_classID,2) = squeeze(pred_classID_hist(:,i_classID,2)) + hist(neg_pred_tmp(:), classID_hist_bins)';
    endif
  endfor
  %%pred_classID_sum = pred_classID_sum + squeeze(sum(squeeze(sum(pred_classID_cube, 1)),1))';
  %%pred_classID_sum2 = pred_classID_sum2 + (squeeze(sum(squeeze(sum(pred_classID_cube, 1)),1)).^2)';
  if i_frame >= min(pred_num_frames, gt_num_frames)-num_Recon_frames_per_layer+1 && i_frame <=  min(pred_num_frames, gt_num_frames)
    pred_classID_cumsum = squeeze(cumsum(pred_classID_hist, 1));
    pred_classID_sum = squeeze(sum(pred_classID_hist, 1));
    pred_classID_norm = repmat(reshape(pred_classID_sum, [1,pred_hdr.nf,2]), [num_classID_bins,1,1]);
    pred_classID_cumprob = pred_classID_cumsum ./ (pred_classID_norm + (pred_classID_norm==0));
    pred_classID_thresh_bin = zeros(pred_hdr.nf,1);
    pred_classID_thresh = zeros(pred_hdr.nf,1);
    pred_classID_true_pos = zeros(pred_hdr.nf,1);
    pred_classID_false_pos = zeros(pred_hdr.nf,1);
    for i_classID = 1 : pred_hdr.nf
      pred_classID_bin_tmp = find((squeeze(pred_classID_cumprob(:,i_classID,1))>squeeze(pred_classID_cumprob(:,i_classID,2))), 1, "first");
      if ~isempty(pred_classID_bin_tmp)
	pred_classID_thresh_bin(i_classID) = pred_classID_bin_tmp;
	pred_classID_thresh(i_classID) = classID_hist_bins(pred_classID_thresh_bin(i_classID));
	pred_classID_true_pos(i_classID) = (1 - pred_classID_cumprob(pred_classID_bin_tmp,i_classID,1));
	pred_classID_false_pos(i_classID) = (1 - pred_classID_cumprob(pred_classID_bin_tmp,i_classID,2));
      else
	pred_classID_thresh(i_classID) = classID_hist_bins(end);
	pred_classID_true_pos(i_classID) = 0.0;
	pred_classID_false_pos(i_classID) = 0.0;
      endif
    endfor
    %%pred_classID_ave = pred_classID_sum / (i_frame * pred_hdr.nx * pred_hdr.ny);
    %%pred_classID_std = sqrt((pred_classID_sum2 / (i_frame * pred_hdr.nx * pred_hdr.ny)) - (pred_classID_ave.^2));
    [pred_classID_val, pred_classID_ndx] = max(pred_classID_cube, [], 3);
    min_pred_classID = min(pred_classID_val(:))
    [max_pred_classID, max_pred_classID_ndx] = max(pred_classID_val(:))
    disp(classes{pred_classID_ndx(max_pred_classID_ndx)});
    mean_pred_classID = mean(pred_classID_val(:))
    std_pred_classID = std(pred_classID_val(:))
    %pred_classID_mask = double(pred_classID_cube >= (mean_pred_classID-1*std_pred_classID));
    %%pred_classID_thresh = zeros(1,1,pred_hdr.nf);
    %%pred_classID_thresh(1,1,:) = pred_classID_ave+0*pred_classID_std
    pred_classID_thresh = reshape(pred_classID_thresh, [1,1, pred_hdr.nf]);
    pred_classID_mask = double(pred_classID_cube >= repmat(pred_classID_thresh, [pred_hdr.ny, pred_hdr.nx, 1]));
    pred_classID_confidences = cell(pred_hdr.nf, 1);
    pred_classID_max_confidence = squeeze(max(squeeze(max(pred_classID_cube, [], 2)), [], 1));
    [pred_classID_sorted_confidence, pred_classID_sorted_ndx] = sort(pred_classID_max_confidence, 'descend');
    for i_pred_classID = 1 : pred_hdr.nf
      pred_classID_confidences{i_pred_classID, 1} = [classes{pred_classID_sorted_ndx(i_pred_classID)}, ...
						     ', ', num2str(pred_classID_sorted_confidence(i_pred_classID)), ...
						     ', ', num2str(pred_classID_thresh(pred_classID_sorted_ndx(i_pred_classID)))];
    endfor
    if plot_flag
      pred_classID_heatmap = zeros(pred_hdr.ny, pred_hdr.nx, 3);
      pred_fig = figure("name", ["Predict: ", num2str(pred_time, "%i")]);
      image(uint8(pred_classID_heatmap)); axis off; axis image, box off;
      hold on
      for i_classID = 1 : pred_hdr.nf
	if ~any(pred_classID_mask(:,:,i_classID))
	  continue;
	endif
	pred_class_color_code = i_classID * num_colors / pred_hdr.nf;
	pred_class_color = getClassColor(pred_class_color_code);
	pred_classID_band = repmat(pred_classID_mask(:,:,i_classID), [1,1,3]);
	pred_classID_band(:,:,1) = pred_classID_band(:,:,1) * pred_class_color(1);
	pred_classID_band(:,:,2) = pred_classID_band(:,:,2) * pred_class_color(2);
	pred_classID_band(:,:,3) = pred_classID_band(:,:,3) * pred_class_color(3);
	pred_classID_heatmap = pred_classID_heatmap + pred_classID_band .* (pred_classID_heatmap < pred_classID_band);
	th = text(1, ceil(i_classID*pred_hdr.ny/pred_hdr.nf), classes{i_classID}, 'color', pred_class_color/255);
				%keyboard;
      endfor
      pred_classID_heatmap = mod(pred_classID_heatmap, 255);
      image(uint8(pred_classID_heatmap)); axis off; axis image, box off;
      drawnow
      %%get(th)
    endif %% plot_flag
    if plot_flag
      hist_fig = figure("name", ["hist_positive: ", num2str(pred_time, "%i")]);
      for i_classID = 1 : pred_hdr.nf
	subplot(4,5,i_classID)
	bh_pos = bar(classID_hist_bins, squeeze(pred_classID_hist(:,i_classID,1)) ./ squeeze(pred_classID_norm(:,i_classID,1)), "stacked", "facecolor", "g", "edgecolor", "g");
	%%hist_fig = figure("name", ["hist_negative: ", num2str(pred_time, "%i")]);
	axis off
	box off
	hold on
	bh_neg = bar(classID_hist_bins, squeeze(pred_classID_hist(:,i_classID,2)) ./ squeeze(pred_classID_norm(:,i_classID,2)), "stacked", "facecolor", "r", "edgecolor", "r");
	title(classes{i_classID});
      endfor
    endif
    imwrite(uint8(pred_classID_heatmap), [output_dir, filesep, 'Recon', filesep, "pred_", num2str(pred_time, "%i"), '.png'], 'png');
    disp(pred_classID_confidences)
    disp([pred_classID_true_pos; pred_classID_false_pos])
    save([output_dir, filesep, 'Recon', filesep, "hist_", num2str(pred_time, "%i"), ".mat"], "classID_hist_bins", "pred_classID_hist", "pred_classID_norm", "pred_classID_cumprob", "pred_classID_cumsum")
    %keyboard
  endif

endfor
