 
clear all;
close all;

%% Initialization
%% File Paths, and useful variables

workspace_path = "/Users/bcrocker/Documents/workspace";
output_dir = [workspace_path "/HyPerHLCA2/output_britni_label_MNIST"]; 
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
checkpoint_dir = [output_dir, filesep, "Checkpoints"];
last_checkpoint_ndx = max(str2num(ls(checkpoint_dir)(:,length(checkpoint_dir)+11:end)));
checkpoint_path = [checkpoint_dir, filesep, "Checkpoint", num2str(last_checkpoint_ndx, "%i")];
recon_dir = [output_dir, filesep, "recon"];
stats_dir = [output_dir, filesep, "Stats"];
V1_dir = [output_dir, filesep, "V1"];
mkdir(recon_dir);

layers = ls([output_dir "/a*"]);
weights = ls([output_dir "/w*"]);
error_stats = ls([output_dir "/*Error_Stats.txt"]);

%% make some plots of the weights
for w = 1:size(weights,1)
    drawpvp_weights(weights(w,:), recon_dir);
end

drawpvp_MNIST_weights(weights, recon_dir);

%% write the activity from the various groups
for l = 1:size(layers,1)
    drawpvp_activity(layers(l,:),[], recon_dir);
end

%% plot error over time
for e = 1:size(error_stats,1)
    plotstats(error_stats(e,:),"sigma",0);
end

%% plot V1 sparsity over time
plotstats([output_dir "/V1_Stats.txt"],"nnz",1, stats_dir);

%% plot some V1 stuff?
%%V1plots(V1_file,Error_file)




%%function v1plots(v1_file,error_file, v1_dir)

function V1plots(V1_file,Error_file, V1_dir)

  %%%%%% PATHS %%%%%%%%%%
  [V1_struct, V1_hdr] = readpvpfile(V1_file, [], [], []);
  n_V1 = V1_hdr.nx * V1_hdr.ny * V1_hdr.nf;
  num_frames = size(V1_struct,1);
  start_frame = 1; %%
  
  %%%%%%%% INITIALIZATION %%%%%%%%%%%%%
  V1_hist = zeros(V1_hdr.nf+1,1);
  V1_hist_edges = [0:1:V1_hdr.nf]+0.5;
  V1_current = zeros(n_V1,1);
  V1_abs_change = zeros(num_frames,1);
  V1_percent_change = zeros(num_frames,1);
  V1_current_active = 0;
  V1_tot_active = zeros(num_frames,1);
  V1_times = zeros(num_frames,1);

  %%%%%%%% CALCULATIONS: ACTIVITY OVER TIME %%%%%%%%%%%%%%%
  for i_frame = 1 : 1 : num_frames
    V1_times(i_frame) = squeeze(V1_struct{i_frame}.time);
    V1_active_ndx = squeeze(V1_struct{i_frame}.values);
    V1_previous = V1_current;
    V1_current = full(sparse(V1_active_ndx+1,1,1,n_V1,1,n_V1));
    V1_abs_change(i_frame) = sum(V1_current(:) ~= V1_previous(:));
    V1_previous_active = V1_current_active;
    V1_current_active = nnz(V1_current(:));
    V1_tot_active(i_frame) = V1_current_active;
    V1_max_active = max(V1_current_active, V1_previous_active);
    V1_percent_change(i_frame) = ...
	V1_abs_change(i_frame) / (V1_max_active + (V1_max_active==0));
    V1_active_kf = mod(V1_active_ndx, V1_hdr.nf) + 1;
    if V1_max_active > 0
      V1_hist_frame = histc(V1_active_kf, V1_hist_edges);
    else
      V1_hist_frame = zeros(V1_hdr.nf+1,1);
    endif
    V1_hist = V1_hist + V1_hist_frame;
  endfor %% i_frame

  %%%%%%%%%% PLOTS: ACTIVITY %%%%%%%%%%%%%%%%%
  V1_hist = V1_hist(1:V1_hdr.nf);
  V1_hist = V1_hist / (num_frames * V1_hdr.nx * V1_hdr.ny); %% (sum(V1_hist(:)) + (nnz(V1_hist)==0));
  [V1_hist_sorted, V1_hist_rank] = sort(V1_hist, 1, "descend");
  V1_hist_title = ["V1_hist", ".png"];
  V1_hist_fig = figure;
  V1_hist_bins = 1:V1_hdr.nf;
  V1_hist_hndl = bar(V1_hist_bins, V1_hist_sorted); axis tight;
  set(V1_hist_fig, "name", ["V1_hist_", num2str(V1_times(num_frames), "%i")]);
  mkdir(V1_dir);
  saveas(V1_hist_fig, ...
	 [V1_dir, filesep, ...
	  "V1_rank_", num2str(V1_times(num_frames), "%i")], "png");

  V1_abs_change_title = ["V1_abs_change", ".png"];
  V1_abs_change_fig = figure;
  V1_abs_change_hndl = plot(V1_times, V1_abs_change); axis tight;
  set(V1_abs_change_fig, "name", ["V1_abs_change"]);
  saveas(V1_abs_change_fig, ...
	 [V1_dir, filesep, "V1_abs_change", num2str(V1_times(num_frames), "%i")], "png");

  V1_percent_change_title = ["V1_percent_change", ".png"];
  V1_percent_change_fig = figure;
  V1_percent_change_hndl = plot(V1_times, V1_percent_change); axis tight;
  set(V1_percent_change_fig, "name", ["V1_percent_change"]);
  saveas(V1_percent_change_fig, ...
	 [V1_dir, filesep, "V1_percent_change", num2str(V1_times(num_frames), "%i")], "png");
  V1_mean_change = mean(V1_percent_change(:));
  disp(["V1_mean_change = ", num2str(V1_mean_change)]);

  V1_tot_active_title = ["V1_tot_active", ".png"];
  V1_tot_active_fig = figure;
  V1_tot_active_hndl = plot(V1_times, V1_tot_active/n_V1); axis tight;
  set(V1_tot_active_fig, "name", ["V1_tot_active"]);
  saveas(V1_tot_active_fig, ...
	 [V1_dir, filesep, "V1_tot_active", num2str(V1_times(num_frames), "%i")], "png");

  V1_mean_active = mean(V1_tot_active(:)/n_V1);
  disp(["V1_mean_active = ", num2str(V1_mean_active)]);

  %%%%%%%%%% ERROR PLOTS %%%%%%%%%%%%%%
  [Error_struct, Error_hdr] = readpvpfile(Error_file, [], [], []);
  num_frames = size(Error_struct,1);
  i_frame = num_frames;
  start_frame = 1; %%
  Error_RMS = zeros(num_frames,1);
  Error_times = zeros(num_frames,1);
  for i_frame = 1 : 1 : num_frames
    Error_times(i_frame) = squeeze(Error_struct{i_frame}.time);
    Error_vals = squeeze(Error_struct{i_frame}.values);
    Error_RMS(i_frame) = std(Error_vals(:));;
  endfor %% i_frame

  Error_RMS_title = ["Error_RMS", ".png"];
  Error_RMS_fig = figure;
  Error_RMS_hndl = plot(Error_times, Error_RMS); axis tight;
  set(Error_RMS_fig, "name", ["Error_RMS"]);
  saveas(Error_RMS_fig, ...
	 [V1_dir, filesep, "Error_RMS", num2str(Error_times(num_frames), "%i")], "png");

  Error_median_RMS = median(Error_RMS(:));
  disp(["Error_median_RMS = ", num2str(Error_median_RMS)]);
  drawnow;  

  %%%%%%%%%% V1toERROR PLOTS %%%%%%%%%%%%%
  V1ToError_path = [checkpoint_path, filesep, "V1ToError_W.pvp"];
  [V1ToError_struct, V1ToError_hdr] = readpvpfile(V1ToError_path,1);
  i_frame = 1;
  i_arbor = 1;
  V1ToError_weights = squeeze(V1ToError_struct{i_frame}.values{i_arbor});
  if ~exist("V1_hist_rank") || isempty(V1_hist_rank)
    V1_hist_rank = (1:V1ToError_hdr.nf);
  endif

  %% make tableau of all patches
  %%keyboard;
  i_patch = 1;
  num_V1_dims = ndims(V1ToError_weights);
  num_patches = size(V1ToError_weights, num_V1_dims);
  num_patches_rows = floor(sqrt(num_patches));
  num_patches_cols = ceil(num_patches / num_patches_rows);
  num_V1_colors = 1;
  if num_V1_dims == 4
    num_V1_colors = size(V1ToError_weights,3);
  endif
  V1ToError_fig = figure;
  set(V1ToError_fig, "name", "V1ToError Weights");
  for j_patch = 1  : num_patches
    i_patch = V1_hist_rank(j_patch);
    subplot(num_patches_rows, num_patches_cols, j_patch); 
    if num_V1_colors == 1
      patch_tmp = squeeze(V1ToError_weights(:,:,i_patch));
    else
      patch_tmp = squeeze(V1ToError_weights(:,:,:,i_patch));
    endif
    patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
    min_patch = min(patch_tmp2(:));
    max_patch = max(patch_tmp2(:));
    patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
    patch_tmp2 = uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
    imagesc(patch_tmp2); 
    if num_V1_colors == 1
      colormap(gray);
    endif
    box off
    axis off
    axis image
    %%drawnow;
  endfor
  saveas(V1ToError_fig, [V1_dir, filesep, "V1ToError"], "png");


  %% make histogram of all weights
  V1ToError_hist_fig = figure;
  [V1ToError_hist, V1ToError_hist_bins] = hist(V1ToError_weights(:), 100);
  bar(V1ToError_hist_bins, log(V1ToError_hist+1));
  set(V1ToError_hist_fig, "name", "V1ToError Histogram");
  V1ToError_weights_hist_dir = [output_dir, filesep, "V1ToError_hist"];
  saveas(V1ToError_hist_fig, [V1_dir, filesep, "V1ToError_hist"], "png");

  V1ToError_weights_file = [V1_dir, filesep, "V1ToError_weights", ".mat"];
  save(  "-mat", V1ToError_weights_file, "V1ToError_weights");



  weights_movie_dir = [V1_dir, filesep, "V1ToError_movie"];
  mkdir(weights_movie_dir);
  V1ToError_path = [output_dir, filesep, "w5_V1ToError.pvp"];
  [V1ToError_struct, V1ToError_hdr] = readpvpfile(V1ToError_path, []);
  num_frames = size(V1ToError_struct,1);
  i_frame = num_frames;
  if isempty(V1_hist_rank)
    V1_hist_rank = (1:V1ToError_hdr.nf);
  endif
  i_arbor = 1;
  for i_frame = 1 : num_frames
    if mod(i_frame, max(floor(num_frames/20),1)) == 0
      disp(["writing frame # ", num2str(i_frame, "%i")]);
    endif
    V1ToError_weights = squeeze(V1ToError_struct{i_frame}.values{i_arbor});
    i_patch = 1;
    nyp = size(V1ToError_weights,1);
    nxp = size(V1ToError_weights,2);
    num_V1_dims = ndims(V1ToError_weights);
    num_patches = size(V1ToError_weights, num_V1_dims);
    num_patches_rows = floor(sqrt(num_patches));
    num_patches_cols = ceil(num_patches / num_patches_rows);
    num_V1_colors = 1;
    if num_V1_dims == 4
      num_V1_colors = size(V1ToError_weights,3);
    endif
    weights_frame = uint8(zeros(num_patches_rows * nyp, num_patches_cols * nxp, num_V1_colors));
    for j_patch = 1  : num_patches
      i_patch = V1_hist_rank(j_patch);
      j_patch_row = ceil(j_patch / num_patches_cols);
      j_patch_col = 1 + mod(j_patch - 1, num_patches_cols);
      %%subplot(num_patches_rows, num_patches_cols, i_patch); 
      if num_V1_colors == 1
	patch_tmp = squeeze(V1ToError_weights(:,:,i_patch));
      else
	patch_tmp = squeeze(V1ToError_weights(:,:,:,i_patch));
      endif
      min_patch = min(patch_tmp(:));
      max_patch = max(patch_tmp(:));
      patch_tmp2 = (patch_tmp - min_patch) * 255 / ((max_patch - min_patch) + ((max_patch - min_patch)==0));
      patch_tmp2 = uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
      patch_tmp2 = uint8(patch_tmp2);
      weights_frame(((j_patch_row - 1) * nyp + 1): (j_patch_row * nyp), ...
		    ((j_patch_col - 1) * nxp + 1): (j_patch_col * nxp), :) = ...
	  patch_tmp2;
      %%imagesc(patch_tmp2);
      box off
      axis off
    endfor  %% i_patch
    frame_title = [num2str(i_frame, "%04d"), "_V1ToError", ".png"];
    imwrite(weights_frame, [weights_movie_dir, filesep, frame_title]);
  endfor %% i_frame
endfunction





