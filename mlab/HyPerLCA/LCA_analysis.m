
clear all;
close all;
setenv("GNUTERM","X11")
%%workspace_path = "/home/gkenyon/workspace";
workspace_path = "/Users/garkenyon/workspace";
%%output_dir = "/nh/compneuro/Data/vine/LCA/cats"; 
output_dir = "/Users/garkenyon/workspace/HyPerHLCA2/output"
%%LCA_path = [output_dir];
LCA_path = [workspace_path, filesep, "HyPerHLCA2"];
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
last_checkpoint_ndx = 250000; %%20706*59; %%
first_checkpoint_ndx = 0; %%600000;
%%last_checkpoint_ndx = 20706*60; %%
%%first_checkpoint_ndx = 0;
use_Last_flag = 0;
if use_Last_flag
  checkpoint_dir = [output_dir, filesep, "Last"];
  checkpoint_path = [checkpoint_dir]
else
  checkpoint_dir = [LCA_path, filesep, "Checkpoints"];
  checkpoint_path = [checkpoint_dir, filesep, "Checkpoint", num2str(last_checkpoint_ndx, "%i")];
endif
max_lines = last_checkpoint_ndx + (last_checkpoint_ndx == 0) * 1000;
max_history = 50000;
begin_statProbe_step = max(max_lines - max_history, 3);
frame_duration = 1000;


%% get DoG kernel
plot_DoG_kernel = 1;
if plot_DoG_kernel
  i_frame = 1;
  i_arbor = 1;
  i_patch = 1;
  DoG_center_path = [checkpoint_path, filesep, "RetinaToGanglionCenter_W.pvp"];
  [DoG_center_struct, DoG_center_hdr] = readpvpfile(DoG_center_path,1);
  DoG_center_weights = squeeze(DoG_center_struct{i_frame}.values{i_arbor});
  DoG_surround_path = [checkpoint_path, filesep, "RetinaToGanglionSurround_W.pvp"];
  [DoG_surround_struct, DoG_surround_hdr] = readpvpfile(DoG_surround_path,1);
  DoG_surround_weights = squeeze(DoG_surround_struct{i_frame}.values{i_arbor});
  DoG_pad = (size(DoG_surround_weights) - size(DoG_center_weights)) / 2;
  DoG_center_padded = zeros(size(DoG_surround_weights));
  DoG_row_start = DoG_pad(1)+1;
  DoG_row_stop = size(DoG_surround_weights,1)-DoG_pad(1);
  DoG_col_start = DoG_pad(2)+1;
  DoG_col_stop = size(DoG_surround_weights,2)-DoG_pad(2);
  DoG_center_padded(DoG_row_start:DoG_row_stop, DoG_col_start:DoG_col_stop) = ...
      DoG_center_weights;
  DoG_weights = ...
      DoG_center_padded - DoG_surround_weights;
  DoG_fig = figure;
  set(DoG_fig, "name", "DoG Weights");
  patch_tmp = DoG_weights;
  patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
  min_patch = min(patch_tmp2(:));
  max_patch = max(patch_tmp2(:));
  patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch);
  patch_tmp2 = uint8(patch_tmp2);
  imagesc(patch_tmp2); colormap(gray);
  box off
  axis off
    %%drawnow;
  saveas(DoG_fig, [output_dir, filesep, "DoG_weights.png"]);
endif


plot_Recon = 1;
if plot_Recon
  %%keyboard;
  Retina_file = [output_dir, filesep, "a1_Retina.pvp"];
  Ganglion_file = [output_dir, filesep, "a2_Ganglion.pvp"];
  Recon_file = [output_dir, filesep, "a3_Recon.pvp"];
  Error_file = [output_dir, filesep, "a4_Error.pvp"];
  write_step = frame_duration;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  start_frame = []; %%1; %% floor((first_checkpoint_ndx) / write_step);
  [Retina_struct, Retina_hdr] = readpvpfile(Retina_file, num_frames, num_frames, start_frame);
  [Ganglion_struct, Ganglion_hdr] = readpvpfile(Ganglion_file, num_frames, num_frames, start_frame);
  [Recon_struct, Recon_hdr] = readpvpfile(Recon_file, num_frames, num_frames, start_frame);
  [Error_struct, Error_hdr] = readpvpfile(Error_file, num_frames, num_frames, start_frame);
  num_frames = size(Retina_struct,1)-1;
  i_frame = num_frames;
  start_frame = num_frames; %% floor(last_checkpoint_ndx / write_step);
  if start_frame > num_frames
    start_frame = num_frames
  endif
  for i_frame = start_frame : 1 : num_frames
    Retina_time = Retina_struct{i_frame}.time;
    Retina_vals = Retina_struct{i_frame}.values;
    Retina_fig(i_frame) = figure;
    set(Retina_fig(i_frame), "name", ["Retina ", num2str(i_frame)]);
    imagesc(Retina_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Retina_fig(i_frame), [output_dir, filesep, "Retina_", num2str(i_frame, "%i")], "png");

    Ganglion_time = Ganglion_struct{i_frame}.time;
    Ganglion_vals = Ganglion_struct{i_frame}.values;
    Ganglion_fig(i_frame) = figure;
    set(Ganglion_fig(i_frame), "name", ["Ganglion ", num2str(i_frame)]);
    imagesc(Ganglion_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Ganglion_fig(i_frame), [output_dir, filesep, "Ganglion_", num2str(i_frame, "%i")], "png");

    Error_time = Error_struct{i_frame}.time;
    Error_vals = Error_struct{i_frame}.values;
    Error_fig(i_frame) = figure;
    set(Error_fig(i_frame), "name", ["Error ", num2str(i_frame)]);
    imagesc(Error_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Error_fig(i_frame), [output_dir, filesep, "Error_", num2str(i_frame, "%i")], "png");

    Recon_time = Recon_struct{i_frame}.time;
    Recon_vals = Recon_struct{i_frame}.values;
    Recon_fig(i_frame) = figure;
    set(Recon_fig(i_frame), "name", ["Recon ", num2str(i_frame)]);
    imagesc(Recon_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Recon_fig(i_frame), [output_dir, filesep, "Recon_", num2str(i_frame, "%i")], "png");


    if plot_DoG_kernel
      max_Error = max(Error_vals(:));
      min_Error = min(Error_vals(:));
      max_Error_thresh = 0.0 * max_Error;
      min_Error_thresh = 0.0 * min_Error;
      thresh_Error_vals = Error_vals;
      thresh_Error_vals(Error_vals > min_Error_thresh & Error_vals < max_Error_thresh) = 0;
      thresh_Error_vals(65:end-64,65:end-64) = 0;
      Recon_vals = Recon_vals + thresh_Error_vals;
    
      mean_Ganglion = mean(Ganglion_vals(:));
      std_Ganglion = std(Ganglion_vals(:));
      [unwhitened_Ganglion_DoG] =  deconvolvemirrorbc(Ganglion_vals', DoG_weights);
      unwhitened_Ganglion_fig(i_frame) = figure;
      set(unwhitened_Ganglion_fig(i_frame), "name", ["unwhitened Ganglion ", num2str(i_frame)]);
      imagesc(unwhitened_Ganglion_DoG); colormap(gray); box off; axis off; axis image;
      saveas(unwhitened_Ganglion_fig(i_frame), ...
	     [output_dir, filesep, "unwhitened_Ganglion_", num2str(i_frame, "%i")], "png");

      mean_Recon = mean(Recon_vals(:));
      std_Recon = std(Recon_vals(:));
      [unwhitened_Recon_vals] = deconvolvemirrorbc(Recon_vals', DoG_weights); 
      unwhitened_Recon_fig(i_frame) = figure;
      set(unwhitened_Recon_fig(i_frame), "name", ["unwhitened Recon ", num2str(i_frame)]);
      imagesc(unwhitened_Recon_vals); colormap(gray); box off; axis off; axis image;
      saveas(unwhitened_Recon_fig(i_frame), ...
	     [output_dir, filesep, "unwhitened_Recon_", num2str(i_frame, "%i")], "png");
    endif

  endfor
  drawnow;
endif


plot_ave_error_vs_time = 1;
if plot_ave_error_vs_time
  Error_Stats_file = [output_dir, filesep, "Error_Stats.txt"];
  Error_Stats_fid = fopen(Error_Stats_file, "r");
  Error_Stats_line = fgets(Error_Stats_fid);
  ave_error = [];
  %% skip startup artifact
  first_statProbe_step = first_checkpoint_ndx;
  for i_line = first_statProbe_step:begin_statProbe_step
    Error_Stats_line = fgets(Error_Stats_fid);
  endfor
  while (Error_Stats_line ~= -1)
    Error_Stats_ndx1 = strfind(Error_Stats_line, "sigma==");
    Error_Stats_ndx2 = strfind(Error_Stats_line, "nnz==");
    Error_Stats_str = Error_Stats_line(Error_Stats_ndx1+7:Error_Stats_ndx2-2);
    Error_Stats_val = str2num(Error_Stats_str);
    if isempty(ave_error)
      ave_error = Error_Stats_val;
    else
      ave_error = [ave_error; Error_Stats_val];
    endif
    Error_Stats_line = fgets(Error_Stats_fid);
    i_line = i_line + 1;
    if i_line > max_lines
      break;
    endif
  endwhile
  fclose(Error_Stats_fid);
  error_vs_time_fig = figure;
  error_vs_time_hndl = plot(ave_error);
  set(error_vs_time_fig, "name", ["ave Error"]);
  saveas(error_vs_time_fig, [output_dir, filesep, "error_vs_time_", num2str(last_checkpoint_ndx, "%i")], "png");
  drawnow;
%%endif
%%
%%plot_ave_V1_vs_time = 0;
%%if plot_ave_V1_vs_time
  V1_Stats_file = [output_dir, filesep, "V1_Stats.txt"];
  V1_Stats_fid = fopen(V1_Stats_file, "r");
  V1_Stats_line = fgets(V1_Stats_fid);
  ave_V1 = [];
  %% skip startup artifact
  for i_line = first_statProbe_step:begin_statProbe_step
    V1_Stats_line = fgets(V1_Stats_fid);
  endfor
    V1_Stats_ndx1 = strfind(V1_Stats_line, "N==");
    V1_Stats_ndx2 = strfind(V1_Stats_line, "Total==");
    V1_Stats_str = V1_Stats_line(V1_Stats_ndx1+3:V1_Stats_ndx2-2);
    num_V1 = str2num(V1_Stats_str);
  while (V1_Stats_line ~= -1)
    V1_Stats_ndx1 = strfind(V1_Stats_line, "nnz==");
    V1_Stats_ndx2 = length(V1_Stats_line); %% strfind(V1_Stats_line, "Max==");
    V1_Stats_str = V1_Stats_line(V1_Stats_ndx1+5:V1_Stats_ndx2-1);
    V1_Stats_val = str2num(V1_Stats_str);
    if isempty(ave_V1)
      ave_V1 = V1_Stats_val/num_V1;
    else
      ave_V1 = [ave_V1; V1_Stats_val/num_V1];
    endif
    V1_Stats_line = fgets(V1_Stats_fid);
    i_line = i_line + 1;
    if i_line > max_lines
      break;
    endif
  endwhile
  fclose(V1_Stats_fid);
  V1_vs_time_fig = figure;
  V1_vs_time_hndl = plot(ave_V1);
  set(V1_vs_time_fig, "name", ["sparseness_vs_time"]);
  saveas(V1_vs_time_fig, [output_dir, filesep, "sparseness_vs_time_", num2str(last_checkpoint_ndx, "%i")], "png");
  drawnow;
endif

plot_V1 = 1;
if plot_V1
  V1_path = [output_dir, filesep, "a5_V1.pvp"];
  write_step = frame_duration;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  [V1_struct, V1_hdr] = readpvpfile(V1_path, num_frames, []);
  nx_V1 = V1_hdr.nx;
  ny_V1 = V1_hdr.ny;
  nf_V1 = V1_hdr.nf;
  n_V1 = nx_V1 * ny_V1 * nf_V1;
  num_frames = size(V1_struct,1);
  i_frame = num_frames;
  start_frame = 1; %%
  V1_hist = zeros(nf_V1+1,1);
  V1_hist_edges = [0:1:nf_V1]+0.5;
  V1_current = zeros(n_V1,1);
  V1_abs_change = zeros(num_frames,1);
  V1_percent_change = zeros(num_frames,1);
  for i_frame = 1 : 1 : num_frames
    V1_time = squeeze(V1_struct{i_frame}.time);
    V1_active_ndx = squeeze(V1_struct{i_frame}.values);
    V1_previous = V1_current;
    V1_current = full(sparse(V1_active_ndx+1,1,1,n_V1,1,n_V1));
    V1_abs_change(i_frame) = sum(V1_current(:) ~= V1_previous(:));
    V1_tot_active = max(sum(V1_current(:)), sum(V1_previous(:)));
    V1_percent_change(i_frame) = ...
	V1_abs_change(i_frame) / (V1_tot_active + (V1_tot_active==0));
    V1_active_kf = mod(V1_active_ndx, nf_V1) + 1;
    if V1_tot_active > 0
      V1_hist_frame = histc(V1_active_kf, V1_hist_edges);
    else
      V1_hist_frame = zeros(nf_V1+1,1);
    endif
    V1_hist = V1_hist + V1_hist_frame;
  endfor %% i_frame
  V1_hist = V1_hist(1:nf_V1);
  V1_hist = V1_hist / (num_frames * nx_V1 * ny_V1); %% (sum(V1_hist(:)) + (nnz(V1_hist)==0));
  [V1_hist_sorted, V1_hist_rank] = sort(V1_hist, 1, "descend");
  V1_hist_title = ["V1_hist", ".png"];
  V1_hist_fig = figure;
  V1_hist_bins = 1:nf_V1;
  V1_hist_hndl = bar(V1_hist_bins, V1_hist_sorted); axis tight;
  set(V1_hist_fig, "name", ["V1_hist_", num2str(V1_time)]);
  saveas(V1_hist_fig, ...
	 [output_dir, filesep, ...
	  "V1_hist_", num2str(V1_time)], "png");

  V1_abs_change_title = ["V1_abs_change", ".png"];
  V1_abs_change_fig = figure;
  V1_abs_change_hndl = plot((1:num_frames)*write_step/1000, V1_abs_change); axis tight;
  set(V1_abs_change_fig, "name", ["V1_abs_change"]);
  saveas(V1_abs_change_fig, ...
	 [output_dir, filesep, "V1_abs_change", num2str(V1_time)], "png");

  V1_percent_change_title = ["V1_percent_change", ".png"];
  V1_percent_change_fig = figure;
  V1_percent_change_hndl = plot((1:num_frames)*write_step/1000, V1_percent_change); axis tight;
  set(V1_percent_change_fig, "name", ["V1_percent_change"]);
  saveas(V1_percent_change_fig, ...
	 [output_dir, filesep, "V1_percent_change", num2str(V1_time)], "png");
  drawnow;  
endif

plot_final_weights = 1;
if plot_final_weights
  V1ToError_path = [checkpoint_path, filesep, "V1ToError_W.pvp"];
  [V1ToError_struct, V1ToError_hdr] = readpvpfile(V1ToError_path,1);
  i_frame = 1;
  i_arbor = 1;
  V1ToError_weights = squeeze(V1ToError_struct{i_frame}.values{i_arbor});
  if ~exist("V1_hist_rank") || isempty(V1_hist_rank)
    V1_hist_rank = (1:V1ToError_hdr.nf);
  endif

  %% make tableau of all patches
  i_patch = 1;
  num_patches = size(V1ToError_weights, 3);
  num_patches_rows = floor(sqrt(num_patches));
  num_patches_cols = ceil(num_patches / num_patches_rows);
  V1ToError_fig = figure;
  set(V1ToError_fig, "name", ["V1ToError Weights: ", num2str(last_checkpoint_ndx)]);
  for j_patch = 1  : num_patches
    i_patch = V1_hist_rank(j_patch);
    subplot(num_patches_rows, num_patches_cols, j_patch); 
    patch_tmp = squeeze(V1ToError_weights(:,:,i_patch));
    patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
    min_patch = min(patch_tmp2(:));
    max_patch = max(patch_tmp2(:));
    patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch);
    patch_tmp2 = uint8(flipud(patch_tmp2'));
    imagesc(patch_tmp2); colormap(gray);
    box off
    axis off
    %%drawnow;
  endfor
  saveas(V1ToError_fig, [output_dir, filesep, "V1ToError_", num2str(last_checkpoint_ndx, "%i")], "png");


  %% make histogram of all weights
  V1ToError_hist_fig = figure;
  [V1ToError_hist, V1ToError_hist_bins] = hist(V1ToError_weights(:), 100);
  bar(V1ToError_hist_bins, log(V1ToError_hist+1));
  set(V1ToError_hist_fig, "name", ["V1ToError Histogram: ", num2str(last_checkpoint_ndx)]);
  saveas(V1ToError_hist_fig, [output_dir, filesep, "V1TpError_hist_", num2str(last_checkpoint_ndx, "%i")], "png");
endif

plot_weights_movie = 1;
if plot_weights_movie
  weights_movie_dir = [output_dir, filesep, "weights_movie"];
  mkdir(weights_movie_dir);
  V1ToError_path = [output_dir, filesep, "w4_V1ToError.pvp"];
  write_step = frame_duration;
  %%num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  num_frames = floor((last_checkpoint_ndx - 0) / write_step) - 1;
  start_frame = 1;
  [V1ToError_struct, V1ToError_hdr] = readpvpfile(V1ToError_path, num_frames, []);
  num_frames = size(V1ToError_struct,1);
  i_frame = num_frames;
  start_frame = 1; 
  if isempty(V1_hist_rank)
    V1_hist_rank = (1:V1ToError_hdr.nf);
  endif
  num_recon = max(floor(frame_duration / write_step) - 1, 0);
  i_arbor = 1;
  for i_frame = start_frame : 1 : num_frames
    if mod(i_frame, max(floor(num_frames/20),1)) == 0
      disp(["writing frame # ", num2str(i_frame)]);
    endif
    V1ToError_weights = squeeze(V1ToError_struct{i_frame}.values{i_arbor});
    i_patch = 1;
    [nyp, nxp, num_patches] = size(V1ToError_weights);
    num_patches_rows = floor(sqrt(num_patches));
    num_patches_cols = ceil(num_patches / num_patches_rows);
    weights_frame = uint8(zeros(num_patches_rows * nyp, num_patches_cols * nxp));
    for j_patch = 1  : num_patches
      i_patch = V1_hist_rank(j_patch);
      j_patch_row = ceil(j_patch / num_patches_cols);
      j_patch_col = 1 + mod(j_patch - 1, num_patches_cols);
      %%subplot(num_patches_rows, num_patches_cols, i_patch); 
      patch_tmp = squeeze(V1ToError_weights(:,:,i_patch));
      patch_tmp2 = flipud(patch_tmp'); %% imresize(patch_tmp, 12);
      min_patch = min(patch_tmp2(:));
      max_patch = max(patch_tmp2(:));
      patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch);
      patch_tmp2 = uint8(patch_tmp2);
      weights_frame(((j_patch_row - 1) * nyp + 1): (j_patch_row * nyp), ...
		    ((j_patch_col - 1) * nxp + 1): (j_patch_col * nxp)) = ...
	  patch_tmp2;
      %%imagesc(patch_tmp2);
      box off
      axis off
    endfor  %% i_patch
    frame_title = [num2str(i_frame, "%04d"), "_V1ToError", ".png"];
    imwrite(weights_frame, [weights_movie_dir, filesep, frame_title]);
  endfor %% i_frame
endif