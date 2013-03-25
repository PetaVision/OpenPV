
clear all;
close all;
setenv("GNUTERM","X11")
if ismac
  workspace_path = "/Users/garkenyon/workspace";
  output_dir = "/Users/garkenyon/workspace/HyPerHLCA2/output"
  LCA_path = [workspace_path, filesep, "HyPerHLCA2"];
  last_checkpoint_ndx = 20000; 
  next_checkpoint_ndx = 40000;
  first_checkpoint_ndx = 0; 
  frame_duration = 1000;
elseif isunix
  workspace_path = "/home/gkenyon/workspace_new";
  output_dir = "/nh/compneuro/Data/vine/LCA/cats"; 
  LCA_path = [output_dir];
  last_checkpoint_ndx = 20706*40; 
  next_checkpoint_ndx = 20706*41; 
  first_checkpoint_ndx = 1; %% 20706*0;
  frame_duration = 500;
endif
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
checkpoint_dir = [LCA_path, filesep, "Checkpoints"];
checkpoint_path = [checkpoint_dir, filesep, "Checkpoint", num2str(last_checkpoint_ndx, "%i")];
next_checkpoint_path = [checkpoint_dir, filesep, "Checkpoint", num2str(next_checkpoint_ndx, "%i")];
max_lines = last_checkpoint_ndx + (last_checkpoint_ndx == 0) * 1000;
max_history = 50000;
begin_statProbe_step = max(max_lines - max_history, 3);
frame_duration = 1000;


%% plot Reconstructions
plot_Recon = 1;
if plot_Recon
  %%keyboard;
  num_recon = 3;
  recon_dir = [output_dir, filesep, "recon"];
  mkdir(recon_dir);
  Retina_file = [output_dir, filesep, "a1_Retina.pvp"];
  Ganglion_file = [output_dir, filesep, "a3_Ganglion.pvp"];
  Recon_file = [output_dir, filesep, "a4_Recon.pvp"];
  Error_file = [output_dir, filesep, "a5_Error.pvp"];
  write_step = frame_duration;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  start_frame = num_frames-num_recon+1; %% floor((first_checkpoint_ndx) / write_step);
  [Retina_struct, Retina_hdr] = readpvpfile(Retina_file, num_frames, [], start_frame);
  num_Retina_frames = size(Retina_struct,1);
  [Ganglion_struct, Ganglion_hdr] = ...
      readpvpfile(Ganglion_file, num_frames, start_frame + num_Retina_frames - 1, start_frame);
  [Recon_struct, Recon_hdr] = ...
      readpvpfile(Recon_file, num_frames, start_frame + num_Retina_frames - 1, start_frame);
  %%[Error_struct, Error_hdr] = readpvpfile(Error_file, num_frames, num_frames, start_frame);
  num_Ganglion_frames = size(Ganglion_struct,1);
  num_Recon_frames = size(Recon_struct,1);

  plot_DoG_kernel = 1;
  if plot_DoG_kernel
    i_frame = 1;
    i_arbor = 1;
    i_patch = 1;
    DoG_center_path = [checkpoint_path, filesep, "BipolarToGanglionCenter_W.pvp"];
    [DoG_center_struct, DoG_center_hdr] = readpvpfile(DoG_center_path,1);
    DoG_center_weights = squeeze(DoG_center_struct{i_frame}.values{i_arbor});
    DoG_surround_path = [checkpoint_path, filesep, "BipolarToGanglionSurround_W.pvp"];
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
    saveas(DoG_fig, [recon_dir, filesep, "DoG_weights.png"]);
  endif

  Retina_fig = figure;
  for i_frame = num_Retina_frames - num_recon + 1: 1 : num_Retina_frames
    Retina_time = Retina_struct{i_frame}.time;
    Retina_vals = Retina_struct{i_frame}.values;
    set(Retina_fig, "name", ["Retina ", num2str(Retina_time)]);
    imagesc(Retina_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Retina_fig, [recon_dir, filesep, "Retina_", num2str(Retina_time)], "png");
  endfor   %% i_frame

  Ganglion_fig = figure;
    if plot_DoG_kernel
      unwhitened_Ganglion_fig = figure;
    endif
    for i_frame = num_Ganglion_frames - num_recon + 1 : 1 : num_Ganglion_frames
    Ganglion_time = Ganglion_struct{i_frame}.time;
    Ganglion_vals = Ganglion_struct{i_frame}.values;
    figure(Ganglion_fig);
    set(Ganglion_fig, "name", ["Ganglion ", num2str(Ganglion_time)]);
    imagesc(Ganglion_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Ganglion_fig, [recon_dir, filesep, "Ganglion_", num2str(Ganglion_time)], "png");
    drawnow

    %%Error_time = Error_struct{i_frame}.time;
    %%Error_vals = Error_struct{i_frame}.values;
    %%Error_fig(i_frame) = figure;
    %%set(Error_fig(i_frame), "name", ["Error ", num2str(i_frame, "%i")]);
    %%imagesc(Error_vals'); colormap(gray); box off; axis off; axis image;
    %%saveas(Error_fig(i_frame), [recon_dir, filesep, "Error_", num2str(i_frame, "%i")], "png");

    if plot_DoG_kernel
      mean_Ganglion = mean(Ganglion_vals(:));
      std_Ganglion = std(Ganglion_vals(:));
      [unwhitened_Ganglion_DoG] =  deconvolvemirrorbc(Ganglion_vals', DoG_weights);
      figure(unwhitened_Ganglion_fig);
      set(unwhitened_Ganglion_fig, "name", ["unwhitened Ganglion ", num2str(Ganglion_time)]);
      imagesc(unwhitened_Ganglion_DoG); colormap(gray); box off; axis off; axis image;
      saveas(unwhitened_Ganglion_fig, ...
	     [recon_dir, filesep, "unwhitened_Ganglion_", num2str(Ganglion_time)], "png");
    endif %% plot_DoG_kernel

  endfor   %% i_frame

  recon_start_frame = num_Recon_frames - num_recon + 1; %%
  Recon_fig = figure;
  if plot_DoG_kernel
      unwhitened_Recon_fig = figure;
  endif
  for i_frame = recon_start_frame : 1 : num_Recon_frames
    Recon_time = Recon_struct{i_frame}.time;
    Recon_vals = Recon_struct{i_frame}.values;
    figure(Recon_fig);
    set(Recon_fig, "name", ["Recon ", num2str(Recon_time)]);
    imagesc(Recon_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Recon_fig, [recon_dir, filesep, "Recon_", num2str(Recon_time)], "png");
    if plot_DoG_kernel
      mean_Recon = mean(Recon_vals(:));
      std_Recon = std(Recon_vals(:));
      [unwhitened_Recon_vals] = deconvolvemirrorbc(Recon_vals', DoG_weights); 
      figure(unwhitened_Recon_fig);
      set(unwhitened_Recon_fig, "name", ["unwhitened Recon ", num2str(Recon_time)]);
      imagesc(unwhitened_Recon_vals); colormap(gray); box off; axis off; axis image;
      saveas(unwhitened_Recon_fig, ...
	     [recon_dir, filesep, "unwhitened_Recon_", num2str(Recon_time)], "png");
    endif %% plot_DoG_kernel

  endfor
  drawnow;
endif


plot_ave_error_vs_time = 1;
if plot_ave_error_vs_time
  sparseness_error_vs_time_dir = [output_dir, filesep, "sparseness_error_vs_time"];
  mkdir(sparseness_error_vs_time_dir);
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
  saveas(error_vs_time_fig, [sparseness_error_vs_time_dir, filesep, "error_vs_time_", num2str(last_checkpoint_ndx, "%i")], "png");
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
  saveas(V1_vs_time_fig, [sparseness_error_vs_time_dir, filesep, "sparseness_vs_time_", num2str(last_checkpoint_ndx, "%i")], "png");
  drawnow;
endif

plot_V1 = 1;
if plot_V1
  V1_path = [output_dir, filesep, "a6_V1.pvp"];
  write_step = frame_duration;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  [V1_struct, V1_hdr] = readpvpfile(V1_path, num_frames, num_frames, 1);
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
  V1_current_active = 0;
  V1_tot_active = zeros(num_frames,1);
  V1_times = zeros(num_frames,1);
  for i_frame = 1 : 1 : num_frames
    V1_times(i_frame) = squeeze(V1_struct{i_frame}.time);
    V1_active_ndx = squeeze(V1_struct{i_frame}.values);
    V1_previous = V1_current;
    V1_current = full(sparse(V1_active_ndx+1,1,1,n_V1,1,n_V1));
    V1_abs_change(i_frame) = sum(V1_current(:) ~= V1_previous(:));
    V1_previous_active = V1_current_active;
    V1_current_active = sum(V1_current(:));
    V1_tot_active(i_frame) = V1_current_active;
    V1_max_active = max(V1_current_active, V1_previous_active);
    V1_percent_change(i_frame) = ...
	V1_abs_change(i_frame) / (V1_max_active + (V1_max_active==0));
    V1_active_kf = mod(V1_active_ndx, nf_V1) + 1;
    if V1_max_active > 0
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
  set(V1_hist_fig, "name", ["V1_hist_", num2str(V1_times(num_frames), "%i")]);
  V1_rank_dir = [output_dir, filesep, "V1_rank"];
  mkdir(V1_rank_dir);
  saveas(V1_hist_fig, ...
	 [V1_rank_dir, filesep, ...
	  "V1_rank_", num2str(V1_times(num_frames), "%i")], "png");

  V1_abs_change_title = ["V1_abs_change", ".png"];
  V1_abs_change_fig = figure;
  V1_abs_change_hndl = plot(V1_times, V1_abs_change); axis tight;
  set(V1_abs_change_fig, "name", ["V1_abs_change"]);
  V1_change_dir = [output_dir, filesep, "V1_rank"];
  mkdir(V1_change_dir);
  saveas(V1_abs_change_fig, ...
	 [V1_change_dir, filesep, "V1_abs_change", num2str(V1_times(num_frames), "%i")], "png");

  V1_percent_change_title = ["V1_percent_change", ".png"];
  V1_percent_change_fig = figure;
  V1_percent_change_hndl = plot(V1_times, V1_percent_change); axis tight;
  set(V1_percent_change_fig, "name", ["V1_percent_change"]);
  saveas(V1_percent_change_fig, ...
	 [V1_change_dir, filesep, "V1_percent_change", num2str(V1_times(num_frames), "%i")], "png");

  V1_tot_active_title = ["V1_tot_active", ".png"];
  V1_tot_active_fig = figure;
  V1_tot_active_hndl = plot(V1_times, V1_tot_active/n_V1); axis tight;
  set(V1_tot_active_fig, "name", ["V1_tot_active"]);
  saveas(V1_tot_active_fig, ...
	 [V1_change_dir, filesep, "V1_tot_active", num2str(V1_times(num_frames), "%i")], "png");

  V1_mean_active = mean(V1_tot_active(:)/n_V1);
  disp(["V1_mean_active = ", num2str(V1_mean_active)]);

  Error_path = [output_dir, filesep, "a5_Error.pvp"];
  write_step = frame_duration;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  [Error_struct, Error_hdr] = readpvpfile(Error_path, num_frames, num_frames, 1);
  nx_Error = Error_hdr.nx;
  ny_Error = Error_hdr.ny;
  nf_Error = Error_hdr.nf;
  n_Error = nx_Error * ny_Error * nf_Error;
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
	 [V1_change_dir, filesep, "Error_RMS", num2str(Error_times(num_frames), "%i")], "png");

  Error_mean_RMS = mean(Error_RMS(:));
  disp(["Error_mean_RMS = ", num2str(Error_mean_RMS)]);

  drawnow;  
endif

plot_final_weights = 1;
if plot_final_weights 
  V1ToError_path = [checkpoint_path, filesep, "V1ToError_W.pvp"];
  if ~exist(V1ToError_path, "file")
    V1ToError_path = [next_checkpoint_path, filesep, "V1ToError_W.pvp"];
  endif
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
  set(V1ToError_fig, "name", ["V1ToError Weights: ", num2str(last_checkpoint_ndx, "%i")]);
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
    axis image
    %%drawnow;
  endfor
  V1ToError_weights_dir = [output_dir, filesep, "V1ToError_weights"];
  mkdir(V1ToError_weights_dir);
  saveas(V1ToError_fig, [V1ToError_weights_dir, filesep, "V1ToError_", num2str(last_checkpoint_ndx, "%i")], "png");


  %% make histogram of all weights
  V1ToError_hist_fig = figure;
  [V1ToError_hist, V1ToError_hist_bins] = hist(V1ToError_weights(:), 100);
  bar(V1ToError_hist_bins, log(V1ToError_hist+1));
  set(V1ToError_hist_fig, "name", ["V1ToError Histogram: ", num2str(last_checkpoint_ndx, "%i")]);
  V1ToError_weights_hist_dir = [output_dir, filesep, "V1ToError_hist"];
  mkdir(V1ToError_weights_hist_dir);
  saveas(V1ToError_hist_fig, [V1ToError_weights_hist_dir, filesep, "V1ToError_hist_", num2str(last_checkpoint_ndx, "%i")], "png");

  mat_dir = [output_dir, filesep, "mat"];
  mkdir(mat_dir);
  V1ToError_weights_file = [mat_dir, filesep, "V1ToError_weights", num2str(last_checkpoint_ndx, "%i"), ".mat"];
  save(  "-mat", V1ToError_weights_file, "V1ToError_weights");
endif

plot_weights_movie = 0;
if plot_weights_movie
  weights_movie_dir = [output_dir, filesep, "V1ToError_movie"];
  mkdir(weights_movie_dir);
  V1ToError_path = [output_dir, filesep, "w5_V1ToError.pvp"];
  write_step = frame_duration;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  start_frame = 1;
  [V1ToError_struct, V1ToError_hdr] = readpvpfile(V1ToError_path, num_frames);
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
      disp(["writing frame # ", num2str(i_frame, "%i")]);
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