
clear all;
close all;
setenv("GNUTERM","X11") 
addpath("/Users/garkenyon/workspace/PetaVision/mlab/util");
checkpoint_dir = "/Users/garkenyon/workspace/HyPerHLCA2/Checkpoints";
last_checkpoint_ndx = 80000;
first_checkpoint_ndx = 0;
write_step = 100;
output_dir = "/Users/garkenyon/workspace/HyPerHLCA2/output";
max_lines = last_checkpoint_ndx + (last_checkpoint_ndx == 0) * 1000;
startup_artifact_length = max(max_lines - 4000, 3);
num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
num_recon = 5; %% min(5, num_frames-1);
start_frame = floor(last_checkpoint_ndx / write_step) + num_recon + 2;

plot_Retina = 1;
if plot_Retina
  Retina_file = [output_dir, filesep, "a1_Retina.pvp"];
  [Retina_struct, Retina_hdr] = readpvpfile(Retina_file, 1, []);
  num_frames = size(Retina_struct,1);
  i_frame = num_frames;
  for i_frame = start_frame : -1 : start_frame - num_recon
    Retina_time = Retina_struct{i_frame}.time;
    Retina_vals = Retina_struct{i_frame}.values;
    Retina_fig(i_frame) = figure;
    set(Retina_fig(i_frame), "name", ["Retina ", num2str(i_frame)]);
    imagesc(Retina_vals'); colormap(gray); box off; axis off; axis image;
  endfor
endif

plot_Ganglion = 1;
if plot_Ganglion
  Ganglion_file = [output_dir, filesep, "a2_Ganglion.pvp"];
  [Ganglion_struct, Ganglion_hdr] = readpvpfile(Ganglion_file, 1, []);
  num_frames = size(Ganglion_struct,1);
  i_frame = num_frames;
  for i_frame = start_frame : -1 : start_frame - num_recon
    Ganglion_time = Ganglion_struct{i_frame}.time;
    Ganglion_vals = Ganglion_struct{i_frame}.values;
    Ganglion_fig(i_frame) = figure;
    set(Ganglion_fig(i_frame), "name", ["Ganglion ", num2str(i_frame)]);
    imagesc(Ganglion_vals'); colormap(gray); box off; axis off; axis image;
  endfor
endif

plot_Recon = 1;
if plot_Recon
  Recon_file = [output_dir, filesep, "a5_Recon.pvp"];
  [Recon_struct, Recon_hdr] = readpvpfile(Recon_file, 1, []);
  num_frames = size(Recon_struct,1);
  i_frame = num_frames;
  for i_frame = start_frame : -1 : start_frame - num_recon
    Recon_time = Recon_struct{i_frame}.time;
    Recon_vals = Recon_struct{i_frame}.values;
    Recon_fig(i_frame) = figure;
    set(Recon_fig(i_frame), "name", ["Recon ", num2str(i_frame)]);
    imagesc(Recon_vals'); colormap(gray); box off; axis off; axis image;
  endfor
endif


plot_ave_error_vs_time = 1;
if plot_ave_error_vs_time
  Error_Stats_file = [output_dir, filesep, "Error_Stats.txt"];
  Error_Stats_fid = fopen(Error_Stats_file, "r");
  Error_Stats_line = fgets(Error_Stats_fid);
  ave_error = [];
  %% skip startup artifact
  for i_line = 1:startup_artifact_length
    Error_Stats_line = fgets(Error_Stats_fid);
  endfor
  while (Error_Stats_line ~= -1)
    Error_Stats_ndx1 = strfind(Error_Stats_line, "Avg==");
    Error_Stats_ndx2 = strfind(Error_Stats_line, "Max==");
    Error_Stats_str = Error_Stats_line(Error_Stats_ndx1+5:Error_Stats_ndx2-2);
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
endif

plot_ave_V1_vs_time = 1;
if plot_ave_V1_vs_time
  V1_Stats_file = [output_dir, filesep, "V1_Stats.txt"];
  V1_Stats_fid = fopen(V1_Stats_file, "r");
  V1_Stats_line = fgets(V1_Stats_fid);
  ave_V1 = [];
  %% skip startup artifact
  for i_line = 1:startup_artifact_length
    V1_Stats_line = fgets(V1_Stats_fid);
  endfor
  while (V1_Stats_line ~= -1)
    V1_Stats_ndx1 = strfind(V1_Stats_line, "Avg==");
    V1_Stats_ndx2 = strfind(V1_Stats_line, "Max==");
    V1_Stats_str = V1_Stats_line(V1_Stats_ndx1+5:V1_Stats_ndx2-2);
    V1_Stats_val = str2num(V1_Stats_str);
    if isempty(ave_V1)
      ave_V1 = V1_Stats_val;
    else
      ave_V1 = [ave_V1; V1_Stats_val];
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
  set(V1_vs_time_fig, "name", ["ave V1"]);
endif

plot_final_weights = 1;
if plot_final_weights
  checkpoint_path = [checkpoint_dir, filesep, "Checkpoint", num2str(last_checkpoint_ndx)];
  V1ToRecon_path = [checkpoint_path, filesep, "V1ToRecon_W.pvp"];
  [V1ToRecon_struct, V1ToRecon_hdr] = readpvpfile(V1ToRecon_path,1);
  i_frame = 1;
  i_arbor = 1;
  V1ToRecon_weights = squeeze(V1ToRecon_struct{i_frame}.values{i_arbor});

  %% make tableau of all patches
  i_patch = 1;
  num_patches = size(V1ToRecon_weights, 3);
  num_patches_rows = floor(sqrt(num_patches));
  num_patches_cols = ceil(num_patches / num_patches_rows);
  V1ToRecon_fig = figure;
  set(V1ToRecon_fig, "name", ["V1ToRecon Weights: ", num2str(last_checkpoint_ndx)]);
  for i_patch = 1  : num_patches
    i_patch_col = ceil((i_patch-1) / num_patches_rows);
    i_patch_row = 1 + mod((i_patch-1), num_patches_rows);
    subplot(num_patches_rows, num_patches_cols, i_patch); 
    patch_tmp = squeeze(V1ToRecon_weights(:,:,i_patch));
    patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
    min_patch = min(patch_tmp2(:));
    max_patch = max(patch_tmp2(:));
    patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch);
    patch_tmp2 = uint8(patch_tmp2);
    imagesc(patch_tmp2);
    box off
    axis off
    %%drawnow;
  endfor

  %% make histogram of all weights
  V1ToRecon_hist_fig = figure;
  [V1ToRecon_hist, V1ToRecon_hist_bins] = hist(V1ToRecon_weights(:), 100);
  bar(V1ToRecon_hist_bins, log(V1ToRecon_hist+1));
  set(V1ToRecon_hist_fig, "name", ["V1ToRecon Histogram: ", num2str(last_checkpoint_ndx)]);
endif

plot_weight_movie = 0;
if plot_weight_movie
  V1ToRecon_path = [checkpoint_path, filesep, "V1ToRecon_W.pvp"];
  [V1ToRecon_struct, V1ToRecon_hdr] = readpvpfile(V1ToRecon_path,1);
  i_frame = 1;
  i_arbor = 1;
  V1ToRecon_weights = squeeze(V1ToRecon_struct{i_frame}.values{i_arbor});
  i_patch = 1;
  num_patches = size(V1ToRecon_vals, 3);
  num_patches_rows = floor(sqrt(num_patches));
  num_patches_cols = ceil(num_patches / num_patches_rows);
  for i_patch = 1  : num_patches
    i_patch_col = ceil((i_patch-1) / num_patches_rows);
    i_patch_row = 1 + mod((i_patch-1), num_patches_rows);
    subplot(num_patches_rows, num_patches_cols, i_patch); 
    patch_tmp = squeeze(V1ToRecon_vals(:,:,i_patch));
    patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
    min_patch = min(patch_tmp2(:));
    max_patch = max(patch_tmp2(:));
    patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch);
    patch_tmp2 = uint8(patch_tmp2);
    imagesc(patch_tmp2);
    box off
    axis off
  endfor
endif