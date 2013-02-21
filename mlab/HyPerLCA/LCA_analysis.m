
clear all;
close all;
setenv("GNUTERM","X11") 
addpath("/Users/garkenyon/workspace/PetaVision/mlab/util");
checkpoint_dir = "/Users/garkenyon/workspace/HyPerHLCA2/Checkpoints";
last_checkpoint_ndx = 60000;
first_checkpoint_ndx = 0;
output_dir = "/Users/garkenyon/workspace/HyPerHLCA2/output";
max_lines = last_checkpoint_ndx + (last_checkpoint_ndx == 0) * 1000;
startup_artifact_length = max(max_lines - 4000, 3);
frame_duration = 500;

%% get DoG kernel
plot_DoG_kernel = 1;
if plot_DoG_kernel
  checkpoint_path = [checkpoint_dir, filesep, "Checkpoint", num2str(last_checkpoint_ndx)];
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
  saveas(DoG_fig, "DoG_weights.png");
endif


plot_Retina = 1;
if plot_Retina
  Retina_file = [output_dir, filesep, "a1_Retina.pvp"];
  write_step = 500;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  [Retina_struct, Retina_hdr] = readpvpfile(Retina_file, num_frames, []);
  num_frames = size(Retina_struct,1);
  i_frame = num_frames;
  start_frame = floor(last_checkpoint_ndx / write_step);
  for i_frame = start_frame : 1 : start_frame
    Retina_time = Retina_struct{i_frame}.time;
    Retina_vals = Retina_struct{i_frame}.values;
    Retina_fig(i_frame) = figure;
    set(Retina_fig(i_frame), "name", ["Retina ", num2str(i_frame)]);
    imagesc(Retina_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Retina_fig(i_frame), ["Retina_", num2str(i_frame)], "png");
  endfor
  drawnow;
endif

plot_Ganglion = 1;
if plot_Ganglion
  Ganglion_file = [output_dir, filesep, "a2_Ganglion.pvp"];
  write_step = 500;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  [Ganglion_struct, Ganglion_hdr] = readpvpfile(Ganglion_file, num_frames, []);
  num_frames = size(Ganglion_struct,1);
  i_frame = num_frames;
  start_frame = floor(last_checkpoint_ndx / write_step);
  for i_frame = start_frame : 1 : start_frame
    Ganglion_time = Ganglion_struct{i_frame}.time;
    Ganglion_vals = Ganglion_struct{i_frame}.values;
    Ganglion_fig(i_frame) = figure;
    set(Ganglion_fig(i_frame), "name", ["Ganglion ", num2str(i_frame)]);
    imagesc(Ganglion_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Ganglion_fig(i_frame), ["Ganglion_", num2str(i_frame)], "png");

    %% apply inverse DoG kernel
    if plot_DoG_kernel
      min_Ganglion_val = min(Ganglion_vals(:));
      max_Ganglion_val = max(Ganglion_vals(:));
      Ganglion_DoG = (Ganglion_vals' - min_Ganglion_val) / (max_Ganglion_val - min_Ganglion_val);
      %%Ganglion_DoG([1 2 3  end-2 end-1 end],:) = 0;
      %%Ganglion_DoG(:, [1 2 3 end-2 end-1 end]) = 0;
      [inv_Ganglion_DoG] =  invertdog(Ganglion_vals', DoG_weights); %%invDoG(Ganglion_DoG, DoG_weights);
      inv_Ganglion_fig(i_frame) = figure;
      set(inv_Ganglion_fig(i_frame), "name", ["inverse Ganglion ", num2str(i_frame)]);
      imagesc(inv_Ganglion_DoG); colormap(gray); box off; axis off; axis image;
      saveas(inv_Ganglion_fig(i_frame), ["inverse_Ganglion_", num2str(i_frame)], "png");
    endif
    
  endfor
  drawnow;
endif

plot_Recon = 1;
if plot_Recon
  Recon_file = [output_dir, filesep, "a3_Recon.pvp"];
  write_step = 500;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  [Recon_struct, Recon_hdr] = readpvpfile(Recon_file, num_frames, []);
  num_frames = size(Recon_struct,1);
  i_frame = num_frames;
  start_frame = floor(last_checkpoint_ndx / write_step);
  num_recon = max(floor(frame_duration / write_step) - 1, 0);
  for i_frame = start_frame - num_recon : 1 : start_frame
    Recon_time = Recon_struct{i_frame}.time;
    Recon_vals = Recon_struct{i_frame}.values;
    Recon_fig(i_frame) = figure;
    set(Recon_fig(i_frame), "name", ["Recon ", num2str(i_frame)]);
    imagesc(Recon_vals'); colormap(gray); box off; axis off; axis image;
    saveas(Recon_fig(i_frame), ["Recon_", num2str(i_frame)], "png");

    %% apply inverse DoG kernel
    if plot_DoG_kernel
      [inverse_Recon_vals] = invertdog(Recon_vals', DoG_weights); %%invertDoGImage(Recon_vals', DoG_weights);
      inverse_Recon_fig(i_frame) = figure;
      set(inverse_Recon_fig(i_frame), "name", ["inverse Recon ", num2str(i_frame)]);
      imagesc(inverse_Recon_vals); colormap(gray); box off; axis off; axis image;
      saveas(inverse_Recon_fig(i_frame), ["inverse_Recon_", num2str(i_frame)], "png");
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
  for i_line = 1:startup_artifact_length
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
  saveas(error_vs_time_fig, ["error_vs_time_", num2str(last_checkpoint_ndx)], "png");
  drawnow;
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
  saveas(V1_vs_time_fig, ["sparseness_vs_time_", num2str(last_checkpoint_ndx)], "png");
  drawnow;
endif

plot_final_weights = 0;
if plot_final_weights
  checkpoint_path = [checkpoint_dir, filesep, "Checkpoint", num2str(last_checkpoint_ndx)];
  V1ToError_path = [checkpoint_path, filesep, "V1ToError_W.pvp"];
  [V1ToError_struct, V1ToError_hdr] = readpvpfile(V1ToError_path,1);
  i_frame = 1;
  i_arbor = 1;
  V1ToError_weights = squeeze(V1ToError_struct{i_frame}.values{i_arbor});

  %% make tableau of all patches
  i_patch = 1;
  num_patches = size(V1ToError_weights, 3);
  num_patches_rows = floor(sqrt(num_patches));
  num_patches_cols = ceil(num_patches / num_patches_rows);
  V1ToError_fig = figure;
  set(V1ToError_fig, "name", ["V1ToError Weights: ", num2str(last_checkpoint_ndx)]);
  for i_patch = 1  : num_patches
    subplot(num_patches_rows, num_patches_cols, i_patch); 
    patch_tmp = squeeze(V1ToError_weights(:,:,i_patch));
    patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
    min_patch = min(patch_tmp2(:));
    max_patch = max(patch_tmp2(:));
    patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch);
    patch_tmp2 = uint8(patch_tmp2);
    imagesc(patch_tmp2); colormap(gray);
    box off
    axis off
    %%drawnow;
  endfor
  saveas(V1ToError_fig, ["V1TpError_", num2str(last_checkpoint_ndx)], "png");


  %% make histogram of all weights
  V1ToError_hist_fig = figure;
  [V1ToError_hist, V1ToError_hist_bins] = hist(V1ToError_weights(:), 100);
  bar(V1ToError_hist_bins, log(V1ToError_hist+1));
  set(V1ToError_hist_fig, "name", ["V1ToError Histogram: ", num2str(last_checkpoint_ndx)]);
  saveas(V1ToError_hist_fig, ["V1TpError_hist_", num2str(last_checkpoint_ndx)], "png");
endif

plot_weights_movie = 0;
if plot_weights_movie
  weights_movie_dir = [output_dir, filesep, "weights_movie"];
  mkdir(weights_movie_dir);
  V1ToError_path = [output_dir, filesep, "w4_V1ToError.pvp"];
  write_step = 500;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  [V1ToError_struct, V1ToError_hdr] = readpvpfile(V1ToError_path, num_frames, []);
  num_frames = size(V1ToError_struct,1);
  i_frame = num_frames;
  start_frame = floor(last_checkpoint_ndx / write_step);
  num_recon = max(floor(frame_duration / write_step) - 1, 0);
  i_arbor = 1;
  for i_frame = 1 : 1 : start_frame
    V1ToError_weights = squeeze(V1ToError_struct{i_frame}.values{i_arbor});
    i_patch = 1;
    [nyp, nxp, num_patches] = size(V1ToError_weights);
    num_patches_rows = floor(sqrt(num_patches));
    num_patches_cols = ceil(num_patches / num_patches_rows);
    weights_frame = uint8(zeros(num_patches_rows * nyp, num_patches_cols * nxp));
    for i_patch = 1  : num_patches
      i_patch_row = ceil(i_patch / num_patches_cols);
      i_patch_col = 1 + mod(i_patch - 1, num_patches_cols);
      %%subplot(num_patches_rows, num_patches_cols, i_patch); 
      patch_tmp = squeeze(V1ToError_weights(:,:,i_patch));
      patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
      min_patch = min(patch_tmp2(:));
      max_patch = max(patch_tmp2(:));
      patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch);
      patch_tmp2 = uint8(patch_tmp2);
      weights_frame(((i_patch_row - 1) * nyp + 1): (i_patch_row * nyp), ...
		    ((i_patch_col - 1) * nxp + 1): (i_patch_col * nxp)) = ...
	  patch_tmp2;
      %%imagesc(patch_tmp2);
      box off
      axis off
    endfor  %% i_patch
    frame_title = [num2str(i_frame, "%03d"), "_V1ToError", ".png"];
    imwrite(weights_frame, [weights_movie_dir, filesep, frame_title]);
  endfor %% i_frame
endif