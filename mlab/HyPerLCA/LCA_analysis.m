  
clear all;
close all;
setenv("GNUTERM","X11")
if ismac
  workspace_path = "/Users/garkenyon/workspace";
  output_dir = "/Users/garkenyon/workspace/HyPerHLCA2/output_animal1200000_color_deep"; %%output_test"; %% output_animal1200000_distractor1200000"; %%
  LCA_path = [output_dir]; %%[workspace_path, filesep, "HyPerHLCA2"];
  last_checkpoint_ndx = 10000*70; 
  next_checkpoint_ndx = 10000*71;
  first_checkpoint_ndx = 0; 
  frame_duration = 1000;
elseif isunix
  workspace_path = "/home/gkenyon/workspace";
  output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_16x16x1024_Overlap_lambda_05X2"; %%MRI/LCA/5_subjects"; %%  
  %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_12x12x1024_lambda_05X2_color_deep"; %%MRI/LCA/5_subjects"; %%  
  LCA_path = [output_dir];
  last_checkpoint_ndx = 42600000; %% 
  next_checkpoint_ndx = 42700000; %%50000*22; %% 
  first_checkpoint_ndx = 0;
  frame_duration = 5000;
endif
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
checkpoint_dir = [LCA_path, filesep, "Checkpoints"];
checkpoint_path = [checkpoint_dir, filesep, "Checkpoint", num2str(last_checkpoint_ndx, "%i")];
next_checkpoint_path = [checkpoint_dir, filesep, "Checkpoint", num2str(next_checkpoint_ndx, "%i")];
max_lines = last_checkpoint_ndx + (last_checkpoint_ndx == 0) * 50000;
max_history = 196000;
begin_statProbe_step = max(max_lines - max_history, 3);
training_flag = 1;
num_recon = 196;
deep_flag = 0;
deep2_flag = 0;


%% plot Reconstructions
plot_Recon = 1;
if plot_Recon
  %%keyboard;
  recon_dir = [output_dir, filesep, "recon"];
  mkdir(recon_dir);
  if deep_flag
    recon2_dir = [output_dir, filesep, "recon2"];
    mkdir(recon2_dir);
    if deep2_flag
      reconInfra_dir = [output_dir, filesep, "reconInfra"];
      mkdir(reconInfra_dir);
    endif
  endif

  %% parse center/surround pre-processing filters
  plot_DoG_kernel = 1;
  if plot_DoG_kernel
    i_frame = 1;
    i_arbor = 1;
    i_patch = 1;
    blur_center_path = [checkpoint_path, filesep, "RetinaToBipolarCenter_W.pvp"];
    [blur_center_struct, blur_center_hdr] = readpvpfile(blur_center_path,1);
    blur_weights = (blur_center_struct{i_frame}.values{i_arbor});
    blur_fig = figure;
    set(blur_fig, "name", "blur Weights");
    size_blur_weights = size(blur_weights);
    num_dims_blur = length(size_blur_weights);
    if num_dims_blur > 2
      num_pre_colors = size_blur_weights(3);
    else
      num_pre_colors = 1;
    endif
    if num_dims_blur > 3
      num_post_colors = size_blur_weights(4);
    else
      num_post_colors = 1;
    endif
    for i_pre_color = 1 : num_pre_colors
      for i_post_color = 1 : num_post_colors
	subplot(num_pre_colors, num_post_colors, (i_pre_color - 1) * num_post_colors + i_post_color);
	patch_tmp = blur_weights(:,:,i_pre_color, i_post_color);
	patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
	min_patch = min(patch_tmp2(:));
	max_patch = max(patch_tmp2(:));
	patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch-min_patch)==0));
	patch_tmp2 = uint8(patch_tmp2);
	imagesc(patch_tmp2); colormap(gray);
	box off
	axis off
      endfor
    endfor
    saveas(blur_fig, [recon_dir, filesep, "blur_weights.png"]);
    drawnow;
    DoG_center_path = [checkpoint_path, filesep, "BipolarToGanglionCenter_W.pvp"];
    [DoG_center_struct, DoG_center_hdr] = readpvpfile(DoG_center_path,1);
    DoG_center_weights = (DoG_center_struct{i_frame}.values{i_arbor});
    size_DoG_center_weights = size(DoG_center_weights);
    DoG_surround_path = [checkpoint_path, filesep, "BipolarToGanglionSurround_W.pvp"];
    [DoG_surround_struct, DoG_surround_hdr] = readpvpfile(DoG_surround_path,1);
    DoG_surround_weights = (DoG_surround_struct{i_frame}.values{i_arbor});
    size_DoG_weights = size(DoG_surround_weights);
    DoG_pad = (size_DoG_weights(1:2) - size_DoG_center_weights(1:2)) / 2;
    num_dims_DoG = length(size_DoG_weights);
    if num_dims_DoG > 2
      num_pre_colors = size_DoG_weights(3);
    else
      num_pre_colors = 1;
    endif
    if num_dims_DoG > 3
      num_post_colors = size_DoG_weights(4);
    else
      num_post_colors = 1;
    endif
    DoG_center_padded = zeros(size_DoG_weights(1:2));
    DoG_row_start = DoG_pad(1)+1;
    DoG_row_stop = size_DoG_weights(1)-DoG_pad(1);
    DoG_col_start = DoG_pad(2)+1;
    DoG_col_stop = size_DoG_weights(2)-DoG_pad(2);
    DoG_fig = figure;
    set(DoG_fig, "name", "DoG Weights");
    for i_pre_color = 1 : num_pre_colors
      for i_post_color = 1 : num_post_colors
	DoG_center_padded(DoG_row_start:DoG_row_stop, DoG_col_start:DoG_col_stop) = ...
	    DoG_center_weights(:,:, i_pre_color, i_post_color);
	DoG_weights = ...
	    DoG_center_padded - DoG_surround_weights(:, :, i_pre_color, i_post_color);
	subplot(num_pre_colors, num_post_colors, (i_pre_color - 1) * num_post_colors + i_post_color);
	patch_tmp = DoG_weights;
	patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
	min_patch = min(patch_tmp2(:));
	max_patch = max(patch_tmp2(:));
	patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch-min_patch)==0));
	patch_tmp2 = uint8(patch_tmp2);
	imagesc(patch_tmp2); colormap(gray);
	box off
	axis off
      endfor
    endfor
    drawnow;
    saveas(DoG_fig, [recon_dir, filesep, "DoG_weights.png"]);
  endif

  %% parse activity files
  Retina_file = [output_dir, filesep, "a1_Retina.pvp"];
  Ganglion_file = [output_dir, filesep, "a3_Ganglion.pvp"];
  %%Error_file = [output_dir, filesep, "a5_Error.pvp"];
  if deep_flag
   Recon_file = [output_dir, filesep, "a6_Recon.pvp"];
   Recon2_file = [output_dir, filesep, "a9_Recon2.pvp"];
    %%Error2_file = [output_dir, filesep, "a7_Error2.pvp"];
   if deep2_flag
     ReconInfra_file = [output_dir, filesep, "a12_ReconInfra.pvp"];
   endif
  else
    Recon_file = [output_dir, filesep, "a4_Recon.pvp"];
  endif
  write_step = frame_duration;
  num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step) + 1;
  start_frame = max(1,num_frames-num_recon+1); %% floor((first_checkpoint_ndx) / write_step);
  [Retina_struct, Retina_hdr] = ...
      readpvpfile(Retina_file, num_frames, num_frames, start_frame);
  num_Retina_frames = size(Retina_struct,1);
  [Ganglion_struct, Ganglion_hdr] = ...
      readpvpfile(Ganglion_file, num_frames, start_frame + num_Retina_frames - 1, start_frame);
  if training_flag
    [Recon_struct, Recon_hdr] = ...
	readpvpfile(Recon_file, num_frames, start_frame + num_Retina_frames - 1, start_frame);
    if deep_flag      
      [Recon2_struct, Recon2_hdr] = ...
	  readpvpfile(Recon2_file, num_frames, start_frame + num_Retina_frames - 1, start_frame);
      if deep2_flag
	[ReconInfra_struct, ReconInfra_hdr] = ...
	    readpvpfile(ReconInfra_file, num_frames, start_frame + num_Retina_frames - 1, start_frame);
      endif
    endif
  else  %% 
    [Recon_struct, Recon_hdr] = ...
	readpvpfile(Recon_file, [], [], []);
    if deep_flag      
      [Recon2_struct, Recon2_hdr] = ...
	  readpvpfile(Recon2_file, [], [], []);
      if deep2_flag
	[ReconInfra_struct, ReconInfra_hdr] = ...
	    readpvpfile(ReconInfra_file, [], [], []);
      endif
    endif
  endif
  %%[Error_struct, Error_hdr] = readpvpfile(Error_file, num_frames, num_frames, start_frame);
  num_Ganglion_frames = size(Ganglion_struct,1);
  num_Recon_frames = size(Recon_struct,1);
  if deep_flag      
    num_Recon2_frames = size(Recon2_struct,1);
    if deep2_flag      
      num_ReconInfra_frames = size(ReconInfra_struct,1);
    endif
  endif

  Retina_fig = figure;
  Retina_mean = 0;
  Retina_std = 0;
  for i_frame = num_Retina_frames - num_recon + 1: 1 : num_Retina_frames
    Retina_time = Retina_struct{i_frame}.time;
    Retina_vals = Retina_struct{i_frame}.values;
    Retina_mean = Retina_mean + mean(Retina_vals(:));
    Retina_std = Retina_std + std(Retina_vals(:));
    set(Retina_fig, "name", ["Retina ", num2str(Retina_time, "%0d")]);
    imagesc(permute(Retina_vals,[2,1,3])); 
    num_Retina_colors = size(Retina_vals,3);
    if num_Retina_colors == 1
      colormap(gray); 
    endif
    box off; axis off; axis image;
    saveas(Retina_fig, [recon_dir, filesep, "Retina_", num2str(Retina_time, "%0d")], "png");
  endfor   %% i_frame
  Retina_mean = Retina_mean / (num_recon + (num_recon == 0));
  Retina_std = Retina_std / (num_recon + (num_recon == 0));
  disp(["Retina_mean = ", num2str(Retina_mean), " +/- ", num2str(Retina_std)]);

  Ganglion_fig = figure;
  Ganglion_mean = 0;
  Ganglion_std = 0;
  if plot_DoG_kernel
    unwhitened_Ganglion_fig = figure;
  endif
  for i_frame = num_Ganglion_frames - num_recon + 1 : 1 : num_Ganglion_frames
    Ganglion_time = Ganglion_struct{i_frame}.time;
    Ganglion_vals = Ganglion_struct{i_frame}.values;
    Ganglion_mean = Ganglion_mean + mean(Ganglion_vals(:));
    Ganglion_std = Ganglion_std + std(Ganglion_vals(:));
    figure(Ganglion_fig);
    set(Ganglion_fig, "name", ["Ganglion ", num2str(Ganglion_time, "%0d")]);
    imagesc(permute(Ganglion_vals,[2,1,3])); 
    num_Ganglion_colors = size(Ganglion_vals,3);
    mean_unwhitened_Ganglion = ones(num_Ganglion_colors,1);
    std_unwhitened_Ganglion = ones(num_Ganglion_colors,1);
    max_unwhitened_Ganglion = ones(num_Ganglion_colors,1);
    min_unwhitened_Ganglion = ones(num_Ganglion_colors,1);
    if num_Ganglion_colors == 1
      colormap(gray); 
    endif
    box off; axis off; axis image;
    saveas(Ganglion_fig, [recon_dir, filesep, "Ganglion_", num2str(Ganglion_time, "%0d")], "png");
    if plot_DoG_kernel
      unwhitened_Ganglion_DoG = zeros(size(permute(Ganglion_vals,[2,1,3])));
      for i_color = 1 : num_Ganglion_colors
	tmp_Ganglion = ...
	    deconvolvemirrorbc(squeeze(Ganglion_vals(:,:,i_color))', DoG_weights);
	mean_unwhitened_Ganglion(i_color) = mean(tmp_Ganglion(:));
	std_unwhitened_Ganglion(i_color) = std(tmp_Ganglion(:));
	max_unwhitened_Ganglion(i_color) = max(tmp_Ganglion(:));
	min_unwhitened_Ganglion(i_color) = min(tmp_Ganglion(:));
	[unwhitened_Ganglion_DoG(:,:,i_color)] = tmp_Ganglion;
      endfor
      figure(unwhitened_Ganglion_fig);
      set(unwhitened_Ganglion_fig, "name", ["unwhitened Ganglion ", num2str(Ganglion_time, "%0d")]);
      imagesc(squeeze(unwhitened_Ganglion_DoG)); 
      if num_Ganglion_colors == 1
	colormap(gray); 
      endif
      box off; axis off; axis image;
      saveas(unwhitened_Ganglion_fig, ...
	     [recon_dir, filesep, "unwhitened_Ganglion_", num2str(Ganglion_time, "%0d")], "png");
      drawnow
    endif %% plot_DoG_kernel
  endfor   %% i_frame
  Ganglion_mean = Ganglion_mean / (num_recon + (num_recon == 0));
  Ganglion_std = Ganglion_std / (num_recon + (num_recon == 0));
  disp(["Ganglion_mean = ", num2str(Ganglion_mean), " +/- ", num2str(Ganglion_std)]);

  if training_flag
    recon_start_frame = num_Recon_frames - num_recon + 1; %%
  else
    recon_start_frame = 1;
  endif
  Recon_fig = figure;
  Recon_mean = 0;
  Recon_std = 0;
  if plot_DoG_kernel
    unwhitened_Recon_fig = figure;
  endif
  for i_frame = recon_start_frame : 1 : num_Recon_frames
    Recon_time = Recon_struct{i_frame}.time;
    Recon_vals = Recon_struct{i_frame}.values;
    Recon_mean = Recon_mean + mean(Recon_vals(:));
    Recon_std = Recon_std + std(Recon_vals(:));
    figure(Recon_fig);
    set(Recon_fig, "name", ["Recon ", num2str(Recon_time, "%0d")]);
    num_Recon_colors = size(Recon_vals,3);
    imagesc(permute(Recon_vals, [2,1,3])); 
    if num_Recon_colors == 1
      colormap(gray); 
    endif
    box off; axis off; axis image;
    saveas(Recon_fig, [recon_dir, filesep, "Recon_", num2str(Recon_time, "%0d")], "png");
    if plot_DoG_kernel
      unwhitened_Recon_vals = zeros(size(permute(Recon_vals,[2,1,3])));
      for i_color = 1 : num_Recon_colors
	[unwhitened_Recon_vals(:,:,i_color)] = ...
	    deconvolvemirrorbc(squeeze(Recon_vals(:,:,i_color))', DoG_weights); 
	tmp_recon = unwhitened_Recon_vals(:,:,i_color);
	mean_recon = mean(tmp_recon(:));
	std_recon = std(tmp_recon(:));
	%% inverse DoG filtering is ambiguous to within an overall additive constant
	tmp_recon = (tmp_recon - mean_recon) * (std_unwhitened_Ganglion(i_color) / std_recon) + mean_unwhitened_Ganglion(i_color); 
	max_recon = max(tmp_recon(:));
	min_recon = min(tmp_recon(:));
	%%tmp_recon = (tmp_recon - min_recon) / (max_recon - min_recon + ((max_recon - min_recon)==0));
	unwhitened_Recon_vals(:,:,i_color) = tmp_recon;
      endfor
      figure(unwhitened_Recon_fig);
      set(unwhitened_Recon_fig, "name", ["unwhitened Recon ", num2str(Recon_time, "%0d")]);
      imagesc(squeeze(unwhitened_Recon_vals)); 
      if num_Recon_colors == 1
	colormap(gray);
      endif
      box off; axis off; axis image;
      saveas(unwhitened_Recon_fig, ...
	     [recon_dir, filesep, "unwhitened_Recon_", num2str(Recon_time, "%0d")], "png");
      drawnow
    endif %% plot_DoG_kernel
  endfor %% i_frames
  Recon_mean = Recon_mean / (num_recon + (num_recon == 0));
  Recon_std = Recon_std / (num_recon + (num_recon == 0));
  disp(["Recon_mean = ", num2str(Recon_mean), " +/- ", num2str(Recon_std)]);
  drawnow;

  if deep_flag
    if training_flag
      recon2_start_frame = num_Recon2_frames - num_recon + 1; %%
    else
      recon2_start_frame = 1;
    endif
    Recon2_fig = figure;
    Recon2_mean = 0;
    Recon2_std = 0;
    for i_frame = recon2_start_frame : 1 : num_Recon2_frames
      Recon2_time = Recon2_struct{i_frame}.time;
      Recon2_vals = Recon2_struct{i_frame}.values;
      Recon2_mean = Recon2_mean + mean(Recon2_vals(:));
      Recon2_std = Recon2_std + std(Recon2_vals(:));
      figure(Recon2_fig);
      set(Recon2_fig, "name", ["Recon2 ", num2str(Recon2_time)]);
      num_Recon2_colors = size(Recon2_vals,3);
      imagesc(permute(Recon2_vals, [2,1,3])); 
      if num_Recon2_colors == 1
	colormap(gray); 
      endif
      box off; axis off; axis image;
      saveas(Recon2_fig, [recon2_dir, filesep, "Recon2_", num2str(Recon2_time, "%0d")], "png");
    endfor %% i_frames
    Recon2_mean = Recon2_mean / (num_recon + (num_recon == 0));
    Recon2_std = Recon2_std / (num_recon + (num_recon == 0));
    disp(["Recon2_mean = ", num2str(Recon2_mean), " +/- ", num2str(Recon2_std)]);
    drawnow;
    if deep2_flag
      if training_flag
	reconInfra_start_frame = num_ReconInfra_frames - num_recon + 1; %%
      else
	reconInfra_start_frame = 1;
      endif
      ReconInfra_fig = figure;
      ReconInfra_mean = 0;
      ReconInfra_std = 0;
      if plot_DoG_kernel
	unwhitened_ReconSum_fig = figure;
      endif
      for i_frame = reconInfra_start_frame : 1 : num_ReconInfra_frames
	ReconInfra_time = ReconInfra_struct{i_frame}.time;
	ReconInfra_vals = ReconInfra_struct{i_frame}.values;
	ReconInfra_mean = ReconInfra_mean + mean(ReconInfra_vals(:));
	ReconInfra_std = ReconInfra_std + std(ReconInfra_vals(:));
	figure(ReconInfra_fig);
	set(ReconInfra_fig, "name", ["ReconInfra ", num2str(ReconInfra_time)]);
	num_ReconInfra_colors = size(ReconInfra_vals,3);
	imagesc(permute(ReconInfra_vals, [2,1,3])); 
	if num_ReconInfra_colors == 1
	  colormap(gray); 
	endif
	box off; axis off; axis image;
	saveas(ReconInfra_fig, [reconInfra_dir, filesep, "ReconInfra_", num2str(ReconInfra_time, "%0d")], "png");
      endfor %% i_frames
      ReconInfra_mean = ReconInfra_mean / (num_recon + (num_recon == 0));
      ReconInfra_std = ReconInfra_std / (num_recon + (num_recon == 0));
      disp(["ReconInfra_mean = ", num2str(ReconInfra_mean), " +/- ", num2str(ReconInfra_std)]);
      drawnow;

      ReconSum_fig = figure;
      ReconSum_mean = 0;
      ReconSum_std = 0;
      for i_frame = reconInfra_start_frame : 1 : num_ReconInfra_frames
	ReconSum_time = ReconInfra_struct{i_frame}.time;
	ReconInfra_vals = ReconInfra_struct{i_frame}.values;
	Recon_vals = Recon_struct{i_frame}.values;
	ReconSum_vals = Recon_vals + ReconInfra_vals;
	ReconSum_mean = ReconSum_mean + mean(ReconSum_vals(:));
	ReconSum_std = ReconSum_std + std(ReconSum_vals(:));
	figure(ReconSum_fig);
	set(ReconSum_fig, "name", ["ReconSum ", num2str(ReconSum_time)]);
	num_ReconSum_colors = size(ReconSum_vals,3);
	imagesc(permute(ReconSum_vals, [2,1,3])); 
	if num_ReconSum_colors == 1
	  colormap(gray); 
	endif
	box off; axis off; axis image;
	saveas(ReconSum_fig, [reconInfra_dir, filesep, "ReconSum_", num2str(ReconSum_time, "%0d")], "png");
	if plot_DoG_kernel
	  unwhitened_ReconSum_vals = zeros(size(permute(ReconSum_vals,[2,1,3])));
	  for i_color = 1 : num_ReconSum_colors
	    [unwhitened_ReconSum_vals(:,:,i_color)] = ...
		deconvolvemirrorbc(squeeze(ReconSum_vals(:,:,i_color))', DoG_weights); 
	    tmp_recon = unwhitened_ReconSum_vals(:,:,i_color);
	    mean_recon = mean(tmp_recon(:));
	    mean_recon = mean(tmp_recon(:));
	    std_recon = std(tmp_recon(:));
	    %% inverse DoG filtering is ambiguous to within an overall additive constant
	    tmp_recon = (tmp_recon - mean_recon) * (std_unwhitened_Ganglion(i_color) / std_recon) + mean_unwhitened_Ganglion(i_color); 
	    %%max_recon = max(tmp_recon(:));
	    %%min_recon = min(tmp_recon(:));
	    %%tmp_recon = (tmp_recon - min_recon) / (max_recon - min_recon + ((max_recon - min_recon)==0));
	    unwhitened_ReconSum_vals(:,:,i_color) = tmp_recon;
	  endfor
	  figure(unwhitened_ReconSum_fig);
	  set(unwhitened_ReconSum_fig, "name", ["unwhitened ReconSum ", num2str(ReconSum_time, "%0d")]);
	  imagesc(squeeze(unwhitened_ReconSum_vals)); 
	  if num_ReconSum_colors == 1
	    colormap(gray);
	  endif
	  box off; axis off; axis image;
	  saveas(unwhitened_ReconSum_fig, ...
		 [recon_dir, filesep, "unwhitened_ReconSum_", num2str(ReconSum_time, "%0d")], "png");
	  drawnow
	endif %% plot_DoG_kernel
      endfor %% i_frames
      ReconSum_mean = ReconSum_mean / (num_recon + (num_recon == 0));
      ReconSum_std = ReconSum_std / (num_recon + (num_recon == 0));
      disp(["ReconSum_mean = ", num2str(ReconSum_mean), " +/- ", num2str(ReconSum_std)]);
      drawnow;
    endif %% deep2_flag
  endif  %% deep_flag
endif  %% plot_Recon


plot_StatsProbe_vs_time = 1;
if plot_StatsProbe_vs_time
  V1_sparseness_error_vs_time_dir = [output_dir, filesep, "V1_sparseness_error_vs_time"];
  mkdir(V1_sparseness_error_vs_time_dir);
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
  error_vs_time_hndl = plot(ave_error); axis tight;
  axis tight
  set(error_vs_time_fig, "name", ["ave Error"]);
  saveas(error_vs_time_fig, [V1_sparseness_error_vs_time_dir, filesep, "error_vs_time_", num2str(last_checkpoint_ndx, "%i")], "png");
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
  V1_vs_time_hndl = plot(ave_V1); axis tight;
  axis tight
  set(V1_vs_time_fig, "name", ["V1_sparseness_vs_time"]);
  saveas(V1_vs_time_fig, [V1_sparseness_error_vs_time_dir, filesep, "V1_sparseness_vs_time_", num2str(last_checkpoint_ndx, "%i")], "png");
  drawnow;
  if deep_flag
    V2_sparseness_error2_vs_time_dir = [output_dir, filesep, "V2_sparseness_error2_vs_time"];
    mkdir(V2_sparseness_error2_vs_time_dir);
    Error2_Stats_file = [output_dir, filesep, "Error2_Stats.txt"];
    Error2_Stats_fid = fopen(Error2_Stats_file, "r");
    Error2_Stats_line = fgets(Error2_Stats_fid);
    ave_error2 = [];
    %% skip startup artifact
    first_statProbe_step = first_checkpoint_ndx;
    for i_line = first_statProbe_step:begin_statProbe_step
      Error2_Stats_line = fgets(Error2_Stats_fid);
    endfor
    while (Error2_Stats_line ~= -1)
      Error2_Stats_ndx1 = strfind(Error2_Stats_line, "sigma==");
      Error2_Stats_ndx2 = strfind(Error2_Stats_line, "nnz==");
      Error2_Stats_str = Error2_Stats_line(Error2_Stats_ndx1+7:Error2_Stats_ndx2-2);
      Error2_Stats_val = str2num(Error2_Stats_str);
      if isempty(ave_error2)
	ave_error2 = Error2_Stats_val;
      else
	ave_error2 = [ave_error2; Error2_Stats_val];
      endif
      Error2_Stats_line = fgets(Error2_Stats_fid);
      i_line = i_line + 1;
      if i_line > max_lines
	break;
      endif
    endwhile
    fclose(Error2_Stats_fid);
    error2_vs_time_fig = figure;
    error2_vs_time_hndl = plot(ave_error2);
    axis tight
    set(error2_vs_time_fig, "name", ["ave Error2"]);
    saveas(error2_vs_time_fig, [V2_sparseness_error2_vs_time_dir, filesep, "error2_vs_time_", num2str(last_checkpoint_ndx, "%i")], "png");
    drawnow;
    %%endif
    %%
    %%plot_ave_V1_vs_time = 0;
    %%if plot_ave_V1_vs_time
    V2_Stats_file = [output_dir, filesep, "V2_Stats.txt"];
    V2_Stats_fid = fopen(V2_Stats_file, "r");
    V2_Stats_line = fgets(V2_Stats_fid);
    ave_V2 = [];
    %% skip startup artifact
    for i_line = first_statProbe_step:begin_statProbe_step
      V2_Stats_line = fgets(V2_Stats_fid);
    endfor
    V2_Stats_ndx1 = strfind(V2_Stats_line, "N==");
    V2_Stats_ndx2 = strfind(V2_Stats_line, "Total==");
    V2_Stats_str = V2_Stats_line(V2_Stats_ndx1+3:V2_Stats_ndx2-2);
    num_V2 = str2num(V2_Stats_str);
    while (V2_Stats_line ~= -1)
      V2_Stats_ndx1 = strfind(V2_Stats_line, "nnz==");
      V2_Stats_ndx2 = length(V2_Stats_line); %% strfind(V2_Stats_line, "Max==");
      V2_Stats_str = V2_Stats_line(V2_Stats_ndx1+5:V2_Stats_ndx2-1);
      V2_Stats_val = str2num(V2_Stats_str);
      if isempty(ave_V2)
	ave_V2 = V2_Stats_val/num_V2;
      else
	ave_V2 = [ave_V2; V2_Stats_val/num_V2];
      endif
      V2_Stats_line = fgets(V2_Stats_fid);
      i_line = i_line + 1;
      if i_line > max_lines
	break;
      endif
    endwhile
    fclose(V2_Stats_fid);
    V2_vs_time_fig = figure;
    V2_vs_time_hndl = plot(ave_V2);
    axis tight
    set(V2_vs_time_fig, "name", ["V2_sparseness_vs_time"]);
    saveas(V2_vs_time_fig, [V2_sparseness_error2_vs_time_dir, filesep, "V2_sparseness_vs_time_", num2str(last_checkpoint_ndx, "%i")], "png");
    drawnow;
    if deep2_flag
      Error1_2_Stats_file = [output_dir, filesep, "Error1_2_Stats.txt"];
      Error1_2_Stats_fid = fopen(Error1_2_Stats_file, "r");
      Error1_2_Stats_line = fgets(Error1_2_Stats_fid);
      ave_error2 = [];
      %% skip startup artifact
      first_statProbe_step = first_checkpoint_ndx;
      for i_line = first_statProbe_step:begin_statProbe_step
	Error1_2_Stats_line = fgets(Error1_2_Stats_fid);
      endfor
      while (Error1_2_Stats_line ~= -1)
	Error1_2_Stats_ndx1 = strfind(Error1_2_Stats_line, "sigma==");
	Error1_2_Stats_ndx2 = strfind(Error1_2_Stats_line, "nnz==");
	Error1_2_Stats_str = Error1_2_Stats_line(Error1_2_Stats_ndx1+7:Error1_2_Stats_ndx2-2);
	Error1_2_Stats_val = str2num(Error1_2_Stats_str);
	if isempty(ave_error2)
	  ave_error2 = Error1_2_Stats_val;
	else
	  ave_error2 = [ave_error2; Error1_2_Stats_val];
	endif
	Error1_2_Stats_line = fgets(Error1_2_Stats_fid);
	i_line = i_line + 1;
	if i_line > max_lines
	  break;
	endif
      endwhile
      fclose(Error1_2_Stats_fid);
      error2_vs_time_fig = figure;
      error2_vs_time_hndl = plot(ave_error2);
      axis tight
      set(error2_vs_time_fig, "name", ["ave Error1_2"]);
      saveas(error2_vs_time_fig, [V2_sparseness_error2_vs_time_dir, filesep, "error2_vs_time_", num2str(last_checkpoint_ndx, "%i")], "png");
      drawnow;
    endif %% deep2_flag
  endif  %% deep_flag
endif  %% ave_error_vs_time

plot_V1 = 1;
if plot_V1
  if deep_flag
    V1_path = [output_dir, filesep, "a5_V1.pvp"];
  else
    V1_path = [output_dir, filesep, "a6_V1.pvp"];
  endif
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
    V1_current_active = nnz(V1_current(:));
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
  V1_mean_change = mean(V1_percent_change(:));
  disp(["V1_mean_change = ", num2str(V1_mean_change)]);

  V1_tot_active_title = ["V1_tot_active", ".png"];
  V1_tot_active_fig = figure;
  V1_tot_active_hndl = plot(V1_times, V1_tot_active/n_V1); axis tight;
  set(V1_tot_active_fig, "name", ["V1_tot_active"]);
  saveas(V1_tot_active_fig, ...
	 [V1_change_dir, filesep, "V1_tot_active", num2str(V1_times(num_frames), "%i")], "png");

  V1_mean_active = mean(V1_tot_active(:)/n_V1);
  disp(["V1_mean_active = ", num2str(V1_mean_active)]);

  plot_Error = 1;
  if plot_Error
  if deep_flag
    Error_path = [output_dir, filesep, "a4_Error.pvp"];
  else
    Error_path = [output_dir, filesep, "a5_Error.pvp"];
  endif
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

  Error_median_RMS = median(Error_RMS(:));
  disp(["Error_median_RMS = ", num2str(Error_median_RMS)]);

  drawnow;  
  endif %% plot_Error
  if deep_flag
    V2_path = [output_dir, filesep, "a8_V2.pvp"];
    write_step = frame_duration;
    num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
    [V2_struct, V2_hdr] = readpvpfile(V2_path, num_frames, num_frames, 1);
    nx_V2 = V2_hdr.nx;
    ny_V2 = V2_hdr.ny;
    nf_V2 = V2_hdr.nf;
    n_V2 = nx_V2 * ny_V2 * nf_V2;
    num_frames = size(V2_struct,1);
    i_frame = num_frames;
    start_frame = 1; %%
    V2_hist = zeros(nf_V2+1,1);
    V2_hist_edges = [0:1:nf_V2]+0.5;
    V2_current = zeros(n_V2,1);
    V2_abs_change = zeros(num_frames,1);
    V2_percent_change = zeros(num_frames,1);
    V2_current_active = 0;
    V2_tot_active = zeros(num_frames,1);
    V2_times = zeros(num_frames,1);
    for i_frame = 1 : 1 : num_frames
      V2_times(i_frame) = squeeze(V2_struct{i_frame}.time);
      V2_active_ndx = squeeze(V2_struct{i_frame}.values);
      V2_previous = V2_current;
      V2_current = full(sparse(V2_active_ndx+1,1,1,n_V2,1,n_V2));
      V2_abs_change(i_frame) = sum(V2_current(:) ~= V2_previous(:));
      V2_previous_active = V2_current_active;
      V2_current_active = nnz(V2_current(:));
      V2_tot_active(i_frame) = V2_current_active;
      V2_max_active = max(V2_current_active, V2_previous_active);
      V2_percent_change(i_frame) = ...
	  V2_abs_change(i_frame) / (V2_max_active + (V2_max_active==0));
      V2_active_kf = mod(V2_active_ndx, nf_V2) + 1;
      if V2_max_active > 0
	V2_hist_frame = histc(V2_active_kf, V2_hist_edges);
      else
	V2_hist_frame = zeros(nf_V2+1,1);
      endif
      V2_hist = V2_hist + V2_hist_frame;
    endfor %% i_frame
    V2_hist = V2_hist(1:nf_V2);
    V2_hist = V2_hist / (num_frames * nx_V2 * ny_V2); %% (sum(V2_hist(:)) + (nnz(V2_hist)==0));
    [V2_hist_sorted, V2_hist_rank] = sort(V2_hist, 1, "descend");
    V2_hist_title = ["V2_hist", ".png"];
    V2_hist_fig = figure;
    V2_hist_bins = 1:nf_V2;
    V2_hist_hndl = bar(V2_hist_bins, V2_hist_sorted); axis tight;
    set(V2_hist_fig, "name", ["V2_hist_", num2str(V2_times(num_frames), "%i")]);
    V2_rank_dir = [output_dir, filesep, "V2_rank"];
    mkdir(V2_rank_dir);
    saveas(V2_hist_fig, ...
	   [V2_rank_dir, filesep, ...
	    "V2_rank_", num2str(V2_times(num_frames), "%i")], "png");

    V2_abs_change_title = ["V2_abs_change", ".png"];
    V2_abs_change_fig = figure;
    V2_abs_change_hndl = plot(V2_times, V2_abs_change); axis tight;
    set(V2_abs_change_fig, "name", ["V2_abs_change"]);
    V2_change_dir = [output_dir, filesep, "V2_rank"];
    mkdir(V2_change_dir);
    saveas(V2_abs_change_fig, ...
	   [V2_change_dir, filesep, "V2_abs_change", num2str(V2_times(num_frames), "%i")], "png");

    V2_percent_change_title = ["V2_percent_change", ".png"];
    V2_percent_change_fig = figure;
    V2_percent_change_hndl = plot(V2_times, V2_percent_change); axis tight;
    set(V2_percent_change_fig, "name", ["V2_percent_change"]);
    saveas(V2_percent_change_fig, ...
	   [V2_change_dir, filesep, "V2_percent_change", num2str(V2_times(num_frames), "%i")], "png");
    V2_mean_change = mean(V2_percent_change(:));
    disp(["V2_mean_change = ", num2str(V2_mean_change)]);

    V2_tot_active_title = ["V2_tot_active", ".png"];
    V2_tot_active_fig = figure;
    V2_tot_active_hndl = plot(V2_times, V2_tot_active/n_V2); axis tight;
    set(V2_tot_active_fig, "name", ["V2_tot_active"]);
    saveas(V2_tot_active_fig, ...
	   [V2_change_dir, filesep, "V2_tot_active", num2str(V2_times(num_frames), "%i")], "png");

    V2_mean_active = mean(V2_tot_active(:)/n_V2);
    disp(["V2_mean_active = ", num2str(V2_mean_active)]);

    Error2_path = [output_dir, filesep, "a7_Error2.pvp"];
    write_step = frame_duration;
    num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
    [Error2_struct, Error2_hdr] = readpvpfile(Error2_path, num_frames, num_frames, 1);
    nx_Error2 = Error2_hdr.nx;
    ny_Error2 = Error2_hdr.ny;
    nf_Error2 = Error2_hdr.nf;
    n_Error2 = nx_Error2 * ny_Error2 * nf_Error2;
    num_frames = size(Error2_struct,1);
    i_frame = num_frames;
    start_frame = 1; %%
    Error2_RMS = zeros(num_frames,1);
    Error2_times = zeros(num_frames,1);
    for i_frame = 1 : 1 : num_frames
      Error2_times(i_frame) = squeeze(Error2_struct{i_frame}.time);
      Error2_vals = squeeze(Error2_struct{i_frame}.values);
      Error2_RMS(i_frame) = std(Error2_vals(:));;
    endfor %% i_frame

    Error2_RMS_title = ["Error2_RMS", ".png"];
    Error2_RMS_fig = figure;
    Error2_RMS_hndl = plot(Error2_times, Error2_RMS); axis tight;
    axis([min(Error2_times(:)), max(Error2_times(:)), min(Error2_RMS(:)), max(Error_RMS(:))]); %% use layer 1 max value
    set(Error2_RMS_fig, "name", ["Error2_RMS"]);
    saveas(Error2_RMS_fig, ...
	   [V2_change_dir, filesep, "Error2_RMS", num2str(Error2_times(num_frames), "%i")], "png");

    Error2_median_RMS = median(Error2_RMS(:));
    disp(["Error2_median_RMS = ", num2str(Error2_median_RMS)]);

    drawnow;  
    if deep2_flag
      Error1_2_path = [output_dir, filesep, "a10_Error1_2.pvp"];
      write_step = frame_duration;
      num_frames = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
      [Error1_2_struct, Error1_2_hdr] = readpvpfile(Error1_2_path, num_frames, num_frames, 1);
      nx_Error1_2 = Error1_2_hdr.nx;
      ny_Error1_2 = Error1_2_hdr.ny;
      nf_Error1_2 = Error1_2_hdr.nf;
      n_Error1_2 = nx_Error1_2 * ny_Error1_2 * nf_Error1_2;
      num_frames = size(Error1_2_struct,1);
      i_frame = num_frames;
      start_frame = 1; %%
      Error1_2_RMS = zeros(num_frames,1);
      Error1_2_times = zeros(num_frames,1);
      for i_frame = 1 : 1 : num_frames
	Error1_2_times(i_frame) = squeeze(Error1_2_struct{i_frame}.time);
	Error1_2_vals = squeeze(Error1_2_struct{i_frame}.values);
	Error1_2_RMS(i_frame) = std(Error1_2_vals(:));;
      endfor %% i_frame
      
      Error1_2_RMS_title = ["Error1_2_RMS", ".png"];
      Error1_2_RMS_fig = figure;
      Error1_2_RMS_hndl = plot(Error1_2_times, Error1_2_RMS); axis tight;
      axis([min(Error1_2_times(:)), max(Error1_2_times(:)), min(Error1_2_RMS(:)), max(Error_RMS(:))]); %% use layer 1 max value
      set(Error1_2_RMS_fig, "name", ["Error1_2_RMS"]);
      saveas(Error1_2_RMS_fig, ...
	     [V2_change_dir, filesep, "Error1_2_RMS", num2str(Error1_2_times(num_frames), "%i")], "png");
      
      Error1_2_median_RMS = median(Error1_2_RMS(:));
      disp(["Error1_2_median_RMS = ", num2str(Error1_2_median_RMS)]);
      
      drawnow;  
    endif  %% deep2_flag
  endif  %% deep_flag
endif  %% plot_V1

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
  set(V1ToError_fig, "name", ["V1ToError Weights: ", num2str(last_checkpoint_ndx, "%i")]);
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


  if deep_flag
    V2ToError2_path = [checkpoint_path, filesep, "V2ToError2_W.pvp"];
    if ~exist(V2ToError2_path, "file")
      V2ToError2_path = [next_checkpoint_path, filesep, "V2ToError2_W.pvp"];
    endif
    [V2ToError2_struct, V2ToError2_hdr] = readpvpfile(V2ToError2_path,1);
    i_frame = 1;
    i_arbor = 1;
    V2ToError2_weights = squeeze(V2ToError2_struct{i_frame}.values{i_arbor});
    if ~exist("V2_hist_rank") || isempty(V2_hist_rank)
      V2_hist_rank = (1:V2ToError2_hdr.nf);
    endif

    %% make tableau of all patches
    %%keyboard;
    i_patch = 1;
    num_V2ToError2_dims = ndims(V2ToError2_weights);
    num_patches = size(V2ToError2_weights, num_V2ToError2_dims);
    num_patches_rows = floor(sqrt(num_patches));
    num_patches_cols = ceil(num_patches / num_patches_rows);
    num_V2ToError2_colors = 1;
    if num_V2ToError2_dims == 4
      num_V2ToError2_colors = size(V2ToError2_weights,3);
    endif
    V2ToError2_fig = figure;
    set(V2ToError2_fig, "name", ["V2ToError2 Weights: ", num2str(last_checkpoint_ndx, "%i")]);
    for j_patch = 1  : num_patches
      i_patch = V2_hist_rank(j_patch);
      subplot(num_patches_rows, num_patches_cols, j_patch); 
      if num_V2ToError2_colors == 1
	patch_tmp = squeeze(V2ToError2_weights(:,:,i_patch));
      else
	patch_tmp = squeeze(V2ToError2_weights(:,:,:,i_patch));
      endif
      patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
      min_patch = min(patch_tmp2(:));
      max_patch = max(patch_tmp2(:));
      patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
      patch_tmp2 = uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
      imagesc(patch_tmp2); 
      if num_V2ToError2_colors == 1
	colormap(gray);
      endif
      box off
      axis off
      axis image
      %%drawnow;
    endfor
    V2ToError2_weights_dir = [output_dir, filesep, "V2ToError2_weights"];
    mkdir(V2ToError2_weights_dir);
    saveas(V2ToError2_fig, [V2ToError2_weights_dir, filesep, "V2ToError2_", num2str(last_checkpoint_ndx, "%i")], "png");

    %% make histogram of all weights
    V2ToError2_hist_fig = figure;
    [V2ToError2_hist, V2ToError2_hist_bins] = hist(V2ToError2_weights(:), 100);
    bar(V2ToError2_hist_bins, log(V2ToError2_hist+1));
    set(V2ToError2_hist_fig, "name", ["V2ToError2 Histogram: ", num2str(last_checkpoint_ndx, "%i")]);
    V2ToError2_weights_hist_dir = [output_dir, filesep, "V2ToError2_hist"];
    mkdir(V2ToError2_weights_hist_dir);
    saveas(V2ToError2_hist_fig, [V2ToError2_weights_hist_dir, filesep, "V2ToError2_hist_", num2str(last_checkpoint_ndx, "%i")], "png");

    mat_dir = [output_dir, filesep, "mat"];
    mkdir(mat_dir);
    V2ToError2_weights_file = [mat_dir, filesep, "V2ToError2_weights", num2str(last_checkpoint_ndx, "%i"), ".mat"];
    save(  "-mat", V2ToError2_weights_file, "V2ToError2_weights");

    if 0 %% deep2_flag
      V2ToError1_2_path = [checkpoint_path, filesep, "V2ToError1_2_W.pvp"];
      if ~exist(V2ToError1_2_path, "file")
	V2ToError1_2_path = [next_checkpoint_path, filesep, "V2ToError1_2_W.pvp"];
      endif
      [V2ToError1_2_struct, V2ToError1_2_hdr] = readpvpfile(V2ToError1_2_path,1);
      i_frame = 1;
      i_arbor = 1;
      V2ToError1_2_weights = squeeze(V2ToError1_2_struct{i_frame}.values{i_arbor});
      if ~exist("V2_hist_rank") || isempty(V2_hist_rank)
	V2_hist_rank = (1:V2ToError1_2_hdr.nf);
      endif

      %% make tableau of all patches
      %%keyboard;
      i_patch = 1;
      num_V2ToError1_2_dims = ndims(V2ToError1_2_weights);
      num_patches = size(V2ToError1_2_weights, num_V2ToError1_2_dims);
      num_patches_rows = floor(sqrt(num_patches));
      num_patches_cols = ceil(num_patches / num_patches_rows);
      num_V2ToError1_2_colors = 1;
      if num_V2ToError1_2_dims == 4
	num_V2ToError1_2_colors = size(V2ToError1_2_weights,3);
      endif
      V2ToError1_2_fig = figure;
      set(V2ToError1_2_fig, "name", ["V2ToError1_2 Weights: ", num2str(last_checkpoint_ndx, "%i")]);
      Retina_V1_nx_ratio = Retina_hdr.nxGlobal / V1_hdr.nxGlobal;
      Retina_V1_ny_ratio = Retina_hdr.nyGlobal / V1_hdr.nyGlobal;
      V1ToError_overlapp_x = V1ToError_hdr.nxp - Retina_V1_nx_ratio;
      V1ToError_overlapp_y = V1ToError_hdr.nyp - Retina_V1_ny_ratio;
      V2ToError1_2_nxp = V1ToError_hdr.nxp + (V2ToError1_2_hdr.nxp - 1) * (V1ToError_hdr.nxp - V1ToError_overlapp_x); 
      V2ToError1_2_nyp = V1ToError_hdr.nyp + (V2ToError1_2_hdr.nyp - 1) * (V1ToError_hdr.nyp - V1ToError_overlapp_y); 
      for j_patch = 1  : num_patches
	i_patch = V2_hist_rank(j_patch);
	subplot(num_patches_rows, num_patches_cols, j_patch); 
	if num_V2ToError1_2_colors == 1
	  patch_ij_tmp = squeeze(V2ToError1_2_weights(:,:,i_patch));
	  patch_ij_tmp = repmat(patch_ij_tmp, [1,1,1,1]);
	else
	  patch_ij_tmp = squeeze(V2ToError1_2_weights(:,:,:,i_patch));
	endif
	patch_ij_array = cell(size(V2ToError1_2_weights,1),size(V2ToError1_2_weights,2));
	patch_ij = zeros(V2ToError1_2_nyp, V2ToError1_2_nxp, num_V1ToError_colors);
	for ij_patch_row = 1 : size(V2ToError1_2_weights,1)
	  for ij_patch_col = 1 : size(V2ToError1_2_weights,2)
	    patch_ij_array{ij_patch_row, ij_patch_col} = zeros(size(V1ToError_weights(1:3)));
	    for ij_patch_feature = 1 : num_V2ToError1_2_colors
	      patch_ij_tmp = patch_tmp(ij_patch_row, ij_path_col, ij_patch_feature);
	      if num_V1ToError_colors == 1
		patch_convert = V1ToError_weights(:,:,ij_patch_feature);
	      else
		patch_convert = V1ToError_weights(:,:,:,ij_patch_feature);
	      endif
	      %%  convert to pixels
	      patch_ij_array{ij_patch_row, ij_patch_col} = ...
		  patch_ij_array{ij_patch_row, ij_patch_col} + ...
		  patch_tmp(ij_patch_row, ij_path_col, ij_patch_feature) * ...
		  patch_convert;
	    endfor %% ij_patch_feature
	    row_start = 1+Retina_V1_ny_ratio*(ij_patch_row-1);
	    row_end = Retina_V1_ny_ratio*(ij_patch_row-1)+V1ToError_hdr.nyp;
	    col_start = 1+Retina_V1_ny_ratio*(ij_patch_col-1);
	    col_end = Retina_V1_ny_ratio*(ij_patch_col-1)+V1ToError_hdr.nyp;
	    patch_ij(row_start:row_end, col_start:col_end, :) = ...
		patch_ij_array{ij_patch_row, ij_patch_col};
	  endfor %% ij_patch_col
	endfor %% ij_patch_row
	min_patch = min(patch_ij(:));
	max_patch = max(patch_ij(:));
	patch_tmp2 = (patch_ij - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
	patch_tmp2 = uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
	imagesc(patch_tmp2); 
	if num_V2ToError1_2_colors == 1
	  colormap(gray);
	endif
	box off
	axis off
	axis image
	%%drawnow;
      endfor
      V2ToError1_2_weights_dir = [output_dir, filesep, "V2ToError1_2_weights"];
      mkdir(V2ToError1_2_weights_dir);
      saveas(V2ToError1_2_fig, [V2ToError1_2_weights_dir, filesep, "V2ToError1_2_", num2str(last_checkpoint_ndx, "%i")], "png");

      %% make histogram of all weights
      V2ToError1_2_hist_fig = figure;
      [V2ToError1_2_hist, V2ToError1_2_hist_bins] = hist(V2ToError1_2_weights(:), 100);
      bar(V2ToError1_2_hist_bins, log(V2ToError1_2_hist+1));
      set(V2ToError1_2_hist_fig, "name", ["V2ToError1_2 Histogram: ", num2str(last_checkpoint_ndx, "%i")]);
      V2ToError1_2_weights_hist_dir = [output_dir, filesep, "V2ToError1_2_hist"];
      mkdir(V2ToError1_2_weights_hist_dir);
      saveas(V2ToError1_2_hist_fig, [V2ToError1_2_weights_hist_dir, filesep, "V2ToError1_2_hist_", num2str(last_checkpoint_ndx, "%i")], "png");

      mat_dir = [output_dir, filesep, "mat"];
      mkdir(mat_dir);
      V2ToError1_2_weights_file = [mat_dir, filesep, "V2ToError1_2_weights", num2str(last_checkpoint_ndx, "%i"), ".mat"];
      save(  "-mat", V2ToError1_2_weights_file, "V2ToError1_2_weights");
      
    endif %% deep2_flag
    
  endif  %% deep_flag
endif  %% plot_final_weights

plot_weights_movie = training_flag;
if plot_weights_movie
  weights_movie_dir = [output_dir, filesep, "V1ToError_movie"];
  mkdir(weights_movie_dir);
  V1ToError_path = [output_dir, filesep, "w5_V1ToError.pvp"];
  V1ToError_old_frames = glob([weights_movie_dir, filesep, "*_V1ToError.png"]);
  num_V1ToError_old_frames = length(V1ToError_old_frames);
  max_old_frame_id = 1;
  for i_old_frame = 1 : num_V1ToError_old_frames
    V1ToError_str = V1ToError_old_frames{i_old_frame};
    old_frame_id_ndx = findstr(V1ToError_str, "_V1ToError.png");
    old_frame_id = str2num(V1ToError_str(old_frame_id_ndx-4:old_frame_id_ndx-1));
    max_old_frame_id = max(max_old_frame_id, old_frame_id);
  endfor
  write_step = frame_duration;
  progress_period = 1000; %%floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
  start_frame = max_old_frame_id;
  V1ToError_fid = fopen(V1ToError_path);
  V1ToError_hdr = readpvpheader(V1ToError_fid);
  fclose(V1ToError_fid);
  max_frames = V1_hdr.nbands;
  frames_per_epoch = 8520;
  num_epochs = ceil((max_frames - start_frame) / frames_per_epoch);
  i_count = 0;
  max_time = -1.0;
  for i_epoch = 1 : num_epochs
    start_epoch = start_frame + (i_epoch-1) * frames_per_epoch
    end_epoch = start_frame + (i_epoch) * frames_per_epoch
    num_frames = min(end_epoch, max_frames)
    [V1ToError_struct, V1ToError_hdr] = readpvpfile(V1ToError_path, progress_period, num_frames, start_frame);
    num_V1ToError_frames = size(V1ToError_struct,1);
    if isempty(V1_hist_rank)
      V1_hist_rank = (1:V1ToError_hdr.nf);
    endif
    i_arbor = 1;
    for i_frame = 1 : num_V1ToError_frames
      if mod(i_frame, max(floor(num_frames/20),1)) == 0
	disp(["writing frame # ", num2str(i_frame+start_frame, "%i")]);
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
      frame_title = [num2str(i_frame+start_frame, "%04d"), "_V1ToError", ".png"];
      imwrite(weights_frame, [weights_movie_dir, filesep, frame_title]);
    endfor %% i_epoch
  endfor %% i_frame

  if deep_flag
    weights_movie_dir = [output_dir, filesep, "V2ToError2_movie"];
    mkdir(weights_movie_dir);
    V2ToError2_path = [output_dir, filesep, "w9_V2ToError2.pvp"];
    V2ToError2_old_frames = glob([weights_movie_dir, filesep, "*_V2ToError2.png"]);
    num_V2ToError2_old_frames = length(V2ToError2_old_frames);
    max_old_frame_id = 0;
    for i_old_frame = 1 : num_V2ToError2_old_frames
      V2ToError2_str = V2ToError2_old_frames{i_old_frame};
      old_frame_id_ndx = findstr(V2ToError2_str, "_V2ToError2.png");
      old_frame_id = str2num(V2ToError2_str(old_frame_id_ndx-4:old_frame_id_ndx-1));
      max_old_frame_id = max(max_old_frame_id, old_frame_id);
    endfor
    write_step = frame_duration;
    progress_period = floor((last_checkpoint_ndx - first_checkpoint_ndx) / write_step);
    start_frame = max_old_frame_id;
    num_frames = start_frame + 1000;
    [V2ToError2_struct, V2ToError2_hdr] = readpvpfile(V2ToError2_path, progress_period, num_frames, start_frame);
    num_frames = size(V2ToError2_struct,1);
    i_frame = num_frames;
    start_frame = 1; 
    if isempty(V2_hist_rank)
      V2_hist_rank = (1:V2ToError2_hdr.nf);
    endif
    num_recon = max(floor(frame_duration / write_step) - 1, 0);
    i_arbor = 1;
    for i_frame = start_frame : 1 : num_frames
      if mod(i_frame, max(floor(num_frames/20),1)) == 0
	disp(["writing frame # ", num2str(i_frame, "%i")]);
      endif
      V2ToError2_weights = squeeze(V2ToError2_struct{i_frame}.values{i_arbor});
      i_patch = 1;
      nyp = size(V2ToError2_weights,1);
      nxp = size(V2ToError2_weights,2);
      num_V1_dims = ndims(V2ToError2_weights);
      num_patches = size(V2ToError2_weights, num_V1_dims);
      num_patches_rows = floor(sqrt(num_patches));
      num_patches_cols = ceil(num_patches / num_patches_rows);
      num_V2_colors = 1;
      if num_V2_dims == 4
	num_V2_colors = size(V2ToError2_weights,3);
      endif
      weights_frame = uint8(zeros(num_patches_rows * nyp, num_patches_cols * nxp, num_V2_colors));
      for j_patch = 1  : num_patches
	i_patch = V2_hist_rank(j_patch);
	j_patch_row = ceil(j_patch / num_patches_cols);
	j_patch_col = 1 + mod(j_patch - 1, num_patches_cols);
	%%subplot(num_patches_rows, num_patches_cols, i_patch); 
	if num_V2_colors == 1
	  patch_tmp = squeeze(V2ToError2_weights(:,:,i_patch));
	else
	  patch_tmp = squeeze(V2ToError2_weights(:,:,:,i_patch));
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
      frame_title = [num2str(i_frame, "%04d"), "_V2ToError2", ".png"];
      imwrite(weights_frame, [weights_movie_dir, filesep, frame_title]);
    endfor %% i_frame

  endif  %% deep_flag
endif  %% plot_weights_movie