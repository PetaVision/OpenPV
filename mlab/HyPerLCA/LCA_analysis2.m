

clear all;
close all;
global plot_flag %% if true, plot graphical output to screen, else do not generate graphical outputy
plot_flag = true;
global load_flag %% if true, then load "saved" data structures rather than computing them 
load_flag = false;
if plot_flag
  setenv("GNUTERM","X11")
endif
no_clobber = false;

%% machine/run_type environment
if ismac
  workspace_path = "/Users/garkenyon/workspace";
  run_type = "CIFAR_deep"; %%"CIFAR_noTask_deep"; %%"CIFAR_noTask"; %%"CIFAR" %%
  output_dir = "/Users/garkenyon/workspace/HyPerHLCA/CIFAR256_RGB_deep_task/test_batch3"
  checkpoint_dir = output_dir;
  checkpoint_parent = "/Users/garkenyon/workspace/HyPerHLCA";
  checkpoint_children = ...
     {"CIFAR256_RGB_deep_task/test_batch3"}; %% ...
  last_checkpoint_ndx = 2000000;
elseif isunix
  workspace_path = "/home/gkenyon/workspace";
  %%run_type = "noPulvinar"; %%
  %%run_type = "color_deep"; %%
  %%run_type = "noTopDown"; %%
  run_type = "lateral"; %% 
  %%run_type = "deep_plus_noise";%%
  %%run_type = "V1";
  if strcmp(run_type, "color_deep")
    %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_02_01/output_2013_02_01_12x12x128_lambda_05X1_deep"; 
    output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X2_realtime_deep";
    checkpoint_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_2013_01_31_12x12x128_lambda_05X2_noise_05_deep"; %%output_dir; 
    checkpoint_parent = "/nh/compneuro/Data/vine/LCA";
    checkpoint_children = {"2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X4_deep"}; 
%%    checkpoint_children = {"2013_02_01/output_2013_02_01_12x12x128_lambda_05X2_deep"}; 
%%    checkpoint_children = {"2013_01_31/output_2013_01_31_12x12x128_lambda_05X2_deep"; ...
%%			   "2013_01_30/output_2013_01_30_12x12x128_lambda_05X2_deep"; ...
%%			   "2013_01_29/output_2013_01_29_12x12x128_lambda_05X2_deep"; ... 
%%			   "2013_01_28/output_2013_01_28_12x12x128_lambda_05X2_deep"; ...
%%			   "2013_01_27/output_2013_01_27_12x12x128_lambda_05X2_deep"; ...
%%			   "2013_01_26/output_2013_01_26_12x12x128_lambda_05X2_deep"; ...
%%			   "2013_01_25/output_2013_01_25_12x12x128_lambda_05X2_deep"; ...
%%			   "2013_01_24/output_2013_01_24_12x12x128_lambda_05X2_deep"; ...
%%			   "2013_02_01/output_2013_02_01_12x12x128_lambda_05X2_deep"};
    last_checkpoint_ndx = 7000000;
  elseif strcmp(run_type, "noTopDown")
    %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_28/output_2013_01_28_12x12x128_lambda_05X2_noTopDown"; 
    output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X4_noTopDown";
    checkpoint_dir = output_dir;%%"/nh/compneuro/Data/vine/LCA/2013_01_31/output_2013_01_31_12x12x128_lambda_05X2_noTopDown"; %% 
    checkpoint_parent = "/nh/compneuro/Data/vine/LCA";
    checkpoint_children = {"2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X4_noTopDown"};
%%    checkpoint_children = {"2013_01_31/output_2013_01_31_12x12x128_lambda_05X2_noTopDown"; ...
%%			   "2013_01_30/output_2013_01_30_12x12x128_lambda_05X2_noTopDown"; ...
%%			   "2013_01_29/output_2013_01_29_12x12x128_lambda_05X2_noTopDown"}; %%"; ...
%%			   %%"2013_01_28/output_2013_01_28_12x12x128_lambda_05X2_noTopDown"; ...
%%			   %%"2013_01_27/output_2013_01_27_12x12x128_lambda_05X2_noTopDown"; ...
%%			   %%"2013_01_26/output_2013_01_26_12x12x128_lambda_05X2_noTopDown"; ...
%%			   %%"2013_01_25/output_2013_01_25_12x12x128_lambda_05X2_noTopDown"; ...
%%			   %%"2013_01_24/output_2013_01_24_12x12x128_lambda_05X2_noTopDown"; ...
%%			   %%"2013_02_01/output_2013_02_01_12x12x128_lambda_05X2_noTopDown"};
    last_checkpoint_ndx = 2800000;
  elseif strcmp(run_type, "V1")
    output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_2013_01_31_12x12x160_lambda_05X2_V1"; 
    checkpoint_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_2013_01_31_12x12x160_lambda_05X2_V1"; %%output_dir; 
    checkpoint_parent = "/nh/compneuro/Data/vine/LCA";
    checkpoint_children = {"2013_01_31/output_2013_01_31_12x12x160_lambda_05X2_V1"};
  elseif strcmp(run_type, "noPulvinar")
    %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_25/output_2013_01_25_12x12x128_lambda_05X2_noPulvinar"; 
    %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X2_noPulvinar";
    output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X1_noPulvinar";
    checkpoint_dir = output_dir;
    checkpoint_parent = "/nh/compneuro/Data/vine/LCA";
    checkpoint_children = {"2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X1_noPulvinar"}; 
    %%checkpoint_children = {"2013_01_24/output_2013_01_24_12x12x128_lambda_05X2_noPulvinar"}; ...
%%    checkpoint_children = {"2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_lambda_05X4_noPulvinar"}; ...
%%    checkpoint_children = {"2013_01_31/output_2013_01_31_12x12x128_lambda_05X2_noPulvinar"; ...
%%			   "2013_01_30/output_2013_01_30_12x12x128_lambda_05X2_noPulvinar"; ...
%%			   "2013_01_29/output_2013_01_29_12x12x128_lambda_05X2_noPulvinar"; ...
%%			   "2013_01_28/output_2013_01_28_12x12x128_lambda_05X2_noPulvinar"; ...
%%			   "2013_01_27/output_2013_01_27_12x12x128_lambda_05X2_noPulvinar"; ...
%%			   "2013_01_26/output_2013_01_26_12x12x128_lambda_05X2_noPulvinar"; ...
%%			   "2013_01_25/output_2013_01_25_12x12x128_lambda_05X2_noPulvinar"}; %%; ...
%%			   %%"2013_01_24/output_2013_01_24_12x12x128_lambda_05X2_noPulvinar"; ...
%%			   %%"2013_02_01/output_2013_02_01_12x12x128_lambda_05X2_noPulvinar"};
    last_checkpoint_ndx = 2700000;
  elseif strcmp(run_type, "lateral")
    output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_2013_01_31_12x12x128_3x3_9x9x256_lambda_05X1_lateral"; 
    checkpoint_dir =  output_dir;
    checkpoint_parent = "/nh/compneuro/Data/vine/LCA";
    checkpoint_children = {"2013_01_31/output_2013_01_31_12x12x128_3x3_9x9x256_lambda_05X1_lateral"};
  endif
endif %% isunix
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);

%% default paths
if ~exist("output_dir") || isempty(output_dir)
  warning("using default output dir");
  output_dir = pwd
endif
if ~exist("checkpoint_path") || isempty(checkpoint_path)
  chechpoint_path = output_dir;
endif
if ~exist("last_checkpoint_ndx") || isempty(last_checkpoint_ndx)
  last_checkpoint_ndx = 0;  %% if used to grab non-plastic weights, doesn't not have to be current
endif

checkpoint_path = [checkpoint_dir, filesep, "Checkpoints", filesep,  "Checkpoint", num2str(last_checkpoint_ndx, "%i")]; %% "Last"];%%
%%output_dir = checkpoint_path;
use_last_checkpoint_ndx = false; %%true;  %% flag to set whether to use last_checkpoint_ndx in determining the maximum frames index to analyze 
layer_write_step = 200;  %% used to compute maximum frame index to process
weight_write_step = 2000;
plot_DoG_kernel = 0;  %% set to true if DoG filtering used and dewhitening of reconstructions is desired
max_patches = 128;  %% maximum number of weight patches to plot, typically ordered from most to least active if Sparse_flag == true
checkpoint_weights_movie = true; %% make movie of weights over time using list of checkpoint folders getCheckpointList(checkpoint_parent, checkpoint_children)

%% plot Reconstructions
plot_Recon = true;
if plot_Recon
  num_Recon_default = 4; %%196; %%5880; %%
  if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Recon_list = ...
%%	{["a1_"], ["Bipolar"];
%%	 ["a2_"], ["Ganglion"];
%%	 ["a5_"], ["Recon"];
%%	 ["a8_"], ["Recon2"];
%%	 ["a11_"], ["ReconInfra"];
%%	 ["a11_"], ["ReconInfra"]};
    Recon_list = ...
	{["a5_"], ["Recon"]; ...
	 ["a11_"], ["ReconInfra"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    unwhiten_list = zeros(num_Recon_list,1);
    %%unwhiten_list([2,3,5,6]) = 1;
    %% list of layers to use as a normalization reference for unwhitening
    normalize_list = 1:num_Recon_list;
    %%normalize_list(2) = 1;
    %%normalize_list(3) = 1;
    %%normalize_list(5) = 1;
    %%normalize_list(6) = 1;
    %% list of (previous) layers to sum with current layer
    sum_list = cell(num_Recon_list,1);
    sum_list{2} = 1;
  elseif strcmp(run_type, "CIFAR_noTask_deep") || strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR_noTask_deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"], ["Image"];
	 ["a3_"], ["Recon"];
	 ["a6_"], ["Recon2"];
	 ["a9_"], ["ReconInfra"];
	 ["a9_"], ["ReconInfra"];
	 ["a14_"], ["ReconInfra4"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    unwhiten_list = zeros(num_Recon_list,1);
    unwhiten_list([2,3,5,6]) = 1;
    %% list of layers to use as a normalization reference for unwhitening
    normalize_list = 1:num_Recon_list;
    normalize_list(2) = 1;
    normalize_list(4) = 1;
    normalize_list(5) = 1;
    %% list of (previous) layers to sum with current layer
    sum_list = cell(num_Recon_list,1);
    sum_list{5} = 2;
  elseif strcmp(run_type, "noPulvinar")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Recon_list = ...
%%	{["a1_"], ["Bipolar"];
%%	 ["a2_"], ["Ganglion"];
%%	 ["a5_"], ["Recon"];
%%	 ["a9_"], ["ReconInfra"]};
    Recon_list = ...
	{["a5_"], ["Recon"];
	 ["a9_"], ["ReconInfra"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    unwhiten_list = zeros(num_Recon_list,1);
    unwhiten_list([2,3,4]) = 1;
    %% list of layers to use as a normalization reference for unwhitening
    normalize_list = 1:num_Recon_list;
    normalize_list(2) = 1;
    normalize_list(3) = 1;
    normalize_list(4) = 1;
    %% list of (previous) layers to sum with current layer
    sum_list = cell(num_Recon_list,1);
  elseif strcmp(run_type, "V1")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% V1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"], ["Image"];
	 ["a2_"], ["Ganglion"];
	 ["a5_"], ["Recon"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    unwhiten_list = zeros(num_Recon_list,1);
    unwhiten_list([2,3]) = 1;
    %% list of layers to use as a normalization reference for unwhitening
    normalize_list = 1:num_Recon_list;
    normalize_list(2) = 1;
    normalize_list(3) = 1;
    %% list of (previous) layers to sum with current layer
    sum_list = cell(num_Recon_list,1);
  elseif strcmp(run_type, "lateral")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"], ["Image"];
	 ["a2_"], ["Ganglion"];
	 ["a5_"], ["Recon"];
	 ["a8_"], ["Recon2"];
	 ["a11_"], ["ReconInfra"];
	 ["a11_"], ["ReconInfra"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    unwhiten_list = zeros(num_Recon_list,1);
    unwhiten_list([2,3,5,6]) = 1;
    %% list of layers to use as a normalization reference for unwhitening
    normalize_list = 1:num_Recon_list;
    normalize_list(2) = 1;
    normalize_list(4) = 1;
    normalize_list(5) = 1;
    %% list of (previous) layers to sum with current layer
    sum_list = cell(num_Recon_list,1);
    sum_list{6} = 4;
  elseif strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR") || strcmp(run_type, "CIFAR_noTask") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    plot_DoG_kernel = false;
    if strcmp(run_type, "CIFAR_noTask") 
      Recon_list = ...
	  {["a0_"], ["Image"];
	   ["a3_"], ["Recon"];
	   ["a7_"], ["ReconInfra"]};
    else
      Recon_list = ...
	  {["a0_"], ["Image"];
	   ["a5_"], ["Recon"];
	   ["a10_"], ["ReconInfra"]};
    endif
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    sum_list = cell(num_Recon_list,1);
    sum_list{5} = 2;
  endif %% run_type
  num_Recon_frames = repmat(num_Recon_default, 1, num_Recon_list);
  
  %%keyboard;
  Recon_dir = [output_dir, filesep, "Recon"]
  [status, msg, msgid] = mkdir(Recon_dir);
  if status ~= 1
    warning(["mkdir(", Recon_dir, ")", " msg = ", msg]);
  endif 
  %% parse center/surround pre-processing filters
  if plot_DoG_kernel
    if strcmp(run_type, "color_deep") || strcmp(run_type, "lateral") || strcmp(run_type, "noPulvinar") || strcmp(run_type, "noTopDown")
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% deep/lateral/noPulvinar list
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      blur_center_path = [checkpoint_path, filesep, "ImageToBipolarCenter_W.pvp"];
      DoG_center_path = [checkpoint_path, filesep, "BipolarToGanglionCenter_W.pvp"];
      DoG_surround_path = [checkpoint_path, filesep, "BipolarToGanglionSurround_W.pvp"];
    elseif strcmp(run_type, "KITTI")
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% KITTI list
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%    blur_center_path = [checkpoint_path, filesep, "LeftRetinaToLeftBipolarCenter_W.pvp"];
      %%    DoG_center_path = [checkpoint_path, filesep, "LeftBipolarToLeftGanglionCenter_W.pvp"];
      %%    DoG_surround_path = [checkpoint_path, filesep, "LeftBipolarToLeftGanglionSurround_W.pvp"];
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    endif %% run_type
    [blur_weights] = get_Blur_weights(blur_center_path);
    [DoG_weights] = get_DoG_weights(DoG_center_path, DoG_surround_path);
  endif  %% plot_DoG_kernel

  Recon_hdr = cell(num_Recon_list,1);
  Recon_fig = zeros(num_Recon_list,1);
  Recon_fig_name = cell(num_Recon_list,1);
  Recon_vals = cell(num_Recon_list,1);
  Recon_times = cell(num_Recon_list,1);
  if plot_DoG_kernel
    unwhitened_Recon_fig = zeros(num_Recon_list,1);
    unwhitened_Recon_DoG = cell(num_Recon_list,1);
  endif
  tot_Recon_frames = zeros(num_Recon_list,1);
  Recon_mean = zeros(num_Recon_list, 1);
  Recon_std = zeros(num_Recon_list, 1);
  mean_unwhitened_Recon = cell(num_Recon_list,1);
  std_unwhitened_Recon = cell(num_Recon_list, 1);
  max_unwhitened_Recon = cell(num_Recon_list, 1);
  min_unwhitened_Recon = cell(num_Recon_list, 1);
  for i_Recon = 1 : num_Recon_list
    Recon_file = [output_dir, filesep, Recon_list{i_Recon,1}, Recon_list{i_Recon,2}, ".pvp"]
    if ~exist(Recon_file, "file")
      warning(["file does not exist: ", Recon_file]);
      continue;
    endif
    Recon_fid(i_Recon) = fopen(Recon_file);
    Recon_hdr{i_Recon} = readpvpheader(Recon_fid(i_Recon));
    fclose(Recon_fid(i_Recon));
    min_tot_Recon_frames = Recon_hdr{i_Recon}.nbands;
  endfor %% i_Recon
  for i_Recon = 1 : num_Recon_list
    Recon_file = [output_dir, filesep, Recon_list{i_Recon,1}, Recon_list{i_Recon,2}, ".pvp"]
    if ~exist(Recon_file, "file")
      warning(["file does not exist: ", Recon_file]);
      continue;
    endif
    tot_Recon_frames(i_Recon) = Recon_hdr{i_Recon}.nbands;
    progress_step = ceil( tot_Recon_frames(i_Recon)/ 10);
    [Recon_struct, Recon_hdr_tmp] = ...
	readpvpfile(Recon_file, ...
		    progress_step, ...
		    tot_Recon_frames(i_Recon), ... 
		    tot_Recon_frames(i_Recon)-num_Recon_frames(i_Recon)+1); 
    if plot_flag
      Recon_fig(i_Recon) = figure;
    endif
    num_Recon_colors = Recon_hdr{i_Recon}.nf;
    mean_unwhitened_Recon{i_Recon,1} = zeros(num_Recon_colors,num_Recon_frames(i_Recon));
    std_unwhitened_Recon{i_Recon, 1} = ones(num_Recon_colors, num_Recon_frames(i_Recon));
    max_unwhitened_Recon{i_Recon, 1} = ones(num_Recon_colors, num_Recon_frames(i_Recon));
    min_unwhitened_Recon{i_Recon, 1} = zeros(num_Recon_colors,num_Recon_frames(i_Recon));
    Recon_vals{i_Recon} = cell(num_Recon_frames(i_Recon),1);
    Recon_times{i_Recon} = zeros(num_Recon_frames(i_Recon),1);
    if plot_DoG_kernel && unwhiten_list(i_Recon)
      if plot_flag
	unwhitened_Recon_fig(i_Recon) = figure;
      endif
      unwhitened_Recon_DoG{i_Recon} = cell(num_Recon_frames(i_Recon),1);
    endif
    for i_frame = 1 : num_Recon_frames(i_Recon)
      Recon_time{i_Recon}(i_frame) = Recon_struct{i_frame}.time;
      Recon_vals{i_Recon}{i_frame} = Recon_struct{i_frame}.values;
      if plot_flag
	figure(Recon_fig(i_Recon));
      endif
      Recon_fig_name{i_Recon} = Recon_list{i_Recon,2};
      num_sum_list = length(sum_list{i_Recon});
      for i_sum = 1 : num_sum_list
	sum_ndx = sum_list{i_Recon}(i_sum);
	%% if simulation still running, current layer might reflect later times
	j_frame = i_frame;
	while (Recon_time{i_Recon}(i_frame) > Recon_time{sum_ndx}(j_frame))
	  j_frame = j_frame + 1;
	  if j_frame > num_Recon_frames(i_Recon)
	    break;
	  endif
	endwhile
	if j_frame > num_Recon_frames(i_Recon)
	  continue;
	endif
	Recon_vals{i_Recon}{i_frame} = Recon_vals{i_Recon}{i_frame} + ...
	    Recon_vals{sum_ndx}{j_frame};
	Recon_fig_name{i_Recon} = [Recon_fig_name{i_Recon}, "_", Recon_list{sum_ndx,2}];
      endfor %% i_sum
      mean_Recon_tmp = mean(Recon_vals{i_Recon}{i_frame}(:));
      std_Recon_tmp = std(Recon_vals{i_Recon}{i_frame}(:));
      Recon_mean(i_Recon) = Recon_mean(i_Recon) + mean_Recon_tmp;
      Recon_std(i_Recon) = Recon_std(i_Recon) + std_Recon_tmp;
      Recon_fig_name{i_Recon} = [Recon_fig_name{i_Recon}, "_", num2str(Recon_time{i_Recon}(i_frame), "%08d")];
      Recon_vals_tmp = ...
	  permute(Recon_vals{i_Recon}{i_frame},[2,1,3]);
      Recon_vals_tmp = ...
	  (Recon_vals_tmp - min(Recon_vals_tmp(:))) / ...
	  ((max(Recon_vals_tmp(:))-min(Recon_vals_tmp(:))) + ...
	   ((max(Recon_vals_tmp(:))-min(Recon_vals_tmp(:)))==0));
      Recon_vals_tmp = uint8(255*squeeze(Recon_vals_tmp));
      if plot_flag
	set(Recon_fig(i_Recon), "name", Recon_fig_name{i_Recon});
	imagesc(Recon_vals_tmp); 
	if num_Recon_colors == 1
	  colormap(gray); 
	endif
	box off; axis off; axis image;
	saveas(Recon_fig(i_Recon), ...
	       [Recon_dir, filesep, Recon_fig_name{i_Recon}, ".png"], "png");
      else
	imwrite(Recon_vals_tmp, [Recon_dir, filesep, Recon_fig_name{i_Recon}, ".png"], "png");
      endif
      for i_color = 1 : num_Recon_colors
	tmp_Recon = ...
	    squeeze(Recon_vals{i_Recon}{i_frame}(:,:,i_color));
	mean_unwhitened_Recon{i_Recon}(i_color, i_frame) = mean(tmp_Recon(:));
 	std_unwhitened_Recon{i_Recon}(i_color, i_frame) = std(tmp_Recon(:));
	max_unwhitened_Recon{i_Recon}(i_color, i_frame) = max(tmp_Recon(:));
	min_unwhitened_Recon{i_Recon}(i_color, i_frame) = min(tmp_Recon(:));
      endfor
      if plot_DoG_kernel && unwhiten_list(i_Recon)
	%%keyboard;
	unwhitened_Recon_DoG{i_Recon}{i_frame} = zeros(size(permute(Recon_vals{i_Recon}{i_frame},[2,1,3])));
	for i_color = 1 : num_Recon_colors
	  tmp_Recon = ...
	      deconvolvemirrorbc(squeeze(Recon_vals{i_Recon}{i_frame}(:,:,i_color))', DoG_weights);
	  mean_unwhitened_Recon{i_Recon}(i_color, i_frame) = mean(tmp_Recon(:));
 	  std_unwhitened_Recon{i_Recon}(i_color, i_frame) = std(tmp_Recon(:));
	  max_unwhitened_Recon{i_Recon}(i_color, i_frame) = max(tmp_Recon(:));
	  min_unwhitened_Recon{i_Recon}(i_color, i_frame) = min(tmp_Recon(:));
	  j_frame = i_frame;
	  while (Recon_time{i_Recon}(i_frame) > Recon_time{normalize_list(i_Recon)}(j_frame))
	    j_frame = j_frame + 1;
	    if j_frame > num_Recon_frames(i_Recon)
	      j_frame = i_frame;
	      break;
	    endif
	  endwhile
	  if j_frame > num_Recon_frames(i_Recon)
	    j_frame = i_frame;
	  endif
	  if (Recon_time{i_Recon}(i_frame) ~= Recon_time{normalize_list(i_Recon)}(j_frame))
	    warning(["i_Recon = ", num2str(i_Recon), ", i_frame = ", num2str(i_frame), ...
		     ", Recon_time{i_Recon}(i_frame) = ", ...
		     num2str(Recon_time{i_Recon}(i_frame)), ...
		     " ~= ", ...
		     "normalize_list(i_Recon) = ", num2str(normalize_list(i_Recon)), ", j_frame = ", num2str(j_frame), ...
		     ", Recon_time{normalize_list(i_Recon)}(j_frame) = ", ...
		     num2str(Recon_time{normalize_list(i_Recon)}(j_frame))]);
	  endif
	  %%j_frame = ceil(i_frame * tot_Recon_frames(normalize_list(i_Recon)) / tot_Recon_frames(i_Recon));
	  if i_Recon ~= normalize_list(i_Recon)
	    tmp_Recon = ...
		(tmp_Recon - mean_unwhitened_Recon{i_Recon}(i_color, j_frame)) * ...
		(std_unwhitened_Recon{normalize_list(i_Recon)}(i_color, j_frame) / ...
		 (std_unwhitened_Recon{i_Recon}(i_color, j_frame) + (std_unwhitened_Recon{i_Recon}(i_color, j_frame)==0))) + ...
		mean_unwhitened_Recon{normalize_list(i_Recon)}(i_color, j_frame); 
	  endif
	  unwhitened_Recon_DoG{i_Recon}{i_frame}(:,:,i_color) = tmp_Recon;
	endfor
	if plot_flag
	  figure(unwhitened_Recon_fig(i_Recon));
	endif
	for i_sum = 1 : num_sum_list
	  sum_ndx = sum_list{i_Recon}(i_sum);
	  unwhitened_Recon_DoG{i_Recon}{i_frame} = unwhitened_Recon_DoG{i_Recon}{i_frame} + ...
	      unwhitened_Recon_DoG{sum_ndx}{i_frame};
	endfor %% i_sum
	unwhitened_Recon_DoG_tmp = ...
	    permute(unwhitened_Recon_DoG{i_Recon}{i_frame},[2,1,3]);
	unwhitened_Recon_DoG_tmp = ...
	    (unwhitened_Recon_DoG_tmp - min(unwhitened_Recon_DoG_tmp(:))) / ...
	    ((max(unwhitened_Recon_DoG_tmp(:))-min(unwhitened_Recon_DoG_tmp(:))) + ...
	     ((max(unwhitened_Recon_DoG_tmp(:))-min(unwhitened_Recon_DoG_tmp(:)))==0));
	unwhitened_Recon_DoG_tmp = uint8(255*squeeze(unwhitened_Recon_DoG_tmp));
	if plot_flag
	  set(unwhitened_Recon_fig(i_Recon), "name", ["unwhitened_", Recon_fig_name{i_Recon}]); 
	  imagesc(unwhitened_Recon_DoG_tmp); 
	  if num_Recon_colors == 1
	    colormap(gray); 
	  endif
	  box off; axis off; axis image;
	  saveas(unwhitened_Recon_fig(i_Recon), ...
		 [Recon_dir, filesep, "unwhitened_", Recon_fig_name{i_Recon}, ".png"], "png");
	  drawnow
	else
	  imwrite(unwhitened_Recon_DoG_tmp, [Recon_dir, filesep, "unwhitened_", Recon_fig_name{i_Recon}, ".png"], "png");
	endif
      endif %% plot_DoG_kernel
    endfor   %% i_frame
    Recon_mean(i_Recon) = Recon_mean(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    Recon_std(i_Recon) = Recon_std(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    disp([ Recon_fig_name{i_Recon}, "_Recon_mean = ", num2str(Recon_mean(i_Recon)), " +/- ", num2str(Recon_std(i_Recon))]);
    
  endfor %% i_Recon
endif %% plot_Recon







%%keyboard;
plot_StatsProbe_vs_time = false;
if plot_StatsProbe_vs_time && plot_flag
  StatsProbe_plot_lines = 20000;
  if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_list = ...
        {["Error"],["_Stats.txt"]; ...
         ["V1"],["_Stats.txt"];
         ["Error2"],["_Stats.txt"]; ...
         ["V2"],["_Stats.txt"];
         ["Error1_2"],["_Stats.txt"]; ...
         ["V1Infra"],["_Stats.txt"]};
  elseif strcmp(run_type, "noPulvinar")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_list = ...
        {["Error"],["_Stats.txt"]; ...
         ["V1"],["_Stats.txt"];
         ["V2"],["_Stats.txt"];
         ["Error1_2"],["_Stats.txt"]; ...
         ["V1Infra"],["_Stats.txt"]};
  elseif strcmp(run_type, "V1")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% V1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_list = ...
        {["Error"],["_Stats.txt"]; ...
         ["V1"],["_Stats.txt"]};
  elseif strcmp(run_type, "lateral")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_list = ...
	{["Error"],["_Stats.txt"]; ...
	 ["V1"],["_Stats.txt"];
	 ["Error2"],["_Stats.txt"]; ...
	 ["V2"],["_Stats.txt"];
	 ["Error1_2"],["_Stats.txt"]; ...
	 ["V1Infra"],["_Stats.txt"]; ...
	 ["V1Intra"],["_Stats.txt"]};
  elseif strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR") || strcmp(run_type, "CIFAR_noTask") || strcmp(run_type, "CIFAR_deep")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_list = ...
	{["LabelError"],["_Stats.txt"]; ...
	 ["Error"],["_Stats.txt"]; ...
	 ["Error2"],["_Stats.txt"]; ...
	 ["Error1_2"],["_Stats.txt"]; ...
	 ["Error2_4"],["_Stats.txt"]; ...
	 ["V1"],["_Stats.txt"];...
	 ["V2"],["_Stats.txt"];...
	 ["V4"],["_Stats.txt"]};
  elseif strcmp(run_type, "KITTI")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% KITTI list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%  StatsProbe_list = ...
    %%      {["LeftError"],["_Stats.txt"]; ...
    %%       ["RightError"],["_Stats.txt"]; ...
    %%       ["BinocularV1"],["_Stats.txt"]};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  StatsProbe_vs_time_dir = [output_dir, filesep, "StatsProbe_vs_time"]
  [status, msg, msgid] = mkdir(StatsProbe_vs_time_dir);
  if status ~= 1
    warning(["mkdir(", StatsProbe_vs_time_dir, ")", " msg = ", msg]);
  endif 
  num_StatsProbe_list = size(StatsProbe_list,1);

  StatsProbe_sigma_flag = ones(1,num_StatsProbe_list);
  if strcmp(run_type, "color_deep")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([2,4,6]) = 0;
  elseif strcmp(run_type, "noPulvinar")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([2,3,5]) = 0;
  elseif strcmp(run_type, "V1")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% V1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([2]) = 0;
  elseif strcmp(run_type, "lateral")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([2,4,6,7]) = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR") || strcmp(run_type, "CIFAR_noTask")  || strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([6]) = 0;
    StatsProbe_sigma_flag([7]) = 0;
    StatsProbe_sigma_flag([8]) = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  StatsProbe_nnz_flag = ~StatsProbe_sigma_flag;
  if use_last_checkpoint_ndx
    max_StatsProbe_line = last_checkpoint_ndx;
  else
    max_StatsProbe_line = 100000000000000;
  endif
  for i_StatsProbe = 1 : num_StatsProbe_list
    StatsProbe_file = [output_dir, filesep, StatsProbe_list{i_StatsProbe,1}, StatsProbe_list{i_StatsProbe,2}]
    if ~exist(StatsProbe_file,"file")
      warning(["StatsProbe_file does not exist: ", StatsProbe_file]);
      continue;
    endif
    [status, wc_output] = system(["cat ",StatsProbe_file," | wc"], true, "sync");
    if status ~= 0
      error(["system call to compute num lines failed in file: ", StatsProbe_file, " with status: ", num2str(status)]);
    endif
    wc_array = strsplit(wc_output, " ", true);
    StatsProbe_num_lines = str2num(wc_array{1});
    StatsProbe_fid = fopen(StatsProbe_file, "r");
    StatsProbe_line = fgets(StatsProbe_fid);
    StatsProbe_sigma_vals = [];
    StatsProbe_nnz_vals = [];
    last_StatsProbe_line = StatsProbe_num_lines - 2;
    last_StatsProbe_line = min(last_StatsProbe_line, max_StatsProbe_line);
    first_StatsProbe_line = max([(last_StatsProbe_line - StatsProbe_plot_lines), 1]);
    StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    for i_line = 1:first_StatsProbe_line-1
      StatsProbe_line = fgets(StatsProbe_fid);
    endfor
    %% extract N
    StatsProbe_N_ndx1 = strfind(StatsProbe_line, "N==");
    StatsProbe_N_ndx2 = strfind(StatsProbe_line, "Total==");
    StatsProbe_N_str = StatsProbe_line(StatsProbe_N_ndx1+3:StatsProbe_N_ndx2-2);
    StatsProbe_N = str2num(StatsProbe_N_str);
    for i_line = first_StatsProbe_line:last_StatsProbe_line
      StatsProbe_line = fgets(StatsProbe_fid);
      %% extract time
      StatsProbe_time_ndx1 = strfind(StatsProbe_line, "t==");
      StatsProbe_time_ndx2 = strfind(StatsProbe_line, "N==");
      StatsProbe_time_str = StatsProbe_line(StatsProbe_time_ndx1+3:StatsProbe_time_ndx2-2);
      StatsProbe_time_vals(i_line-first_StatsProbe_line+1) = str2num(StatsProbe_time_str);
      %% extract sigma
      StatsProbe_sigma_ndx1 = strfind(StatsProbe_line, "sigma==");
      StatsProbe_sigma_ndx2 = strfind(StatsProbe_line, "nnz==");
      StatsProbe_sigma_str = StatsProbe_line(StatsProbe_sigma_ndx1+7:StatsProbe_sigma_ndx2-2);
      StatsProbe_sigma_vals(i_line-first_StatsProbe_line+1) = str2num(StatsProbe_sigma_str);
      %% extract nnz
      StatsProbe_nnz_ndx1 = strfind(StatsProbe_line, "nnz==");
      StatsProbe_nnz_ndx2 = length(StatsProbe_line); 
      StatsProbe_nnz_str = StatsProbe_line(StatsProbe_nnz_ndx1+5:StatsProbe_nnz_ndx2-1);
      StatsProbe_nnz_vals(i_line-first_StatsProbe_line+1) = str2num(StatsProbe_nnz_str);
    endfor %%i_line
    fclose(StatsProbe_fid);
    if plot_flag
      StatsProbe_vs_time_fig(i_StatsProbe) = figure;
    endif
    if StatsProbe_nnz_flag(i_StatsProbe)
      if plot_flag
	StatsProbe_vs_time_hndl = plot(StatsProbe_time_vals, StatsProbe_nnz_vals/StatsProbe_N);
	axis tight
	set(StatsProbe_vs_time_fig(i_StatsProbe), "name", [StatsProbe_list{i_StatsProbe,1}, " nnz"]);
	saveas(StatsProbe_vs_time_fig(i_StatsProbe), ...
	       [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
		"_nnz_vs_time_", num2str(StatsProbe_time_vals(end), "%08d")], "png");
	else
	  %% don't know how to imwrite a scatter plot
	endif
    else
      if plot_flag
	StatsProbe_vs_time_hndl = plot(StatsProbe_time_vals, StatsProbe_sigma_vals); axis tight;
	axis tight
	set(StatsProbe_vs_time_fig(i_StatsProbe), "name", [StatsProbe_list{i_StatsProbe,1}, " sigma"]);
	saveas(StatsProbe_vs_time_fig(i_StatsProbe), ...
	       [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
		"_sigma_vs_time_", num2str(StatsProbe_time_vals(end), "%08d")], "png");
      else
	%% don't know how to imwrite a scatter plot
      endif
      save("-mat", ...
	   [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
	    "_sigma_vs_time_", num2str(StatsProbe_time_vals(end), "%08d"), ".mat"], ...
	   "StatsProbe_time_vals", "StatsProbe_sigma_vals");
    endif %% 
    drawnow;
  endfor %% i_StatsProbe
endif  %% plot_StatsProbe_vs_time




plot_Sparse = true;
if plot_Sparse
  if strcmp(run_type, "color_deep") || strcmp(run_type, "lateral") || strcmp(run_type, "noTopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep/lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a4_"], ["V1"]; ...
	 ["a7_"], ["V2"]};
  elseif strcmp(run_type, "CIFAR_noTask_deep")  || strcmp(run_type, "CIFAR_deep")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR_noTask_deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a2_"], ["V1"]; ...
	 ["a5_"], ["V2"]; ...
	 ["a10_"], ["V4"]};
  elseif strcmp(run_type, "noPulvinar")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a4_"], ["V1"]; ...
	 ["a6_"], ["V2"]};
  elseif strcmp(run_type, "V1")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% V1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a4_"], ["V1"]};
  elseif strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR") || strcmp(run_type, "CIFAR_noTask")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a2_"], ["V1"]; ...
	 ["a4_"], ["V2"]};
  endif %% run_type
  num_Sparse_list = size(Sparse_list,1);
  Sparse_hdr = cell(num_Sparse_list,1);
  Sparse_hist_rank_array = cell(num_Sparse_list,1);
  Sparse_times_array = cell(num_Sparse_list,1);
  Sparse_percent_active_array = cell(num_Sparse_list,1);
  Sparse_percent_change_array = cell(num_Sparse_list,1);
  Sparse_std_array = cell(num_Sparse_list,1);
  Sparse_dir = [output_dir, filesep, "Sparse"]
  [status, msg, msgid] = mkdir(Sparse_dir);
  if status ~= 1
    warning(["mkdir(", Sparse_dir, ")", " msg = ", msg]);
  endif 
  for i_Sparse = 1 : num_Sparse_list
    Sparse_file = [output_dir, filesep, Sparse_list{i_Sparse,1}, Sparse_list{i_Sparse,2}, ".pvp"]
    if ~exist(Sparse_file, "file")
      warning(["file does not exist: ", Sparse_file]);
    endif
    Sparse_fid = fopen(Sparse_file);
    Sparse_hdr{i_Sparse} = readpvpheader(Sparse_fid);
    fclose(Sparse_fid);
    tot_Sparse_frames = Sparse_hdr{i_Sparse}.nbands;
    %% if using checkpoint files for ploting weights, then chose the last activity time to match the last checkpoint time
    if use_last_checkpoint_ndx
      tot_Sparse_frames = min(tot_Sparse_frames, fix(last_checkpoint_ndx / layer_write_step));  %% use to specify maximum frame to display
    endif
    num_Sparse = fix(tot_Sparse_frames/1);  %% number of activity frames to analyze, counting backward from last frame, maximum is tot_Sparse_frames
    progress_step = ceil(tot_Sparse_frames / 10);
    if ~load_flag
      [Sparse_struct, Sparse_hdr_tmp] = ...
	  readpvpfile(Sparse_file, progress_step, tot_Sparse_frames, tot_Sparse_frames-num_Sparse+5,1);
    else %% just read last frame
      [Sparse_struct, Sparse_hdr_tmp] = ...
	  readpvpfile(Sparse_file, progress_step, tot_Sparse_frames, tot_Sparse_frames,1); 
    endif
    nx_Sparse = Sparse_hdr{i_Sparse}.nx;
    ny_Sparse = Sparse_hdr{i_Sparse}.ny;
    nf_Sparse = Sparse_hdr{i_Sparse}.nf;
    n_Sparse = nx_Sparse * ny_Sparse * nf_Sparse;
    num_Sparse_frames = size(Sparse_struct,1);
    Sparse_hist = zeros(nf_Sparse+1,1);
    Sparse_hist_edges = [0:1:nf_Sparse]+0.5;
    Sparse_current = zeros(n_Sparse,1);
    Sparse_abs_change = zeros(num_Sparse_frames,1);
    Sparse_percent_change = zeros(num_Sparse_frames,1);
    Sparse_std = ones(num_Sparse_frames,1);
    Sparse_current_active = 0;
    Sparse_tot_active = zeros(num_Sparse_frames,1);
    Sparse_times = zeros(num_Sparse_frames,1);
    for i_frame = 1 : 1 : num_Sparse_frames
      Sparse_times(i_frame) = squeeze(Sparse_struct{i_frame}.time);
      Sparse_values_tmp = squeeze(Sparse_struct{i_frame}.values);
      Sparse_active_ndx = Sparse_values_tmp(:,1);
      Sparse_previous = Sparse_current;
      if columns(Sparse_values_tmp) == 2
	Sparse_active_vals = Sparse_values_tmp(:,2);
      else
	Sparse_active_vals = ones(size(Sparse_active_ndx));
      endif
      Sparse_current = full(sparse(Sparse_active_ndx+1,1,Sparse_active_vals,n_Sparse,1,n_Sparse));
      Sparse_abs_change(i_frame) = sum((Sparse_current(:)~=0) ~= (Sparse_previous(:)~=0));
      Sparse_previous_active = Sparse_current_active;
      Sparse_current_active = nnz(Sparse_current(:));
      Sparse_tot_active(i_frame) = Sparse_current_active;
      Sparse_OR_active = sum((Sparse_current(:)~=0) | (Sparse_previous(:)~=0));
      Sparse_percent_change(i_frame) = ...
	  Sparse_abs_change(i_frame) / (Sparse_OR_active + (Sparse_OR_active==0));
      Sparse_active_kf = mod(Sparse_active_ndx, nf_Sparse) + 1;
      if Sparse_current_active > 0
	Sparse_hist_frame = histc(Sparse_active_kf, Sparse_hist_edges);
      else
	Sparse_hist_frame = zeros(nf_Sparse+1,1);
      endif
      Sparse_hist = Sparse_hist + Sparse_hist_frame;
      Sparse_std(i_frame) = std(Sparse_current);
    endfor %% i_frame
    Sparse_percent_active = Sparse_tot_active/n_Sparse;
    if ~load_flag
      Sparse_hist = Sparse_hist(1:nf_Sparse);
      Sparse_hist = Sparse_hist / (num_Sparse_frames * nx_Sparse * ny_Sparse); 
      [Sparse_hist_sorted, Sparse_hist_rank] = sort(Sparse_hist, 1, "descend");
      Sparse_hist_bins = 1:nf_Sparse;
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_hist_bins_", Sparse_list{i_Sparse,2}, "_", ...
	    num2str(Sparse_times(num_Sparse_frames), "%08d"), ".mat"], ...
	   "Sparse_hist_bins");
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_hist_sorted_", Sparse_list{i_Sparse,2}, "_", ...
	    num2str(Sparse_times(num_Sparse_frames), "%08d"), ".mat"], ...
	   "Sparse_hist_sorted");
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_hist_rank_", Sparse_list{i_Sparse,2}, "_", ...
	    num2str(Sparse_times(num_Sparse_frames), "%08d"), ".mat"], ...
	   "Sparse_hist_rank");
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_percent_change_", Sparse_list{i_Sparse,2}, "_", ...
	    num2str(Sparse_times(num_Sparse_frames), "%08d"), ".mat"], ...
	   "Sparse_times", "Sparse_percent_change");    
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_percent_active_", Sparse_list{i_Sparse,2}, "_", ...
	    num2str(Sparse_times(num_Sparse_frames), "%08d"), ".mat"], ...
	   "Sparse_times", "Sparse_percent_active");	 
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_std_", Sparse_list{i_Sparse,2}, "_", ...
	    num2str(Sparse_times(num_Sparse_frames), "%08d"), ".mat"], ...
	   "Sparse_times", "Sparse_std");	 
    else
      Sparse_hist_bins_str = ...
	  [Sparse_dir, filesep, "Sparse_hist_bins_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_hist_bins_glob = glob(Sparse_hist_bins_str);
      num_Sparse_hist_bins_glob = length(Sparse_hist_bins_glob);
      if num_Sparse_hist_bins_glob <= 0
	warning(["load_flag is true but no files to load in: ", Sparse_hist_bins_str]);
	break;
      endif
      load("-mat", Sparse_hist_bins_glob{num_Sparse_hist_bins_glob});
      Sparse_hist_sorted_str = ...
	  [Sparse_dir, filesep, "Sparse_hist_sorted_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_hist_sorted_glob = glob(Sparse_hist_sorted_str);
      num_Sparse_hist_sorted_glob = length(Sparse_hist_sorted_glob);
      if num_Sparse_hist_sorted_glob <= 0
	warning(["load_flag is true but no files to load in: ", Sparse_hist_sorted_str]);
	break;
      endif
      load("-mat", Sparse_hist_sorted_glob{num_Sparse_hist_sorted_glob});
      Sparse_hist_rank_str = ...
	  [Sparse_dir, filesep, "Sparse_hist_rank_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_hist_rank_glob = glob(Sparse_hist_rank_str);
      num_Sparse_hist_rank_glob = length(Sparse_hist_rank_glob);
      if num_Sparse_hist_rank_glob <= 0
	warning(["load_flag is true but no files to load in: ", Sparse_hist_rank_str]);
	break;
      endif
      load("-mat", Sparse_hist_rank_glob{num_Sparse_hist_rank_glob});
      Sparse_percent_change_str = ...
	  [Sparse_dir, filesep, "Sparse_percent_change_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_percent_change_glob = glob(Sparse_percent_change_str);
      num_Sparse_percent_change_glob = length(Sparse_percent_change_glob);
      if num_Sparse_percent_change_glob <= 0
	warning(["load_flag is true but no files to load in: ", Sparse_percent_change_str]);
	break;
      endif
      load("-mat", Sparse_percent_change_glob{num_Sparse_percent_change_glob});
      Sparse_percent_active_str = ...
	  [Sparse_dir, filesep, "Sparse_percent_active_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_percent_active_glob = glob(Sparse_percent_active_str);
      num_Sparse_percent_active_glob = length(Sparse_percent_active_glob);
      if num_Sparse_percent_active_glob <= 0
	warning(["load_flag is true but no files to load in: ", Sparse_percent_active_str]);
	break;
      endif
      load("-mat", Sparse_percent_active_glob{num_Sparse_percent_active_glob});
      Sparse_std_str = ...
	  [Sparse_dir, filesep, "Sparse_std_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_std_glob = glob(Sparse_std_str);
      num_Sparse_std_glob = length(Sparse_std_glob);
      if num_Sparse_std_glob <= 0
	warning(["load_flag is true but no files to load in: ", Sparse_std_str]);
	break;
      endif
      load("-mat", Sparse_std_glob{num_Sparse_std_glob});
    endif %% load_flag
    num_Sparse_frames = length(Sparse_times);
    if plot_flag %%&& ~load_flag
      Sparse_hist_fig = figure;
      Sparse_hist_hndl = bar(Sparse_hist_bins, Sparse_hist_sorted); axis tight;
      set(Sparse_hist_fig, "name", ["Hist_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_Sparse_frames), "%i")]);
      saveas(Sparse_hist_fig, ...
	     [Sparse_dir, filesep, ...
	      "Hist_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_Sparse_frames), "%i")], "png");
      Sparse_percent_change_fig = figure;
      Sparse_percent_change_hndl = plot(Sparse_times, Sparse_percent_change); 
      set(Sparse_percent_change_hndl, "linewidth", 1.5);
      axis([Sparse_times(1) Sparse_times(end) 0 1]); %%axis tight;
      set(Sparse_percent_change_fig, ...
	  "name", ["percent_change_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_Sparse_frames), "%08d")]);
      saveas(Sparse_percent_change_fig, ...
	     [Sparse_dir, filesep, ...
	      "percent_change_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_Sparse_frames), "%08d")], "png");
      
      Sparse_percent_active_fig = figure;
      Sparse_percent_active_hndl = plot(Sparse_times, Sparse_percent_active); axis tight;
      set(Sparse_percent_active_hndl, "linewidth", 1.5);
      set(Sparse_percent_active_fig, "name", ["percent_active_", Sparse_list{i_Sparse,2}, "_", ...
					  num2str(Sparse_times(num_Sparse_frames), "%08d")]);
      saveas(Sparse_percent_active_fig, ...
	     [Sparse_dir, filesep, "percent_active_", Sparse_list{i_Sparse,2}, "_", ...
	      num2str(Sparse_times(num_Sparse_frames), "%08d")], "png");
    endif

    Sparse_median_active = median(Sparse_percent_active(:));
    disp([Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_Sparse_frames), "%i"), ...
	  " median_active = ", num2str(Sparse_median_active)]);
    
    Sparse_mean_percent_change = mean(Sparse_percent_change(:));
    disp([Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_Sparse_frames), "%i"), ...
	  " mean_percent_change = ", num2str(Sparse_mean_percent_change)]);
    Sparse_hist_rank_array{i_Sparse} = Sparse_hist_rank;
    Sparse_times_array{i_Sparse} = Sparse_times;
    Sparse_percent_active_array{i_Sparse} = Sparse_percent_active;
    Sparse_percent_change_array{i_Sparse} = Sparse_percent_change;
    Sparse_std_array{i_Sparse} = Sparse_std;

  endfor  %% i_Sparse
endif %% plot_Sparse











plot_nonSparse = true;
if plot_nonSparse && plot_flag
  if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
        {["a3_"], ["Error"]; ...
         ["a6_"], ["Error2"]; ...
         ["a9_"], ["Error1_2"]; ...
	 ["a15_"], ["NoiselessError"]};
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_skip(1) = 1;
    nonSparse_skip(2) = 1;
    nonSparse_skip(3) = 1;
    nonSparse_skip(4) = 1;
    nonSparse_norm_list = ...
        {["a2_"], ["Ganglion"]; ...
         ["a8_"], ["Recon2"]; ...
         ["a10_"], ["V1Infra"]; ...
         ["a14_"], ["NoiselessGanglion"]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "CIFAR_noTask_deep") || strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR_noTask_deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
        {["a1_"], ["Error"]; ...
         ["a4_"], ["Error2"]; ...
         ["a7_"], ["Error1_2"]; ...
         ["a11_"], ["Error2_4"]; ...
	 ["a16_"], ["LabelError"]};
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_skip(1) = 1;
    nonSparse_skip(2) = 1;
    nonSparse_skip(3) = 1;
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a6_"], ["Recon2"]; ...
         ["a2_"], ["V1"]; ...
         ["a5_"], ["V2"]; ...
         ["a15_"], ["Labels"]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = ...
	1/sqrt(32*32);
    Sparse_std_ndx = [0 0 1 2 0];
  elseif strcmp(run_type, "noPulvinar")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
        {["a3_"], ["Error"]; ...
         ["a7_"], ["Error1_2"]; ...
	 ["a13_"], ["NoiselessError"]};
%%    nonSparse_list = ...
%%        {["a3_"], ["Error"]};
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_skip(1) = 1;
    nonSparse_skip(2) = 1;
    nonSparse_skip(3) = 1;
    nonSparse_norm_list = ...
        {["a2_"], ["Ganglion"]; ...
         ["a8_"], ["V1Infra"]; ...
         ["a12_"], ["NoiselessGanglion"]};
%%    nonSparse_norm_list = ...
%%        {[], []};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "V1")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% V1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
        {["a3_"], ["Error"]};
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_skip(1) = 1;
  elseif strcmp(run_type, "lateral")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
        {["a3_"], ["Error"]; ...
         ["a6_"], ["Error2"]; ...
         ["a10_"], ["Error1_2"]};
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_skip(1) = 1;
    nonSparse_skip(2) = 1;
    nonSparse_skip(3) = 1;
    nonSparse_norm_list = ...
        {["a2_"], ["Ganglion"]; ...
         ["a8_"], ["Recon2"]; ...
         ["a12_"], ["V1Infra"]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a2_"], ["Error"]; ...
	 ["a3_"], ["LabelError"]; ...
	 ["a8_"], ["Error1_2"]};
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_skip(1) = 1;
    nonSparse_skip(2) = 1;
    nonSparse_skip(3) = 1;
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]; ...
	 ["a1_"], ["Labels"]
         ["a9_"], ["V1Infra"]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength = ...
	[1/sqrt(32*32); 1.0; 1.0];
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "CIFAR_noTask")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST_noTask list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a1_"], ["Error"]; ...
	 ["a5_"], ["Error1_2"]};
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_skip(1) = 1;
    nonSparse_skip(2) = 1;
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a6_"], ["V1Infra"]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength = ...
	[1/sqrt(32*32); 1.0];
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  if ~exist("Sparse_std_ndx")
    Sparse_std_ndx = zeros(num_nonSparse_list,1);
  endif

  nonSparse_times_array = cell(num_nonSparse_list,1);
  nonSparse_RMS_array = cell(num_nonSparse_list,1);
  nonSparse_norm_RMS_array = cell(num_nonSparse_list,1);
  nonSparse_hdr = cell(num_nonSparse_list,1);
  nonSparse_dir = [output_dir, filesep, "nonSparse"]
  [status, msg, msgid] = mkdir(nonSparse_dir);
  if status ~= 1
    warning(["mkdir(", nonSparse_dir, ")", " msg = ", msg]);
  endif 
  nonSparse_RMS_fig = zeros(num_nonSparse_list,1);
  max_nonSparse_RMS = zeros(num_nonSparse_list,1);
  min_nonSparse_RMS = zeros(num_nonSparse_list,1);
  for i_nonSparse = 1 : num_nonSparse_list
    nonSparse_file = ...
	[output_dir, filesep, nonSparse_list{i_nonSparse,1}, nonSparse_list{i_nonSparse,2}, ".pvp"]
    if ~exist(nonSparse_file, "file")
      warning(["file does not exist: ", nonSparse_file]);
      continue;
    endif
    nonSparse_fid = fopen(nonSparse_file);
    nonSparse_hdr{i_nonSparse} = readpvpheader(nonSparse_fid);
    fclose(nonSparse_fid);
    tot_nonSparse_frames = nonSparse_hdr{i_nonSparse}.nbands;
    num_nonSparse = tot_nonSparse_frames;
    progress_step = ceil(tot_nonSparse_frames / 10);
    [nonSparse_struct, nonSparse_hdr_tmp] = ...
	readpvpfile(nonSparse_file, progress_step, ...
		    tot_nonSparse_frames, tot_nonSparse_frames-num_nonSparse+5, ...
		    nonSparse_skip(i_nonSparse));
    num_nonSparse_frames = size(nonSparse_struct,1);
    nonSparse_times = zeros(num_nonSparse_frames,1);
    nonSparse_RMS = zeros(num_nonSparse_frames,1);

    num_nonSparse_norm_frames = 0;
    if ~isempty(nonSparse_norm_list{i_nonSparse,1}) && ~isempty(nonSparse_norm_list{i_nonSparse,2})
      nonSparse_norm_file = ...
	  [output_dir, filesep, nonSparse_norm_list{i_nonSparse,1}, ...
	   nonSparse_norm_list{i_nonSparse,2}, ".pvp"]
      if ~exist(nonSparse_norm_file, "file")
	warning(["file does not exist: ", nonSparse_norm_file]);
	continue;
      endif
      progress_step = ceil(tot_nonSparse_frames / 10);
      [nonSparse_norm_struct, nonSparse_norm_hdr_tmp] = ...
	  readpvpfile(nonSparse_norm_file, ...
		      progress_step, tot_nonSparse_frames, tot_nonSparse_frames-num_nonSparse+5, ...
		      nonSparse_skip(i_nonSparse));
      num_nonSparse_norm_frames = size(nonSparse_norm_struct,1);
      nonSparse_norm_times = zeros(num_nonSparse_frames,1);
      nonSparse_norm_RMS = zeros(num_nonSparse_frames,1);
    elseif Sparse_std_ndx(i_nonSparse) > 0
      num_nonSparse_norm_frames = length(Sparse_times_array{Sparse_std_ndx(i_nonSparse)});
      nonSparse_norm_struct = cell(num_nonSparse_norm_frames,1);
      for i_frame = 1 : 1 : num_nonSparse_norm_frames
	nonSparse_norm_struct{i_frame}.time = Sparse_times_array{Sparse_std_ndx(i_nonSparse)}(i_frame);
      endfor
    else
      nonSparse_norm_RMS = ones(num_nonSparse_frames,1);
      nonSparse_norm_times = zeros(num_nonSparse_frames,1);
      nonSparse_norm_struct = [];
    endif %% ~isempty(nonSparse_norm_list{i_nonSparse,1})

    frame_diff = 0;
    if num_nonSparse_norm_frames > 0
      frame_diff = ...
	  max((nonSparse_struct{num_nonSparse_frames}.time - ...
	       nonSparse_norm_struct{num_nonSparse_norm_frames}.time), ...
	      1);
    endif
    for i_frame = 1 : 1 : num_nonSparse_frames
      if ~isempty(nonSparse_struct{i_frame})
	nonSparse_times(i_frame) = squeeze(nonSparse_struct{i_frame}.time);
	nonSparse_vals = squeeze(nonSparse_struct{i_frame}.values);
	nonSparse_RMS(i_frame) = std(nonSparse_vals(:));

	if ~isempty(nonSparse_norm_struct)
	  nonSparse_norm_time = nonSparse_norm_struct{i_frame}.time;
	  nonSparse_time_shift = 0;
	  while (nonSparse_norm_time - nonSparse_times(i_frame)) > frame_diff && ...
		(i_frame-(nonSparse_time_shift+1)) >= 1
	    nonSparse_time_shift = nonSparse_time_shift + 1;
	    nonSparse_norm_time = nonSparse_norm_struct{i_frame-nonSparse_time_shift}.time;
	  endwhile
	  while (nonSparse_norm_time - nonSparse_times(i_frame)) < -frame_diff && ...
		(i_frame-(nonSparse_time_shift-1)) <= num_nonSparse_norm_frames
	    nonSparse_time_shift = nonSparse_time_shift - 1;
	    nonSparse_norm_time = nonSparse_norm_struct{i_frame-nonSparse_time_shift}.time;
	  endwhile
	  if (i_frame-nonSparse_time_shift) < 1 || (i_frame-nonSparse_time_shift) > num_nonSparse_norm_frames
	    last_frame = i_frame - 1;
	    num_nonSparse_frames = i_frame - 1;
	    nonSparse_times = nonSparse_times(1:num_nonSparse_frames);
	    nonSparse_RMS = nonSparse_RMS(1:num_nonSparse_frames);
	    nonSparse_norm_RMS = nonSparse_norm_RMS(1:num_nonSparse_frames);
	    break;
	  else
	    if Sparse_std_ndx(i_nonSparse) == 0
	      nonSparse_norm_vals = ...
		  squeeze(nonSparse_norm_struct{i_frame-nonSparse_time_shift}.values);
	      nonSparse_norm_RMS(i_frame) = ...
		  std(nonSparse_norm_strength(i_nonSparse) * nonSparse_norm_vals(:));
	    else
	      nonSparse_norm_RMS(i_frame) = ...
		  Sparse_std_array{Sparse_std_ndx(i_nonSparse)}(i_frame-nonSparse_time_shift);
	    endif
	  endif
	endif %% ~isempty(nonSparse_norm_struct)

      else %% ~isempty(nonSparse_struct{i_frame})
	num_nonSparse_frames = i_frame - 1;
	nonSparse_times = nonSparse_times(1:num_nonSparse_frames);
	nonSparse_RMS = nonSparse_RMS(1:num_nonSparse_frames);
	break;
      endif %% ~isempty(nonSparse_struct{i_frame})
    endfor %% i_frame

    if num_nonSparse_frames <= 0
      disp(["num_nonSparse_frames = ", num2str(num_nonSparse_frames)]);
      continue;
    endif
    if plot_flag
      nonSparse_RMS_fig(i_nonSparse) = figure;
      normalized_nonSparse_RMS = nonSparse_RMS ./ (nonSparse_norm_RMS + (nonSparse_norm_RMS==0));
      nonSparse_RMS_hndl = plot(nonSparse_times, normalized_nonSparse_RMS); 
      set(nonSparse_RMS_hndl, "linewidth", 1.5);
      max_nonSparse_RMS(i_nonSparse) = max(normalized_nonSparse_RMS(:));
      min_nonSparse_RMS(i_nonSparse) = min(normalized_nonSparse_RMS(:));
%%      axis(nonSparse_times(1) nonSparse_times(num_nonSparse_frames) ...
%%	   min_nonSparse_RMS(i_nonSparse) max_nonSparse_RMS(i_nonSparse));
      set(nonSparse_RMS_fig(i_nonSparse), "name", ...
	  ["RMS_", nonSparse_list{i_nonSparse,1}, nonSparse_list{i_nonSparse,2}, ...
	   "_", num2str(nonSparse_times(num_nonSparse_frames), "%08d")]);
     saveas(nonSparse_RMS_fig(i_nonSparse), ...
	     [nonSparse_dir, filesep, ...
	      "RMS_", nonSparse_list{i_nonSparse,1}, nonSparse_list{i_nonSparse,2}, ...
	      "_", num2str(nonSparse_times(num_nonSparse_frames), "%08d")], "png");
    endif
    nonSparse_median_RMS = median(nonSparse_RMS(:) ./ (nonSparse_norm_RMS + (nonSparse_norm_RMS==0)));
    disp([nonSparse_list{i_nonSparse,2}, "_", num2str(nonSparse_times(num_nonSparse_frames), "%i"), ...
	  " median RMS = ", num2str(nonSparse_median_RMS)]);
    nonSparse_times_array{i_nonSparse} = nonSparse_times;
    nonSparse_RMS_array{i_nonSparse} = nonSparse_RMS;
    nonSparse_norm_RMS_array{i_nonSparse} = nonSparse_norm_RMS;
    %%keyboard;
  endfor  %% i_nonSparse
endif %% plot_nonSparse







plot_ErrorVsSparse = true && plot_Sparse && plot_nonSparse;
if plot_ErrorVsSparse
  Sparse_axis_index = ones(num_nonSparse,1);
  Sparse_axis_index(2) = 2;
  Sparse_axis_index(3) = 2;
  Sparse_axis_index(4) = 3;
  for i_nonSparse = 1 : num_nonSparse_list
    i_Sparse = Sparse_axis_index(i_nonSparse);
    nonSparse_times = nonSparse_times_array{i_nonSparse};
    nonSparse_RMS = nonSparse_RMS_array{i_nonSparse};
    nonSparse_norm_RMS = nonSparse_norm_RMS_array{i_nonSparse};
    num_nonSparse_frames = length(nonSparse_times);
    if num_nonSparse_frames < 2
      continue;
    endif
    Sparse_hist_rank = Sparse_hist_rank_array{i_Sparse};
    Sparse_times = Sparse_times_array{i_Sparse};
    Sparse_percent_active = Sparse_percent_active_array{i_Sparse};
    Sparse_percent_change = Sparse_percent_change_array{i_Sparse};
    first_nonSparse_time = nonSparse_times(1);
    second_nonSparse_time = nonSparse_times(2);
    last_nonSparse_time = nonSparse_times(end);    
    [first_Sparse_ndx, ~, first_Sparse_diff] = find((Sparse_times - first_nonSparse_time) >= 0, 1, "first");
    [second_Sparse_ndx, ~, second_Sparse_diff] = find(Sparse_times - second_nonSparse_time >= 0, 1, "first");
    if max(Sparse_times(:)) >= last_nonSparse_time
      [last_Sparse_ndx, ~, last_Sparse_diff] = find(Sparse_times - last_nonSparse_time >= 0, 1, "first");
      %%last_nonSparse_ndx = num_nonSparse_frames;
    else
      %%[last_nonSparse_ndx, ~, last_nonSparse_diff] = find(nonSparse_times - Sparse_times(end) < 0, 1, "last");
      last_Sparse_ndx = length(Sparse_times);
    endif
    skip_Sparse_ndx = max(second_Sparse_ndx - first_Sparse_ndx, 1);
    Sparse_vals = 1-Sparse_percent_active(first_Sparse_ndx:skip_Sparse_ndx:last_Sparse_ndx);
    num_Sparse_vals = length(Sparse_vals);
    if num_Sparse_vals < 1
      continue;
    endif
    num_Sparse_bins = 10;
    min_Sparse_val = min(Sparse_vals(:));
    max_Sparse_val = max(Sparse_vals(:));
    skip_Sparse_val = (max_Sparse_val - min_Sparse_val) / num_Sparse_bins;
    if skip_Sparse_val == 0
      skip_Sparse_val = 1;
    endif
    Sparse_bins = min_Sparse_val : skip_Sparse_val : max_Sparse_val;
    Sparse_bins = Sparse_bins(1:end-1);
    Sparse_bin_ndx = ceil((Sparse_vals - min_Sparse_val) / skip_Sparse_val);
    Sparse_bin_ndx(Sparse_bin_ndx < 1) = 1;
    Sparse_bin_ndx(Sparse_bin_ndx > num_Sparse_bins) = num_Sparse_bins;
    mean_nonSparse_RMS = zeros(num_Sparse_bins, 1); 
    std_nonSparse_RMS = zeros(num_Sparse_bins, 1); 
    last_nonSparse_ndx = length(Sparse_vals);
    for i_Sparse_bin = 1 : num_Sparse_bins
      if ~isempty(nonSparse_norm_RMS(Sparse_bin_ndx == i_Sparse_bin))
	mean_nonSparse_RMS(i_Sparse_bin) = ...
	    mean(nonSparse_RMS(Sparse_bin_ndx == i_Sparse_bin) ./ ...
		 nonSparse_norm_RMS(Sparse_bin_ndx == i_Sparse_bin));
	std_nonSparse_RMS(i_Sparse_bin) = ...
	    std(nonSparse_RMS(Sparse_bin_ndx == i_Sparse_bin) ./ ...
		nonSparse_norm_RMS(Sparse_bin_ndx == i_Sparse_bin));
      endif
    endfor %% i_Sparse_bin
    last_nonSparse_ndx = length(Sparse_vals);
    if plot_flag
      normalized_nonSparse_RMS = nonSparse_RMS(1:last_nonSparse_ndx) ./ ...
	       (nonSparse_norm_RMS(1:last_nonSparse_ndx) + (nonSparse_norm_RMS(1:last_nonSparse_ndx) == 0));
      max_nonSparse_RMS = max(normalized_nonSparse_RMS(:));
      ErrorVsSparse_fig = figure;
      ErrorVsSparse_hndl = ...
	  plot(Sparse_vals, ...
	       normalized_nonSparse_RMS, ...
	       "."); 
      if max_nonSparse_RMS <= 1.0
	axis([min(min_Sparse_val,0.90) 1.0 0 1.0]);
      else
	axis([min(min_Sparse_val,0.90) 1.0 0 max_nonSparse_RMS]);
      endif
      hold on
      eh = errorbar(Sparse_bins+skip_Sparse_val/2, mean_nonSparse_RMS, std_nonSparse_RMS);
      set(eh, "color", [0 0 0]);
      set(eh, "linewidth", 1.5);
      set(ErrorVsSparse_fig, "name", ...
	  ["ErrorVsSparse_", nonSparse_list{i_nonSparse,2}, "_", ...
	   num2str(nonSparse_times(num_nonSparse_frames), "%08d")]);
      saveas(ErrorVsSparse_fig, ...
	     [nonSparse_dir, filesep, ...
	      "ErrorVsSparse_", nonSparse_list{i_nonSparse,2}, "_", ...
	      num2str(nonSparse_times(num_nonSparse_frames), "%08d")], "png");
    endif %% plot_flag
    save("-mat", ...
	 [nonSparse_dir, filesep, "ErrorVsSparse_", nonSparse_list{i_nonSparse,2}, "_", ...
	  num2str(nonSparse_times(num_nonSparse_frames), "%08d"), ".mat"], ...
	 "nonSparse_times", "Sparse_vals", "nonSparse_RMS", "nonSparse_norm_RMS", ...
	 "Sparse_bins", "mean_nonSparse_RMS", "std_nonSparse_RMS");	 
  endfor  %% i_nonSparse
endif %% plot_ErrorVsSparse







plot_ReconError = true && plot_nonSparse;
ReconError_nonSparse_ndx = [];
if plot_ReconError && plot_flag
  if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ReconError_list = ...
        {["a5_"], ["Recon"]; ...
         ["a11_"], ["ReconInfra"]; ...
	 ["a5_"], ["Recon"]; ...
         ["a11_"], ["ReconInfra"]};
    num_ReconError_list = size(ReconError_list,1);
    ReconError_skip = repmat(1, num_ReconError_list, 1);
    ReconError_skip(1) = 1;
    ReconError_skip(2) = 1;
    ReconError_norm_list = ...
        {["a2_"], ["Ganglion"]; ...
         ["a2_"], ["Ganglion"]; ...
	 ["a14_"], ["NoiselessGanglion"]; ...
         ["a14_"], ["NoiselessGanglion"]};
    ReconError_norm_strength = ones(num_ReconError_list,1);
    ReconError_nonSparse_ndx = [1 1 4 4];
  elseif strcmp(run_type, "CIFAR_noTask_deep") || strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR_noTask_deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ReconError_list = ...
        {["a3_"], ["Recon"]; ...
         ["a9_"], ["ReconInfra"]};
    num_ReconError_list = size(ReconError_list,1);
    ReconError_skip = repmat(1, num_ReconError_list, 1);
    ReconError_skip(1) = 1;
    ReconError_skip(2) = 1;
    ReconError_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a0_"], ["Image"]};
    ReconError_norm_strength = ones(num_ReconError_list,1);
    ReconError_nonSparse_ndx = [1 1];
  elseif strcmp(run_type, "noPulvinar")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ReconError_list = ...
        {["a5_"], ["Recon"]; ...
         ["a9_"], ["ReconInfra"]; ...
	 ["a5_"], ["Recon"]; ...
         ["a9_"], ["ReconInfra"]};
    num_ReconError_list = size(ReconError_list,1);
    ReconError_skip = repmat(1, num_ReconError_list, 1);
    ReconError_skip(1) = 1;
    ReconError_skip(2) = 1;

    ReconError_norm_list = ...
        {["a2_"], ["Ganglion"]; ...
         ["a2_"], ["Ganglion"]; ...
	 ["a12_"], ["NoiselessGanglion"]; ...
         ["a12_"], ["NoiselessGanglion"]};
%%    ReconError_norm_list = {[],[]};
    ReconError_norm_strength = ones(num_ReconError_list,1);
    ReconError_nonSparse_ndx = [1 1 3 3];
  elseif strcmp(run_type, "V1")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% V1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ReconError_list = ...
        {["a5_"], ["Recon"]};
    num_ReconError_list = size(ReconError_list,1);
    ReconError_skip = repmat(1, num_ReconError_list, 1);
    ReconError_skip(1) = 1;
    ReconError_norm_list = ...
        {["a2_"], ["Ganglion"]};
    ReconError_norm_strength = ones(num_ReconError_list,1);
    ReconError_norm_strength = ones(num_ReconError_list,1);
  elseif strcmp(run_type, "lateral")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ReconError_list = ...
        {["a5_"], ["Recon"]; ...
         ["a8_"], ["Recon2"]; ...
         ["a11_"], ["ReconInfra"]};
    num_ReconError_list = size(ReconError_list,1);
    ReconError_skip = repmat(1, num_ReconError_list, 1);
    ReconError_skip(1) = 1;
    ReconError_skip(2) = 1;
    ReconError_skip(3) = 1;
    ReconError_norm_list = ...
        {["a2_"], ["Ganglion"]; ...
	 ["a2_"], ["Ganglion"]; ...
	 ["a2_"], ["Ganglion"]};
    ReconError_norm_strength = ones(num_ReconError_list,1);
    ReconError_nonSparse_ndx = [1 2 2];
  elseif strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR") || strcmp(run_type, "CIFAR_noTask")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(run_type, "CIFAR_noTask")
      ReconError_list = ...
	  {["a3_"], ["Recon"]; ...
	   ["a7_"], ["ReconInfra"]};
    else
      ReconError_list = ...
	  {["a5_"], ["Recon"]; ...
	   ["a10_"], ["ReconInfra"]};
    endif
    num_ReconError_list = size(ReconError_list,1);
    ReconError_skip = repmat(1, num_ReconError_list, 1);
    ReconError_skip(1) = 1;
    ReconError_skip(2) = 1;
    ReconError_norm_list = ...
        {["a0_"], ["Image"]; ...
	 ["a0_"], ["Image"]};
    ReconError_norm_strength = ones(num_ReconError_list,1);
    ReconError_norm_strength = ...
	[1/sqrt(32*32); 1/sqrt(32*32)];
    ReconError_nonSparse_ndx = [1 1];
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  ReconError_colormap = jet(num_ReconError_list+1);

  %% num frames to skip between stored frames, default is 
  ReconError_hdr = cell(num_ReconError_list,1);
  ReconError_dir = [output_dir, filesep, "ReconError"]
  [status, msg, msgid] = mkdir(ReconError_dir);
  if status ~= 1
    warning(["mkdir(", ReconError_dir, ")", " msg = ", msg]);
  endif 
  for i_ReconError = 1 : num_ReconError_list
    ReconError_file = [output_dir, filesep, ...
		       ReconError_list{i_ReconError,1}, ReconError_list{i_ReconError,2}, ".pvp"]
    if ~exist(ReconError_file, "file")
      warning(["file does not exist: ", ReconError_file]);
      continue;
    endif
    ReconError_fid = fopen(ReconError_file);
    ReconError_hdr{i_ReconError} = readpvpheader(ReconError_fid);
    fclose(ReconError_fid);
    tot_ReconError_frames = ReconError_hdr{i_ReconError}.nbands;
    num_ReconError = tot_ReconError_frames;
    progress_step = ceil(tot_ReconError_frames / 10);
    [ReconError_struct, ReconError_hdr_tmp] = ...
	readpvpfile(ReconError_file, progress_step, ...
		    tot_ReconError_frames, tot_ReconError_frames-num_ReconError+5, ...
		    ReconError_skip(i_ReconError));
    num_ReconError_frames = size(ReconError_struct,1);
    ReconError_times = zeros(num_ReconError_frames,1);
    ReconError_RMS = zeros(num_ReconError_frames,1);

    num_ReconError_norm_frames = 0;
    if ~isempty(ReconError_norm_list{i_ReconError,1}) && ~isempty(ReconError_norm_list{i_ReconError,2})
      ReconError_norm_file = ...
	  [output_dir, filesep, ...
	   ReconError_norm_list{i_ReconError,1}, ...
	   ReconError_norm_list{i_ReconError,2}, ".pvp"]
      if ~exist(ReconError_norm_file, "file")
	warning(["file does not exist: ", ReconError_norm_file]);
	continue;
      endif
      ReconError_norm_fid = fopen(ReconError_norm_file);
      ReconError_norm_hdr{i_ReconError} = readpvpheader(ReconError_norm_fid);
      fclose(ReconError_norm_fid);
      tot_ReconError_norm_frames = ReconError_norm_hdr{i_ReconError}.nbands;
      num_ReconError_norm = tot_ReconError_norm_frames;
      progress_step = ceil(tot_ReconError_norm_frames / 10);
      [ReconError_norm_struct, ReconError_norm_hdr_tmp] = ...
	  readpvpfile(ReconError_norm_file, progress_step, tot_ReconError_norm_frames, ...
		      tot_ReconError_norm_frames-num_ReconError_norm+5, ...
		      ReconError_skip(i_ReconError));
      num_ReconError_norm_frames = size(ReconError_norm_struct,1);
    else
      ReconError_norm_vals = zeros(num_ReconError_frames,1);
      ReconError_norm_struct = [];
    endif

    frame_diff = ...
	(ReconError_struct{num_ReconError_frames}.time - ReconError_struct{1}.time) / ...
	num_ReconError_frames;
    for i_frame = 1 : 1 : num_ReconError_frames
      if ~isempty(ReconError_struct{i_frame})
	ReconError_times(i_frame) = squeeze(ReconError_struct{i_frame}.time);
	ReconError_vals = squeeze(ReconError_struct{i_frame}.values);
	if ~isempty(ReconError_norm_struct)
	  ReconError_norm_time = ReconError_norm_struct{i_frame}.time;
	  ReconError_time_shift = 0;
	  while ReconError_norm_time - ReconError_times(i_frame) > frame_diff && ...
		(i_frame-(ReconError_time_shift+1)) >= 1
	    ReconError_time_shift = ReconError_time_shift + 1;
	    ReconError_norm_time = ReconError_norm_struct{i_frame-ReconError_time_shift}.time;
	  endwhile
	  while ReconError_norm_time - ReconError_times(i_frame) < -frame_diff && ...
		(i_frame-(ReconError_time_shift-1)) <= num_ReconError_norm_frames
	    ReconError_time_shift = ReconError_time_shift - 1;
	    ReconError_norm_time = ReconError_norm_struct{i_frame-ReconError_time_shift}.time;
	  endwhile
	  ReconError_norm_vals = squeeze(ReconError_norm_struct{i_frame-ReconError_time_shift}.values);
	endif
	if (i_frame-ReconError_time_shift) < 1 || (i_frame-ReconError_time_shift) > num_ReconError_norm_frames
	  num_ReconError_frames = i_frame - 1;
	  ReconError_times = ReconError_times(1:num_ReconError_frames);
	  ReconError_RMS = ReconError_RMS(1:num_ReconError_frames);
	  break;
	else
	  ReconError_RMS(i_frame) = ...
	      std(ReconError_vals(:) - ReconError_norm_strength(i_ReconError) * ReconError_norm_vals(:)) / ...
	      (std(ReconError_norm_strength(i_ReconError) * ReconError_norm_vals(:)) + ...
	       (std(ReconError_norm_strength(i_ReconError) * ReconError_norm_vals(:))==0));
	endif
      else
	num_ReconError_frames = i_frame - 1;
	ReconError_times = ReconError_times(1:num_ReconError_frames);
	ReconError_RMS = ReconError_RMS(1:num_ReconError_frames);
	break;
      endif
    endfor %% i_frame

    ReconError_RMS = ReconError_RMS(1:i_frame);
    ReconError_times = ReconError_times(1:i_frame);
    if plot_flag
      if ~isempty(ReconError_nonSparse_ndx)
	ReconError_RMS_fig(i_ReconError) = nonSparse_RMS_fig(ReconError_nonSparse_ndx(i_ReconError));
      else 
	ReconError_RMS_fig(i_ReconError) = figure;
	axis tight;
      endif
      figure(ReconError_RMS_fig(i_ReconError));
      hold on
      ReconError_RMS_hndl = plot(ReconError_times, ReconError_RMS); 
      set(ReconError_RMS_hndl, "linewidth", 1.5);
      set(ReconError_RMS_hndl, "color", ReconError_colormap(i_ReconError+1,:));
      set(ReconError_RMS_fig(i_ReconError), "name", ...
	  ["RMS_", ReconError_list{i_ReconError,2}, "_", ReconError_norm_list{i_ReconError,2}, "_", ...
	   num2str(ReconError_times(num_ReconError_frames), "%08d")]);
      saveas(ReconError_RMS_fig(i_ReconError), ...
	     [ReconError_dir, filesep, ...
	      "RMS_", ...
	      ReconError_list{i_ReconError,2}, "_", ReconError_norm_list{i_ReconError,2}, ...
	      "_", ...
	      num2str(ReconError_times(num_ReconError_frames), "%08d"), ...
	      ], "png");
    endif
    ReconError_median_RMS = median(ReconError_RMS(:));
    disp([ReconError_list{i_ReconError,2}, "_", num2str(ReconError_times(num_ReconError_frames), "%i"), ...
	  " median RMS = ", num2str(ReconError_median_RMS)]);
  endfor  %% i_ReconError
endif %% plot_ReconError








%%keyboard;
plot_weights = false;
if plot_weights
  weights_list = {};
  labelWeights_list = {};
  labels_list = {};
  labelRecon_list = {};
  if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_ndx = [1; 2];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w4_"], ["V1ToError"]; ...
           ["w8_"], ["V2ToError2"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["V1ToError"], "_W"; ...
           ["V2ToError2"], "_W"};
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
  elseif strcmp(run_type, "CIFAR_noTask_deep") || strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR_noTask_deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_ndx = [1; 2];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["V1ToError"]; ...
           ["w5_"], ["V2ToError2"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["V1ToError"], "_W"; ...
           ["V2ToError2"], "_W"};
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
  elseif strcmp(run_type, "noPulvinar")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     sparse_ndx = [1];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w4_"], ["V1ToError"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["V1ToError"], "_W"};
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
  elseif strcmp(run_type, "V1")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% V1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    weights_list = ...
        {["w4_"], ["V1ToError"]};
    sparse_ndx = [1];
  elseif strcmp(run_type, "lateral")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_ndx = [1; 2];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w4_"], ["V1ToError"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["V1ToError"], "_W"};
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
  elseif strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR")  || strcmp(run_type, "CIFAR_noTask") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_ndx = [1];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["V1ToError"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["V1ToError"], "_W"};
      labelWeights_list = ...
	  {[], []};
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  num_weights_list = size(weights_list,1);
  weights_hdr = cell(num_weights_list,1);
  pre_hdr = cell(num_weights_list,1);
  if checkpoint_weights_movie
    weights_movie_dir = [output_dir, filesep, "weights_movie"]
    [status, msg, msgid] = mkdir(weights_movie_dir);
    if status ~= 1
      warning(["mkdir(", weights_movie_dir, ")", " msg = ", msg]);
    endif 
  endif
  weights_dir = [output_dir, filesep, "weights"]
  [status, msg, msgid] = mkdir(weights_dir);
  if status ~= 1
    warning(["mkdir(", weights_dir, ")", " msg = ", msg]);
  endif 
  for i_weights = 1 : num_weights_list
    max_weight_time = 0;
    max_checkpoint = 0;
    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights_file = [checkpoint_dir, filesep, weights_list{i_weights,1}, weights_list{i_weights,2}, ".pvp"];
      if ~exist(weights_file, "file")
	warning(["file does not exist: ", weights_file]);
	continue;
      endif
      weights_fid = fopen(weights_file);
      weights_hdr{i_weights} = readpvpheader(weights_fid);    
      fclose(weights_fid);

      weight_time = weights_hdr{i_weights}.time;
      if weight_time > max_weight_time
	max_weight_time = weight_time;
	max_checkpoint = i_checkpoint;
      endif
    endfor %% i_checkpoint

    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights_file = [checkpoint_dir, filesep, weights_list{i_weights,1}, weights_list{i_weights,2}, ".pvp"];
      if ~exist(weights_file, "file")
	warning(["file does not exist: ", weights_file]);
	continue;
      endif
      weights_fid = fopen(weights_file);
      weights_hdr{i_weights} = readpvpheader(weights_fid);    
      fclose(weights_fid);
      weights_filedata = dir(weights_file);
      weights_framesize = ...
	  weights_hdr{i_weights}.recordsize*weights_hdr{i_weights}.numrecords+weights_hdr{i_weights}.headersize;
      tot_weights_frames = weights_filedata(1).bytes/weights_framesize;
      num_weights = 1;
      progress_step = ceil(tot_weights_frames / 10);
      [weights_struct, weights_hdr_tmp] = ...
	  readpvpfile(weights_file, progress_step, tot_weights_frames, tot_weights_frames-num_weights+1);
      i_frame = num_weights;
      i_arbor = 1;
      weight_vals = squeeze(weights_struct{i_frame}.values{i_arbor});
      weight_time = squeeze(weights_struct{i_frame}.time);
      weights_name =  [weights_list{i_weights,1}, weights_list{i_weights,2}, "_", num2str(weight_time, "%08d")];
      if no_clobber && exist([weights_movie_dir, filesep, weights_name, ".png"]) && i_checkpoint ~= max_checkpoint
	continue;
      endif
      tmp_ndx = sparse_ndx(i_weights);
      if plot_Sparse
	tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
	tmp_rank = [];
      endif
      if plot_Sparse && ~isempty(tmp_rank)
	pre_hist_rank = tmp_rank;
      else
	pre_hist_rank = (1:weights_hdr{i_weights}.nf);
      endif

      if length(labelWeights_list) >= i_weights && ...
	    ~isempty(labelWeights_list{i_weights}) && ...
	    plot_flag && ...
	    i_checkpoint == max_checkpoint
	labelWeights_file = ...
	    [checkpoint_dir, filesep, labelWeights_list{i_weights,1}, labelWeights_list{i_weights,2}, ".pvp"]
	if ~exist(labelWeights_file, "file")
	  warning(["file does not exist: ", labelWeights_file]);
	  continue;
	endif
	labelWeights_fid = fopen(labelWeights_file);
	labelWeights_hdr{i_weights} = readpvpheader(labelWeights_fid);    
	fclose(labelWeights_fid);
	num_labelWeights = 1;
	labelWeights_filedata = dir(labelWeights_file);
	labelWeights_framesize = ...
	    labelWeights_hdr{i_weights}.recordsize * ...
	    labelWeights_hdr{i_weights}.numrecords+labelWeights_hdr{i_weights}.headersize;
	tot_labelWeights_frames = labelWeights_filedata(1).bytes/labelWeights_framesize;
	[labelWeights_struct, labelWeights_hdr_tmp] = ...
	    readpvpfile(labelWeights_file, ...
			progress_step, ...
			tot_labelWeights_frames, ...
			tot_labelWeights_frames-num_labelWeights+1);
	labelWeights_vals = squeeze(labelWeights_struct{i_frame}.values{i_arbor});
	labelWeights_time = squeeze(labelWeights_struct{i_frame}.time);
      else
	labelWeights_vals = [];
	labelWeights_time = [];
      endif

      %% make tableau of all patches
      %%keyboard;
      i_patch = 1;
      num_weights_dims = ndims(weight_vals);
      num_patches = size(weight_vals, num_weights_dims);
      num_patches = min(num_patches, max_patches);
      num_patches_rows = floor(sqrt(num_patches));
      num_patches_cols = ceil(num_patches / num_patches_rows);
      num_weights_colors = 1;
      if num_weights_dims == 4
	num_weights_colors = size(weight_vals,3);
      endif
      if plot_flag && i_checkpoint == max_checkpoint
	weights_fig = figure;
	set(weights_fig, "name", weights_name);
      endif
      weight_patch_array = [];
      for j_patch = 1  : num_patches
	i_patch = pre_hist_rank(j_patch);
	if plot_flag && i_checkpoint == max_checkpoint
	  subplot(num_patches_rows, num_patches_cols, j_patch); 
	endif
	if num_weights_colors == 1
	  patch_tmp = squeeze(weight_vals(:,:,i_patch));
	else
	  patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
	endif
	patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
	min_patch = min(patch_tmp2(:));
	max_patch = max(patch_tmp2(:));
	patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
	patch_tmp2 = uint8(permute(patch_tmp2, [2,1,3])); %% uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
	if plot_flag && i_checkpoint == max_checkpoint
	  imagesc(patch_tmp2); 
	  if num_weights_colors == 1
	    colormap(gray);
	  endif
	  box off
	  axis off
	  axis image
	  if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
	    [~, max_label] = max(squeeze(labelWeights_vals(:,i_patch)));
	    text(size(weight_vals,1)/2, -size(weight_vals,2)/6, num2str(max_label-1), "color", [1 0 0]);
	  endif %% ~empty(labelWeights_vals)
	  %%drawnow;
	endif %% plot_flag
	if isempty(weight_patch_array)
	  weight_patch_array = ...
	      zeros(num_patches_rows*size(patch_tmp2,1), num_patches_cols*size(patch_tmp2,2), size(patch_tmp2,3));
	endif
	col_ndx = 1 + mod(j_patch-1, num_patches_cols);
	row_ndx = 1 + floor((j_patch-1) / num_patches_cols);
	weight_patch_array(((row_ndx-1)*size(patch_tmp2,1)+1):row_ndx*size(patch_tmp2,1), ...
			   ((col_ndx-1)*size(patch_tmp2,2)+1):col_ndx*size(patch_tmp2,2),:) = ...
	    patch_tmp2;
      endfor  %% j_patch
      if plot_flag && i_checkpoint == max_checkpoint
	if ~isempty(labelWeights_vals)
	  saveas(weights_fig, [weights_dir, filesep, [weights_name,"_labeled"], ".png"], "png");
	else
	  saveas(weights_fig, [weights_dir, filesep, weights_name, ".png"], "png");
	endif
      endif
      imwrite(uint8(weight_patch_array), [weights_movie_dir, filesep, weights_name, ".png"], "png");
      %% make histogram of all weights
      if plot_flag && i_checkpoint == max_checkpoint
	weights_hist_fig = figure;
	[weights_hist, weights_hist_bins] = hist(weight_vals(:), 100);
	bar(weights_hist_bins, log(weights_hist+1));
	set(weights_hist_fig, "name", ...
	    ["Hist_",  weights_list{i_weights,1}, weights_list{i_weights,2}, "_", num2str(weight_time, "%08d")]);
	saveas(weights_hist_fig, ...
	       [weights_dir, filesep, "weights_hist_", num2str(weight_time, "%08d")], "png");
      endif

      if ~isempty(labelWeights_vals) && ...
	    ~isempty(labelWeights_time) && ...
	    plot_flag && ...
	    i_checkpoint == max_checkpoint
	%% plot label weights as matrix of column vectors
	[~, maxnum] = max(labelWeights_vals,[],1);
	[maxnum,maxind] = sort(maxnum);
	label_weights_fig = figure;
	imagesc(labelWeights_vals(:,maxind))
	label_weights_str = ...
	    ["LabelWeights_", labelWeights_list{i_weights,2}, "_", num2str(labelWeights_time, "%08d")];
	%%title(label_weights_fig, label_weights_str);
	figure(label_weights_fig); title(label_weights_str);
	saveas(label_weights_fig, [weights_dir, filesep, label_weights_str, ".png"] , "png");

	%% Plot the average movie weights for a label %%
	for label = 0 : size(labelWeights_vals,1)-1 %% anything 0:0
	  labeledWeights_fig = figure;
	  if num_weights_colors == 1
	    imagesc(squeeze(mean(weight_vals(:,:,maxind(maxnum==(label+1))),3))')
	  else
	    imagesc(permute(squeeze(mean(weight_vals(:,:,:,maxind(maxnum==(label+1))),4)),[2,1,3]));
	  endif
	  labeledWeights_str = ...
	      ["labeledWeightsFig_", ...
	       weights_list{i_weights,1}, weights_list{i_weights,2}, "_", num2str(label, "%d"), "_", ...
	       num2str(weight_time, "%08d")];
	  %%title(labeledWeights_fig, labeledWeights_str);
	  title(labeledWeights_str);
	  saveas(labeledWeights_fig,  [weights_dir, filesep, labeledWeights_str, ".png"], "png");
	endfor %% label
      endif  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

    endfor %% i_checkpoint
  endfor %% i_weights
endif  %% plot_weights




%%keyboard;
plot_labelRecon = true;
labels_list = {};
labelRecon_list = {};
if plot_labelRecon
  if strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR") || strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    labels_list = ...
	{["a15_"], ["Labels"]};
    labelRecon_list = ...
	{["a17_"], ["LabelRecon"]};
  endif %% run_type
endif
i_label = 1;
if plot_labelRecon && ~isempty(labels_list) && ~isempty(labelRecon_list)
  labels_file = ...
      [output_dir, filesep, labels_list{i_label,1}, labels_list{i_label,2}, ".pvp"]
  if ~exist(labels_file, "file")
    warning(["does not exist: ", labels_file]);
  else
    labels_fid = fopen(labels_file);
    labels_hdr{i_label} = readpvpheader(labels_fid);    
    fclose(labels_fid);
    tot_labels_frames =  labels_hdr{i_label}.nbands;
    num_labels = min(tot_labels_frames, 1000);  %% number of label guesses to analyze
    progress_step = fix(tot_labels_frames / 10);
    [labels_struct, labels_hdr_tmp] = ...
	readpvpfile(labels_file, ...
		    progress_step, ...
		    tot_labels_frames, ...
		    tot_labels_frames-num_labels+1);
    label_vals = zeros(labels_hdr{i_label}.nf, num_labels);
    label_time = zeros(num_labels,1);
    num_labels_frames = length(labels_struct);
    for i_frame = num_labels_frames:-1:num_labels_frames-num_labels+1
      tmp = squeeze(labels_struct{i_frame}.values);
      if ndims(tmp) > 2
	label_vals(:,i_frame) = squeeze(tmp(fix(size(tmp,1)/2),fix(size(tmp,2)/2),:));
      else
	label_vals(:,i_frame) = squeeze(tmp);
      endif
      label_time(i_frame) = squeeze(labels_struct{i_frame}.time);
    endfor
    
    labelRecon_file = ...
	[output_dir, filesep, labelRecon_list{i_label,1}, labelRecon_list{i_label,2}, ".pvp"]
    if ~exist(labelRecon_file, "file")
      warning(["does not exist: ", labelRecon_file]);
      break;
    endif
    labelRecon_fid = fopen(labelRecon_file);
    labelRecon_hdr{i_label} = readpvpheader(labelRecon_fid);    
    fclose(labelRecon_fid);
    tot_labelRecon_frames = labelRecon_hdr{i_label}.nbands;
    progress_step = fix(tot_labelRecon_frames / 10);
    [labelRecon_struct, labelRecon_hdr_tmp] = ...
	readpvpfile(labelRecon_file, ...
		    progress_step, ...
		    tot_labelRecon_frames, ...
		    tot_labelRecon_frames-num_labels+1);
    labelRecon_vals = zeros(labelRecon_hdr{i_label}.nf, num_labels);
    labelRecon_time = zeros(num_labels,1);
    num_labelRecon_frames = length(labelRecon_struct);
    for i_frame = num_labelRecon_frames:-1:num_labelRecon_frames-num_labels+1
      tmp = squeeze(labelRecon_struct{i_frame}.values);
      if ndims(tmp) > 2
	labelRecon_vals(:,i_frame) = squeeze(tmp(fix(size(tmp,1)/2),fix(size(tmp,2)/2),:));
      else
	labelRecon_vals(:,i_frame) = squeeze(tmp);
      endif
      labelRecon_time(i_frame) = squeeze(labelRecon_struct{i_frame}.time);
    endfor
    delta_frames = 1;
    [max_label_vals, max_label_ndx] = max(label_vals);
    [max_labelRecon_vals, max_labelRecon_ndx] = max(labelRecon_vals);
    for i_shift = 0:2 %% correct i_shift should be 1 but if simulation is running during analysis could be off
      accuracy = ...
	  sum(max_label_ndx(1:end-i_shift)==max_labelRecon_ndx(i_shift+1:end)) / ...
	  (numel(max_label_vals)-i_shift)
    endfor
    
  endif
endif  %% plot_weightLabels


%%keyboard;
plot_weights1_2 = false; %%(true && ~strcmp(run_type, "MNIST"));
if plot_weights1_2
  weights1_2_list = {};
  if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      weights1_2_list = ...
          {["w12_"], ["V2ToError1_2"]};
      post1_2_list = ...
          {["a4_"], ["V1"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["w4_"], ["V1ToError"]};
      image_list = ...
          {["a1_"], ["Bipolar"]};
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["V2ToError1_2"], "_W"};
      post1_2_list = ...
          {["V1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["V1ToError"], ["_W"]};
%%      image_list = ...
%%          {["a1_"], ["Image"]};
      image_list = ...
          {["Bipolar"], ["_A"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "CIFAR_noTask_deep")  || strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR_noTask_deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      weights1_2_list = ...
          {["w9_"], ["V2ToError1_2"]};
      post1_2_list = ...
          {["a2_"], ["V1"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["w1_"], ["V1ToError"]};
      image_list = ...
          {["a0_"], ["Image"]};
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["V2ToError1_2"], "_W"};
      post1_2_list = ...
          {["V1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["V1ToError"], ["_W"]};
%%      image_list = ...
%%          {["a1_"], ["Image"]};
      image_list = ...
          {["Image"], ["_A"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "noPulvinar")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% noPulvinar
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      weights1_2_list = ...
          {["w8_"], ["V2ToError1_2"]};
      post1_2_list = ...
          {["a4_"], ["V1"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["w4_"], ["V1ToError"]};
      image_list = ...
          {["a1_"], ["Bipolar"]};
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["V2ToError1_2"], "_W"};
      post1_2_list = ...
          {["V1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["V1ToError"], ["_W"]};
%%      image_list = ...
%%          {["a0_"], ["Image"]};
      image_list = ...
          {["Bipolar"], ["_A"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "lateral")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% lateral list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      weights1_2_list = ...
          {["w8_"], ["V2ToError2"];...
	   ["w13_"], ["V2ToError1_2"]};
      post1_2_list = ...
          {["a4_"], ["V1"]; ...
	   ["a4_"], ["V1"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["w4_"], ["V1ToError"]; ...
	   ["w4_"], ["V1ToError"]};
      image_list = ...
          {["a1_"], ["Bipolar"]};
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["V2ToError2"], ["_W"]; ...
	   ["V2ToError1_2"], ["_W"]};
      post1_2_list = ...
          {["V1"], ["_A"]; ...
	   ["V1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["V1ToError"], ["_W"]; ...
	   ["V1ToError"], ["_W"]};
%%      image_list = ...
%%          {["a1_"], ["Image"]};
      image_list = ...
          {["Bipolar"], ["_A"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2,2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR") || strcmp(run_type, "CIFAR_noTask")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      weights1_2_list = ...
          {[], []};
      post1_2_list = ...
          {[], []};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {[], []};
      image_list = ...
          {[""], [""]};
      labelWeights_list = ...
	  {["w10_"], ["V2ToLabelError"]};
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["V2ToError1_2"], "_W"};
      post1_2_list = ...
          {["V1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["V1ToError"], ["_W"]};
      image_list = ...
          {["a0_"], ["Image"]};
%%      image_list = ...
%%          {["Image"], ["_A"]};
      if strcmp(run_type, "CIFAR_noTask")
	labelWeights_list = ...
	    {[], []};
      else
	labelWeights_list = ...
	    {["V2ToLabelError"], ["_W"]};
      endif
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2];
    num_checkpoints = size(checkpoints_list,1);
  endif %% run_type
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  num_weights1_2_list = size(weights1_2_list,1);
  if num_weights1_2_list == 0
    break;
  endif

  %% get image header (to get image dimensions)
  i_image = 1;
  image_file = ...
      [output_dir, filesep, image_list{i_image,1}, image_list{i_image,2}, ".pvp"]
  if ~exist(image_file, "file")
    i_checkpoint = 1;
    image_file = ...
	[checkpoints_list{i_checkpoint,:}, filesep, image_list{i_image,1}, image_list{i_image,2}, ".pvp"]
  endif
  if ~exist(image_file, "file")
    error(["file does not exist: ", image_file]);
  endif
  image_fid = fopen(image_file);
  image_hdr = readpvpheader(image_fid);
  fclose(image_fid);

  weights1_2_hdr = cell(num_weights1_2_list,1);
  pre1_2_hdr = cell(num_weights1_2_list,1);
  post1_2_hdr = cell(num_weights1_2_list,1);

  if checkpoint_weights_movie
    weights1_2_movie_dir = [output_dir, filesep, "weights1_2_movie"]
    [status, msg, msgid] = mkdir(weights1_2_movie_dir);
    if status ~= 1
      warning(["mkdir(", weights1_2_movie_dir, ")", " msg = ", msg]);
    endif 
  endif
  weights1_2_dir = [output_dir, filesep, "weights1_2"]
  [status, msg, msgid] = mkdir(weights1_2_dir);
  if status ~= 1
    warning(["mkdir(", weights1_2_dir, ")", " msg = ", msg]);
  endif 
  for i_weights1_2 = 1 : num_weights1_2_list

    max_weight1_2_time = 0;
    max_checkpoint = 0;
    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};

      %% get weight 2->1 file
      weights1_2_file = ...
	  [checkpoint_dir, filesep, weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, ".pvp"]
      if ~exist(weights1_2_file, "file")
	warning(["file does not exist: ", weights1_2_file]);
	continue;
      endif
      weights1_2_fid = fopen(weights1_2_file);
      weights1_2_hdr{i_weights1_2} = readpvpheader(weights1_2_fid);    
      fclose(weights1_2_fid);

      weight1_2_time = weights1_2_hdr{i_weights1_2}.time;
      if weight1_2_time > max_weight1_2_time
	max_weight1_2_time = weight1_2_time;
	max_checkpoint = i_checkpoint;
      endif
    endfor %% i_checkpoint

    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights1_2_file = ...
	  [checkpoint_dir, filesep, weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, ".pvp"]
      if ~exist(weights1_2_file, "file")
	warning(["file does not exist: ", weights1_2_file]);
	continue;
      endif
      weights1_2_fid = fopen(weights1_2_file);
      weights1_2_hdr{i_weights1_2} = readpvpheader(weights1_2_fid);    
      fclose(weights1_2_fid);

      weights1_2_filedata = dir(weights1_2_file);
      weights1_2_framesize = ...
	  weights1_2_hdr{i_weights1_2}.recordsize*weights1_2_hdr{i_weights1_2}.numrecords+weights1_2_hdr{i_weights1_2}.headersize;
      tot_weights1_2_frames = weights1_2_filedata(1).bytes/weights1_2_framesize;
      weights1_2_nxp = weights1_2_hdr{i_weights1_2}.additional(1);
      weights1_2_nyp = weights1_2_hdr{i_weights1_2}.additional(2);
      weights1_2_nfp = weights1_2_hdr{i_weights1_2}.additional(3);

      %% read 2 -> 1 weights
      num_weights1_2 = 1;
      progress_step = ceil(tot_weights1_2_frames / 10);
      [weights1_2_struct, weights1_2_hdr_tmp] = ...
	  readpvpfile(weights1_2_file, progress_step, tot_weights1_2_frames, tot_weights1_2_frames-num_weights1_2+1);
      i_frame = num_weights1_2;
      i_arbor = 1;
      weights1_2_vals = squeeze(weights1_2_struct{i_frame}.values{i_arbor});
      weights1_2_time = squeeze(weights1_2_struct{i_frame}.time);
      weights1_2_name = ...
	  [weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, "_", num2str(weights1_2_time, "%08d")];
      if no_clobber && exist([weights1_2_movie_dir, filesep, weights1_2_name, ".png"]) && i_checkpoint ~= max_checkpoint
	continue;
      endif
      
      %% get weight 1->0 file
      i_weights0_1 = i_weights1_2;
      weights0_1_file = ...
	  [checkpoint_dir, filesep, weights0_1_list{i_weights0_1,1}, weights0_1_list{i_weights0_1,2}, ".pvp"]
      if ~exist(weights0_1_file, "file")
	warning(["file does not exist: ", weights0_1_file]);
	continue;
      endif
      weights0_1_fid = fopen(weights0_1_file);
      weights0_1_hdr{i_weights0_1} = readpvpheader(weights0_1_fid);    
      fclose(weights0_1_fid);
      weights0_1_filedata = dir(weights0_1_file);
      weights0_1_framesize = ...
	  weights0_1_hdr{i_weights0_1}.recordsize*weights0_1_hdr{i_weights0_1}.numrecords+weights0_1_hdr{i_weights0_1}.headersize;
      tot_weights0_1_frames = weights0_1_filedata(1).bytes/weights0_1_framesize;
      weights0_1_nxp = weights0_1_hdr{i_weights0_1}.additional(1);
      weights0_1_nyp = weights0_1_hdr{i_weights0_1}.additional(2);
      weights0_1_nfp = weights0_1_hdr{i_weights0_1}.additional(3);

      %% get post header (to get post layer dimensions)
      i_post1_2 = i_weights1_2;
      post1_2_file = [checkpoint_dir, filesep, post1_2_list{i_post1_2,1}, post1_2_list{i_post1_2,2}, ".pvp"]
      if ~exist(post1_2_file, "file")
	warning(["file does not exist: ", post1_2_file]);
	continue;
      endif
      post1_2_fid = fopen(post1_2_file);
      post1_2_hdr{i_post1_2} = readpvpheader(post1_2_fid);
      fclose(post1_2_fid);
      post1_2_nf = post1_2_hdr{i_post1_2}.nf;

      %% read 1 -> 0 weights
      num_weights0_1 = 1;
      progress_step = ceil(tot_weights0_1_frames / 10);
      [weights0_1_struct, weights0_1_hdr_tmp] = ...
	  readpvpfile(weights0_1_file, progress_step, tot_weights0_1_frames, tot_weights0_1_frames-num_weights0_1+1);
      i_frame = num_weights0_1;
      i_arbor = 1;
      weights0_1_vals = squeeze(weights0_1_struct{i_frame}.values{i_arbor});
      weights0_1_time = squeeze(weights0_1_struct{i_frame}.time);
      
      %% get rank order of presynaptic elements
      tmp_ndx = sparse_ndx(i_weights1_2);
      if plot_Sparse
	tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
	tmp_rank = [];
      endif
      if plot_Sparse && ~isempty(tmp_rank)
	pre_hist_rank = tmp_rank;
      else
	pre_hist_rank = (1:weights1_2_hdr{i_weights1_2}.nf);
      endif

      if exist("labelWeights_list") && length(labelWeights_list) >= i_weights1_2 && ...
	    ~isempty(labelWeights_list{i_weights1_2}) && ...
	    plot_flag && ...
	    i_checkpoint == max_checkpoint
	labelWeights_file = ...
	    [checkpoint_dir, filesep, ...
	     labelWeights_list{i_weights1_2,1}, labelWeights_list{i_weights1_2,2}, ".pvp"]
	if ~exist(labelWeights_file, "file")
	  warning(["file does not exist: ", labelWeights_file]);
	  continue;
	endif
	labelWeights_fid = fopen(labelWeights_file);
	labelWeights_hdr{i_weights1_2} = readpvpheader(labelWeights_fid);    
	fclose(labelWeights_fid);
	num_labelWeights = 1;
	labelWeights_filedata = dir(labelWeights_file);
	labelWeights_framesize = ...
	    labelWeights_hdr{i_weights1_2}.recordsize * ...
	    labelWeights_hdr{i_weights1_2}.numrecords+labelWeights_hdr{i_weights1_2}.headersize;
	tot_labelWeights_frames = labelWeights_filedata(1).bytes/labelWeights_framesize;
	[labelWeights_struct, labelWeights_hdr_tmp] = ...
	    readpvpfile(labelWeights_file, ...
			progress_step, ...
			tot_labelWeights_frames, ...
			tot_labelWeights_frames-num_labelWeights+1);
	labelWeights_vals = squeeze(labelWeights_struct{i_frame}.values{i_arbor});
	labelWeights_time = squeeze(labelWeights_struct{i_frame}.time);
	labeledWeights0_2 = cell(size(labelWeights_vals,1),1);
      else
	labelWeights_vals = [];
	labelWeights_time = [];
      endif


      %% compute layer 2 -> 1 patch size in pixels
      image2post_nx_ratio = image_hdr.nxGlobal / post1_2_hdr{i_post1_2}.nxGlobal;
      image2post_ny_ratio = image_hdr.nyGlobal / post1_2_hdr{i_post1_2}.nyGlobal;
      weights0_1_overlapp_x = weights0_1_nxp - image2post_nx_ratio;
      weights0_1_overlapp_y = weights0_1_nyp - image2post_ny_ratio;
      weights0_2_nxp = ...
	  weights0_1_nxp + ...
	  (weights1_2_nxp - 1) * (weights0_1_nxp - weights0_1_overlapp_x); 
      weights0_2_nyp = ...
	  weights0_1_nyp + ...
	  (weights1_2_nyp - 1) * (weights0_1_nyp - weights0_1_overlapp_y); 

      %% make tableau of all patches
      %%keyboard;
      i_patch = 1;
      num_weights1_2_dims = ndims(weights1_2_vals);
      num_patches0_2 = size(weights1_2_vals, num_weights1_2_dims);
      num_patches0_2 = min(num_patches0_2, max_patches);
      %% algorithms assumes weights1_2 are one to many
      num_patches0_2_rows = floor(sqrt(num_patches0_2));
      num_patches0_2_cols = ceil(num_patches0_2 / num_patches0_2_rows);
      %% for one to many connections: dimensions of weights1_2 are:
      %% weights1_2(nxp, nyp, nf_post, nf_pre)
      if plot_flag && i_checkpoint == max_checkpoint
	weights1_2_fig = figure;
	set(weights1_2_fig, "name", weights1_2_name);
      endif
      max_shrinkage = 8; %% 
      weight_patch0_2_array = [];
      for kf_pre1_2_rank = 1  : num_patches0_2
	kf_pre1_2 = pre_hist_rank(kf_pre1_2_rank);
	if plot_flag && i_checkpoint == max_checkpoint
	  subplot(num_patches0_2_rows, num_patches0_2_cols, kf_pre1_2_rank); 
	endif
	if ndims(weights1_2_vals) == 4
	  patch1_2_tmp = squeeze(weights1_2_vals(:,:,:,kf_pre1_2));
	elseif ndims(weights1_2_vals) == 3
	  patch1_2_tmp = squeeze(weights1_2_vals(:,:,kf_pre1_2));
	  patch1_2_tmp = reshape(patch1_2_tmp, [1,1,1,size(weights1_2_vals,2)]);
	elseif ndims(weights1_2_vals) == 2
	  patch1_2_tmp = squeeze(weights1_2_vals(:,kf_pre1_2));
	  patch1_2_tmp = reshape(patch1_2_tmp, [1,1,1,size(weights1_2_vals,2)]);
	endif
	%% patch0_2_array stores the sum over all post layer 1 neurons, weighted by weights1_2, 
	%% of image patches for each columun of weights0_1 for pre layer 2 neuron kf_pre
	patch0_2_array = cell(size(weights1_2_vals,1),size(weights1_2_vals,2));
	%% patch0_2 stores the complete image patch of the layer 2 neuron kf_pre
	patch0_2 = zeros(weights0_2_nyp, weights0_2_nxp, weights0_1_nfp);
	%% loop over weights1_2 rows and columns
	for weights1_2_patch_row = 1 : weights1_2_nyp
	  for weights1_2_patch_col = 1 : weights1_2_nxp
	    patch0_2_array{weights1_2_patch_row, weights1_2_patch_col} = ...
		zeros([weights0_1_nxp, weights0_1_nyp, weights0_1_nfp]);
	    %% accumulate weights0_1 patches for each post feature separately for each weights0_1 column 
	    for kf_post1_2 = 1 : post1_2_nf
	      patch1_2_weight = patch1_2_tmp(weights1_2_patch_row, weights1_2_patch_col, kf_post1_2);
	      if patch1_2_weight == 0
		continue;
	      endif
	      if weights0_1_nfp == 1
		weights0_1_patch = squeeze(weights0_1_vals(:,:,kf_post1_2));
	      else
		weights0_1_patch = squeeze(weights0_1_vals(:,:,:,kf_post1_2));
	      endif
	      %%  store weights0_1_patch by column
	      patch0_2_array{weights1_2_patch_row, weights1_2_patch_col} = ...
		  patch0_2_array{weights1_2_patch_row, weights1_2_patch_col} + ...
		  patch1_2_weight .* ...
		  weights0_1_patch;
	    endfor %% kf_post1_2
	    row_start = 1+image2post_ny_ratio*(weights1_2_patch_row-1);
	    row_end = image2post_ny_ratio*(weights1_2_patch_row-1)+weights0_1_nyp;
	    col_start = 1+image2post_nx_ratio*(weights1_2_patch_col-1);
	    col_end = image2post_nx_ratio*(weights1_2_patch_col-1)+weights0_1_nxp;
	    patch0_2(row_start:row_end, col_start:col_end, :) = ...
		patch0_2(row_start:row_end, col_start:col_end, :) + ...
		patch0_2_array{weights1_2_patch_row, weights1_2_patch_col};
	  endfor %% weights1_2_patch_col
	endfor %% weights1_2_patch_row
	patch_tmp2 = flipdim(permute(patch0_2, [2,1,3]),1);
	patch_tmp3 = patch_tmp2;
	weights0_2_nyp_shrunken = size(patch_tmp3, 1);
	patch_tmp4 = patch_tmp3(1, :, :);
	while ~any(patch_tmp4(:)) %% && ((weights0_2_nyp - weights0_2_nyp_shrunken) <= max_shrinkage/2)
	  weights0_2_nyp_shrunken = weights0_2_nyp_shrunken - 1;
	  patch_tmp3 = patch_tmp3(2:weights0_2_nyp_shrunken, :, :);
	  patch_tmp4 = patch_tmp3(1, :, :);
	endwhile
	weights0_2_nyp_shrunken = size(patch_tmp3, 1);
	patch_tmp4 = patch_tmp3(weights0_2_nyp_shrunken, :, :);
	while ~any(patch_tmp4(:))
	  weights0_2_nyp_shrunken = weights0_2_nyp_shrunken - 1;
	  patch_tmp3 = patch_tmp3(1:weights0_2_nyp_shrunken, :, :);
	  patch_tmp4 = patch_tmp3(weights0_2_nyp_shrunken, :, :);
	endwhile
	weights0_2_nxp_shrunken = size(patch_tmp3, 2);
	patch_tmp4 = patch_tmp3(:, 1, :);
	while ~any(patch_tmp4(:)) %% && ((weights0_2_nyp - weights0_2_nyp_shrunken) <= max_shrinkage/2)
	  weights0_2_nxp_shrunken = weights0_2_nxp_shrunken - 1;
	  patch_tmp3 = patch_tmp3(:, 2:weights0_2_nxp_shrunken, :);
	  patch_tmp4 = patch_tmp3(:, 1, :);
	endwhile
	weights0_2_nxp_shrunken = size(patch_tmp3, 2);
	patch_tmp4 = patch_tmp3(:, weights0_2_nxp_shrunken, :);
	while ~any(patch_tmp4(:))
	  weights0_2_nxp_shrunken = weights0_2_nxp_shrunken - 1;
	  patch_tmp3 = patch_tmp3(:, 1:weights0_2_nxp_shrunken, :);
	  patch_tmp4 = patch_tmp3(:, weights0_2_nxp_shrunken, :);
	endwhile
	min_patch = min(patch_tmp3(:));
	max_patch = max(patch_tmp3(:));
	patch_tmp5 = ...
	    uint8((flipdim(patch_tmp3,1) - min_patch) * 255 / ...
		  (max_patch - min_patch + ((max_patch - min_patch)==0)));
	
	if plot_flag && i_checkpoint == max_checkpoint
	  imagesc(patch_tmp5); 
	  if weights0_1_nfp == 1
	    colormap(gray);
	  endif
	  box off
	  axis off
	  axis image
	  if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
	    [~, max_label] = max(squeeze(labelWeights_vals(:,kf_pre1_2)));
	    text(weights0_2_nyp_shrunken/2, -weights0_2_nxp_shrunken/6, num2str(max_label-1), "color", [1 0 0]);
	  endif %% ~empty(labelWeights_vals)
	  %%drawnow;
	endif %% plot_flag && i_checkpoint == max_checkpoint

	if isempty(weight_patch0_2_array)
	  weight_patch0_2_array = ...
	      zeros(num_patches0_2_rows*weights0_2_nyp_shrunken, num_patches0_2_cols*weights0_2_nxp_shrunken, weights0_1_nfp);
	endif
	col_ndx = 1 + mod(kf_pre1_2_rank-1, num_patches0_2_cols);
	row_ndx = 1 + floor((kf_pre1_2_rank-1) / num_patches0_2_cols);
	weight_patch0_2_array((1+(row_ndx-1)*weights0_2_nyp_shrunken):(row_ndx*weights0_2_nyp_shrunken), ...
			      (1+(col_ndx-1)*weights0_2_nxp_shrunken):(col_ndx*weights0_2_nxp_shrunken),:) = ...
	    patch_tmp5;

	%% Plot the average movie weights for a label %%
	if ~isempty(labelWeights_vals) 
	  if ~isempty(labeledWeights0_2{max_label})
	    labeledWeights0_2{max_label} = labeledWeights0_2{max_label} + double(patch_tmp5);
	  else
	    labeledWeights0_2{max_label} = double(patch_tmp5);
	  endif
	endif %%  ~isempty(labelWeights_vals) 

      endfor %% kf_pre1_2_ank

      if plot_flag && i_checkpoint == max_checkpoint
	saveas(weights1_2_fig, [weights1_2_dir, filesep, weights1_2_name, ".png"], "png");
	if ~isempty(labelWeights_vals) 
	  for label = 1:size(labelWeights_vals,1)
	    labeledWeights_str = ...
		["labeledWeightsFig_", ...
		 weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, ...
		 "_", num2str(label, "%d"), "_", ...
		 num2str(weight_time, "%08d")];
	    labeledWeights_fig = figure;
	    title(labeledWeights_str);
	    imagesc(squeeze(labeledWeights0_2{label}));
	    saveas(labeledWeights_fig,  [weights_dir, filesep, labeledWeights_str, ".png"], "png");
	  endfor %% label = 1:size(labelWeights_vals,1)
	endif %%  ~isempty(labelWeights_vals) 

      endif
      imwrite(uint8(weight_patch0_2_array), [weights1_2_movie_dir, filesep, weights1_2_name, ".png"], "png");


      %% make histogram of all weights
      if plot_flag && i_checkpoint == max_checkpoint
	weights1_2_hist_fig = figure;
	[weights1_2_hist, weights1_2_hist_bins] = hist(weights1_2_vals(:), 100);
	bar(weights1_2_hist_bins, log(weights1_2_hist+1));
	set(weights1_2_hist_fig, "name", ...
	    ["Hist_", weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, "_", ...
	     num2str(weights1_2_time, "%08d")]);
	saveas(weights1_2_hist_fig, ...
	       [weights1_2_dir, filesep, "weights1_2_hist_", weights1_2_list{i_weights1_2,2}, "_", ...
		num2str(weights1_2_time, "%08d")], "png");
      endif

      %% plot average labelWeights for each label
      if ~isempty(labelWeights_vals) && ...
	    ~isempty(labelWeights_time) && ...
	    plot_flag && ...
	    i_checkpoint == max_checkpoint

	%% plot label weights as matrix of column vectors
	ranked_labelWeights = labelWeights_vals(:, pre_hist_rank(1:num_patches0_2));
	[~, max_label] = max(ranked_labelWeights,[],1);
	[max_label_sorted, max_label_ndx] = sort(max_label);
	label_weights_fig = figure;
	imagesc(ranked_labelWeights(:,max_label_ndx))
	label_weights_str = ...
	    ["LabelWeights_", labelWeights_list{i_weights,2}, "_", num2str(labelWeights_time, "%08d")];
	%%title(label_weights_fig, label_weights_str);
	figure(label_weights_fig); title(label_weights_str);
	saveas(label_weights_fig, [weights_dir, filesep, label_weights_str, ".png"] , "png");

      endif  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

    endfor %% i_checkpoint

  endfor %% i_weights
  
endif  %% plot_weights









%%keyboard;
plot_weightsN_Nplus1 = false;
weightsN_Nplus1_list = {};
if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown")
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% deep list
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(run_type, "CIFAR_noTask_deep")  || strcmp(run_type, "CIFAR_deep") 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% CIFAR_noTask_deep list
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% list of weights from layerN to layer0 (Image)
  if ~checkpoint_weights_movie
    checkpoints_list = {output_dir};
    weightsN_Nplus1_list = ...
        {["w15_"], ["V4ToError2_4"], ["w9_"], ["V2ToError1_2"], ["w1_"], ["V1ToError"]};
    layersN_Nplus1_list = ...
        {["a10_"], ["V4"], ["a4_"], ["V2"], ["a2_"], ["V1"], ["a0_"], ["Image"]};
    labelWeights_list = ...
	{["w21_"], ["V4ToLabelError"]};
  else
    checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    weightsN_Nplus1_list = ...
        {["V4ToError2_4"], ["_W"], ["V2ToError1_2"], ["_W"], ["V1ToError"], ["_W"]};
    layersN_Nplus1_list = ...
        {["V4"], ["_A"], ["V2"], ["_A"], ["V1"], ["_A"], ["Image"], ["_A"]};
    labelWeights_list = ...
	{["V4ToLabelError"], ["_W"]};
  endif %% checkpoint_weights_movie
  %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
  sparse_ndx = [3];
  num_checkpoints = size(checkpoints_list,1);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(run_type, "noPulvinar")
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% noPulvinar
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(run_type, "lateral")
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% lateral list
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(run_type, "MNIST") || strcmp(run_type, "CIFAR") || strcmp(run_type, "CIFAR_noTask")
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% MNIST/CIFAR
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
endif %% run_type
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_weightsN_Nplus1_list = size(weightsN_Nplus1_list,1);
num_layersN_Nplus1_list = size(layersN_Nplus1_list,2)/2;
if size(weightsN_Nplus1_list,2)/2 ~= num_layersN_Nplus1_list-1
  error(["num_weightsN_Nplus1_list ~= num_layersN_Nplus1_list-1", ...
	 ", num_weightsN_Nplus1_list = ", num2str(num_weightsN_Nplus1_list), ...
	 ", num_layersN_Nplus1_list = ", num2str(num_layersN_Nplus1_list)]);
endif
if num_weightsN_Nplus1_list == 0
  plot_weightsN_Nplus1 = false;
endif

if plot_weightsN_Nplus1
  weightsN_Nplus1_vals = cell(num_weightsN_Nplus1_list, num_layersN_Nplus1_list-1);
  weightsN_Nplus1_hdr = cell(num_weightsN_Nplus1_list, num_layersN_Nplus1_list-1);
  weightsN_Nplus1_framesize = zeros(num_weightsN_Nplus1_list, num_layersN_Nplus1_list-1);
  weightsN_Nplus1_nxp = zeros(num_weightsN_Nplus1_list, num_layersN_Nplus1_list-1);
  weightsN_Nplus1_nyp = zeros(num_weightsN_Nplus1_list, num_layersN_Nplus1_list-1);
  weightsN_Nplus1_nfp = zeros(num_weightsN_Nplus1_list, num_layersN_Nplus1_list-1);
  layersN_Nplus1_hdr = cell(num_weightsN_Nplus1_list, num_layersN_Nplus1_list);
  layersN_Nplus1_nx = zeros(num_weightsN_Nplus1_list, num_layersN_Nplus1_list);
  layersN_Nplus1_ny = zeros(num_weightsN_Nplus1_list, num_layersN_Nplus1_list);
  layersN_Nplus1_nf = zeros(num_weightsN_Nplus1_list, num_layersN_Nplus1_list);

  if checkpoint_weights_movie
    weightsN_Nplus1_movie_dir = [output_dir, filesep, "weightsN_Nplus1_movie"]
    [status, msg, msgid] = mkdir(weightsN_Nplus1_movie_dir);
    if status ~= 1
      warning(["mkdir(", weightsN_Nplus1_movie_dir, ")", " msg = ", msg]);
    endif 
  endif
  weightsN_Nplus1_dir = [output_dir, filesep, "weightsN_Nplus1"]
  [status, msg, msgid] = mkdir(weightsN_Nplus1_dir);
  if status ~= 1
    warning(["mkdir(", weightsN_Nplus1_dir, ")", " msg = ", msg]);
  endif 
  for i_weightN_Nplus1 = 1 : num_weightsN_Nplus1_list

    %% find last (most recent) checkpoint
    max_weightN_Nplus1_time = 0;
    max_checkpoint = 0;
    for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};

      %% get weight N->N+1 file
      weightsN_Nplus1_file = ...
	  [checkpoint_dir, filesep, ...
	   weightsN_Nplus1_list{i_weightN_Nplus1,1}, ...
	   weightsN_Nplus1_list{i_weightN_Nplus1,2}, ".pvp"]
      if ~exist(weights1_2_file, "file")
	warning(["file does not exist: ", weightsN_Nplus1_file]);
	continue;
      endif
      weightsN_Nplus1_fid = fopen(weightsN_Nplus1_file);
      weightsN_Nplus1_hdr{i_weights1_2} = readpvpheader(weightsN_Nplus1_fid);    
      fclose(weightsN_Nplus1_fid);

      weightN_Nplus1_time = weightsN_Nplus1_hdr{i_weightN_Nplus1}.time;
      if weightN_Nplus1_time > max_weightN_Nplus1_time
	max_weightN_Nplus1_time = weightN_Nplus1_time;
	max_checkpoint = i_checkpoint;
      endif
    endfor %% i_checkpoint


    %% get weights headers
    i_layerN_Nplus1 = 1;
    for i_layerN_Nplus1 = 1 : num_layersN_Nplus1_list-1
      weightsN_Nplus1_file = ...
	  [checkpoint_dir, filesep, ...
	   weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2-1}, ...
	   weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2}, ".pvp"]
      if ~exist(weightsN_Nplus1_file, "file")
	warning(["file does not exist: ", weightsN_Nplus1_file]);
	continue;
      endif
      weightsN_Nplus1_fid = fopen(weightsN_Nplus1_file);
      weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1} = readpvpheader(weightsN_Nplus1_fid);    
      fclose(weightsN_Nplus1_fid);
      weightsN_Nplus1_filedata = dir(weightsN_Nplus1_file);
      weightsN_Nplus1_framesize = ...
	  weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.recordsize * ...
	  weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.numrecords + ...
	  weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.headersize;
      weightsN_Nplus1_totframes(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	  weightsN_Nplus1_filedata(1).bytes/weightsN_Nplus1_framesize;
      weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	  weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.additional(1);
      weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	  weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.additional(2);
      weightsN_Nplus1_nfp(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	  weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.additional(3);
    endfor %% i_layerN_Nplus1
    
    %% get layer headers
    checkpoint_dir = checkpoints_list{max_checkpoint,:};
    for i_layerN_Nplus1 = 1 : num_layersN_Nplus1_list
      layersN_Nplus1_file = ...
	  [checkpoint_dir, filesep, ...
	   layersN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2-1}, ...
	   layersN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2}, ...
	   ".pvp"]
      if ~exist(layersN_Nplus1_file, "file")
	warning(["file does not exist: ", layersN_Nplus1_file]);
	continue;
      endif
      layersN_Nplus1_fid = fopen(layersN_Nplus1_file);
      layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1} = readpvpheader(layersN_Nplus1_fid);
      fclose(layersN_Nplus1_fid);
      layersN_Nplus1_nx(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	  layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.nx;
      layersN_Nplus1_ny(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	  layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.ny;
      layersN_Nplus1_nf(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	  layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.nf;
    endfor %% i_layerN_Nplus1

    %% labels (if present)
    if length(labelWeights_list) >= i_weightN_Nplus1 && ...
	  ~isempty(labelWeights_list{i_weightN_Nplus1}) && ...
	  plot_flag 
      labelWeights_file = ...
	  [checkpoint_dir, filesep, ...
	   labelWeights_list{i_weightN_Nplus1,1}, labelWeights_list{i_weightN_Nplus1,2}, ".pvp"]
      if ~exist(labelWeights_file, "file")
	warning(["file does not exist: ", labelWeights_file]);
	continue;
      endif
      labelWeights_fid = fopen(labelWeights_file);
      labelWeights_hdr{i_weightN_Nplus1} = readpvpheader(labelWeights_fid);    
      fclose(labelWeights_fid);
      num_labelWeights = 1;
      labelWeights_filedata = dir(labelWeights_file);
      labelWeights_framesize = ...
	  labelWeights_hdr{i_weightN_Nplus1}.recordsize * ...
	  labelWeights_hdr{i_weightN_Nplus1}.numrecords+labelWeights_hdr{i_weightN_Nplus1}.headersize;
      tot_labelWeights_frames = labelWeights_filedata(1).bytes/labelWeights_framesize;
      [labelWeights_struct, labelWeights_hdr_tmp] = ...
	  readpvpfile(labelWeights_file, ...
		      progress_step, ...
		      tot_labelWeights_frames, ...
		      tot_labelWeights_frames-num_labelWeights+1);
      labelWeights_vals = squeeze(labelWeights_struct{i_frame}.values{i_arbor});
      labelWeights_time = squeeze(labelWeights_struct{i_frame}.time);
      labeledWeightsNminus1_Nplus1 = cell(size(labelWeights_vals,1),1);
    else
      labelWeights_vals = [];
      labelWeights_time = [];
    endif %% labels
    


    %% loop over checkpoints
    for i_checkpoint = max_checkpoint %% 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};

      %% read the top layer of weights to initialize weightsN_Nplus1_vals
      i_layerN_Nplus1 = 1;
      weightsN_Nplus1_file = ...
	  [checkpoint_dir, filesep, ...
	   weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2-1}, ...
	   weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2}, ".pvp"]
      if ~exist(weightsN_Nplus1_file, "file")
	warning(["file does not exist: ", weightsN_Nplus1_file]);
	continue;
      endif      
      num_weightsN_Nplus1 = 1;
      tot_weightsN_Nplus1_frames = weightsN_Nplus1_totframes(i_weightN_Nplus1, i_layerN_Nplus1);
      progress_step = ceil( tot_weightsN_Nplus1_frames/ 10);
      [weightsN_Nplus1_struct, weightsN_Nplus1_hdr_tmp] = ...
	  readpvpfile(weightsN_Nplus1_file, progress_step, ...
		      tot_weightsN_Nplus1_frames, ...
		      tot_weightsN_Nplus1_frames-num_weightsN_Nplus1+1);
      i_frame = num_weightsN_Nplus1;
      i_arbor = 1;
      weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1} = ...
	  squeeze(weightsN_Nplus1_struct{i_frame}.values{i_arbor});
      if ndims(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1}) == 4
	weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1} = ...
	    permute(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1}, [2,1,3,4]);
      else
	weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1} = ...
	    permute(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1}, [2,1,3]);
      endif
      weightsN_Nplus1_time = squeeze(weightsN_Nplus1_struct{i_frame}.time);
      weightsN_Nplus1_name = ...
	  [weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2-1}, ...
	   weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2}, ...
	   "_", num2str(weightsN_Nplus1_time, "%08d")];
      if no_clobber && ...
	    exist([weightsN_Nplus1_movie_dir, filesep, weightsN_Nplus1_name, ".png"]) && ...
	    i_checkpoint ~= max_checkpoint
	continue;
      endif
      num_weightsN_Nplus1_dims = ...
	  ndims(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1});
      num_patchesN_Nplus1 = ...
	  size(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1}, num_weightsN_Nplus1_dims);
      %% algorithms assumes weights1_2 are one to many
      num_patchesN_Nplus1_rows = floor(sqrt(min(num_patchesN_Nplus1, max_patches)));
      num_patchesN_Nplus1_cols = ceil(min(num_patchesN_Nplus1, max_patches) / num_patchesN_Nplus1_rows);
      
      %% get rank order of presynaptic elements
      tmp_ndx = sparse_ndx(i_weightN_Nplus1);
      if plot_Sparse
	tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
	tmp_rank = [];
      endif
      if plot_Sparse && ~isempty(tmp_rank)
	pre_hist_rank = tmp_rank;
      else
	pre_hist_rank = (1:weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.nf);
      endif

      %% loop over lower layers
      for i_layerN_Nplus1 = 1 : num_layersN_Nplus1_list - 2  %% last layer is image

	weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	    size(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1},1);
	weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	    size(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1},2);
	weightsN_Nplus1_nfp(i_weightN_Nplus1, i_layerN_Nplus1) = ...
	    size(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1},3);

	weightsNminus1_Nplus1_name = ...
	    [checkpoint_dir, filesep, ...
	     weightsN_Nplus1_list{i_weightN_Nplus1, (i_layerN_Nplus1+1)*2-1}, ...
	     weightsN_Nplus1_list{i_weightN_Nplus1, (i_layerN_Nplus1+1)*2}, ...
	     ".pvp"]

	%% get weight N-1->N file (next set of weights in hierarchy)
	weightsNminus1_N_file = ...
	    [checkpoint_dir, filesep, ...
	     weightsN_Nplus1_list{i_weightN_Nplus1, (i_layerN_Nplus1+1)*2-1}, ...
	     weightsN_Nplus1_list{i_weightN_Nplus1, (i_layerN_Nplus1+1)*2}, ...
	     ".pvp"]
	if ~exist(weightsNminus1_N_file, "file")
	  warning(["file does not exist: ", weightsNminus1_N_file]);
	  continue;
	endif
	num_weightsNminus1_N = 1;
	tot_weightsNminus1_N_frames = weightsN_Nplus1_totframes(i_weightN_Nplus1, i_layerN_Nplus1+1);
	progress_step = ceil(tot_weightsNminus1_N_frames / 10);
	[weightsNminus1_N_struct, weightsNminus1_N_hdr_tmp] = ...
	    readpvpfile(weightsNminus1_N_file, progress_step, ...
			tot_weightsNminus1_N_frames, ...
			tot_weightsNminus1_N_frames-num_weightsNminus1_N+1);
	i_frame = num_weightsNminus1_N;
	i_arbor = 1;
	weightsNminus1_N_vals = squeeze(weightsNminus1_N_struct{i_frame}.values{i_arbor});
	if ndims(weightsNminus1_N_vals) == 4
	  weightsNminus1_N_vals = permute(weightsNminus1_N_vals, [2,1,3,4]);
	else
	  weightsNminus1_N_vals = permute(weightsNminus1_N_vals, [2,1,3]);
	endif
	weightsNminus1_N_time = squeeze(weightsNminus1_N_struct{i_frame}.time);
	weightsNminus1_N_nyp = size(weightsNminus1_N_vals,1);
	weightsNminus1_N_nxp = size(weightsNminus1_N_vals,2);
	weightsNminus1_N_nfp = size(weightsNminus1_N_vals,3);
	
	%% compute layer N+1 -> N-1 patch size
	Nminus1_N_nx_ratio = ...
	    layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1+2}.nxGlobal / ...
	    layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1+1}.nxGlobal;
	Nminus1_N_ny_ratio = ...
	    layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1+2}.nyGlobal / ...
	    layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1+1}.nyGlobal;
	weightsNminus1_N_overlapp_x = ...
	    weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1+1) - Nminus1_N_nx_ratio;
	weightsNminus1_N_overlapp_y = ...
	    weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1+1) - Nminus1_N_ny_ratio;
	weightsNminus1_Nplus1_nxp = ...
	    weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1+1) + ...
	    (weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1) - 1) * ...
	    (weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1+1) - weightsNminus1_N_overlapp_x); 
	weightsNminus1_Nplus1_nyp = ...
	    weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1+1) + ...
	    (weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1) - 1) * ...
	    (weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1+1) - weightsNminus1_N_overlapp_y); 
	weightsNminus1_Nplus1_nfp = ...
	    layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1+2}.nf;


	%% make tableau of all patches
	%% for one to many connections: dimensions of weightsN_Nplus1 are:
	%% weightsN_Nplus1_vals{i_weightN_Nplus1, i_layersN_Nplus}(nxp, nyp, nf_post, nf_pre)
	if plot_flag && ...
	      i_checkpoint == max_checkpoint && ...
	      i_layerN_Nplus1 == num_layersN_Nplus1_list-2 
	  weightsNminus1_Nplus1_fig = figure;
	  set(weightsNminus1_Nplus1_fig, "name", weightsN_Nplus1_name);
	endif %% plot_flag

	max_shrinkage = 8; %% 
	%% storage for next iteration of deconvolved weights
	weightsNminus1_Nplus1_array = [];
	%% plot weights in rank order
	for kf_preN_Nplus1_rank = 1  : num_patchesN_Nplus1
	  kf_preN_Nplus1 = pre_hist_rank(kf_preN_Nplus1_rank);

	  plotNminus1_Nplus1_flag = ...
	      plot_flag && ...
	      i_layerN_Nplus1 == num_layersN_Nplus1_list-2 && ...
	      i_checkpoint == max_checkpoint && ...
	      kf_preN_Nplus1_rank <= max_patches;
	  if plotNminus1_Nplus1_flag
	    subplot(num_patchesN_Nplus1_rows, num_patchesN_Nplus1_cols, kf_preN_Nplus1_rank); 
	  endif

	  if num_weightsN_Nplus1_dims == 4
	    patchN_Nplus1_tmp = ...
		squeeze(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1}(:,:,:,kf_preN_Nplus1));
	  elseif num_weightsN_Nplus1_dims <= 3
	    patchN_Nplus1_tmp = ...
		squeeze(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1}(:,:,kf_preN_Nplus1));
	    patchN_Nplus1_tmp = ...
		reshape(patchN_Nplus1_tmp, ...
			[weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1), ...
			 weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1), ...
			 1]);
	  endif

	  %% patchNminus1_Nplus1_array stores the sum over all patches, given by 
	  %% weightsNminus1_Nplus1_vals, for every neuron in layer i_layerN_Nplus1+1 
	  %% that is postsynaptic to layer i_layerN_Nplus1 as denoted by
	  %% weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1}.
	  %% In other words, for the presynaptic neuron in layer i_layerN_Nplus1
	  %% specified by feature index kf_preN_Nplus1,
	  %% we deconvolve each of its postsynaptic  targets in layer i_layerN_Nplus1+1,
	  %% specified by weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1},
	  %% with each of its targets in layer i_layerN_Nplus1+2, 
	  %% specified by weightsNminus1_Nplus1_vals.
	  patchNminus1_Nplus1_array = ...
	      cell(weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1), ...
		   weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1));
	  %% patchNminus1_Nplus1 stores the complete image patch of the layer 
	  %% i_layerN_Nplus1 neuron kf_preN_Nplus1
	  patchNminus1_Nplus1 = ...
	      zeros(weightsNminus1_Nplus1_nyp, ...
		    weightsNminus1_Nplus1_nxp, ...
		    weightsNminus1_Nplus1_nfp);
	  %% loop over weightsN_Nplus1 rows and columns
	  for N_Nplus1_patch_row = 1 : weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1)
	    for N_Nplus1_patch_col = 1 : weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1)

	      patchNminus1_Nplus1_array{N_Nplus1_patch_row, N_Nplus1_patch_col} = ...
		  zeros([weightsN_Nplus1_nyp(i_weightN_Nplus1, i_layerN_Nplus1+1), ...
			 weightsN_Nplus1_nxp(i_weightN_Nplus1, i_layerN_Nplus1+1), ...
			 weightsN_Nplus1_nfp(i_weightN_Nplus1, i_layerN_Nplus1+1)]);

	      %% accumulate weightsNminus1_N patches for each post feature separately 
	      %% for each weightsN_Nplus1 column 
	      for kf_postN_Nplus1 = 1 : layersN_Nplus1_nf(i_weightN_Nplus1,i_layerN_Nplus1+1)
		patchN_Nplus1_weight = ...
		    patchN_Nplus1_tmp(N_Nplus1_patch_row, N_Nplus1_patch_col, kf_postN_Nplus1);
		if patchN_Nplus1_weight == 0
		  continue;
		endif
		if weightsNminus1_N_nfp == 1
		  weightsNminus1_N_patch = squeeze(weightsNminus1_N_vals(:,:,kf_postN_Nplus1));
		else
		  weightsNminus1_N_patch = squeeze(weightsNminus1_N_vals(:,:,:,kf_postN_Nplus1));
		endif
		%%  store weightsNminus1_N_patch by column
		patchNminus1_Nplus1_array{N_Nplus1_patch_row, N_Nplus1_patch_col} = ...
		    patchNminus1_Nplus1_array{N_Nplus1_patch_row, N_Nplus1_patch_col} + ...
		    patchN_Nplus1_weight .* ...
		    weightsNminus1_N_patch;
	      endfor %% kf_postN_Nplus1
	      Nminus1_Nplus1_row_start = 1+Nminus1_N_ny_ratio*(N_Nplus1_patch_row-1);
	      Nminus1_Nplus1_row_end = Nminus1_N_ny_ratio*(N_Nplus1_patch_row-1)+weightsNminus1_N_nyp;
	      Nminus1_Nplus1_col_start = 1+Nminus1_N_nx_ratio*(N_Nplus1_patch_col-1);
	      Nminus1_Nplus1_col_end = Nminus1_N_nx_ratio*(N_Nplus1_patch_col-1)+weightsNminus1_N_nxp;
	      patchNminus1_Nplus1(Nminus1_Nplus1_row_start:Nminus1_Nplus1_row_end, ...
				  Nminus1_Nplus1_col_start:Nminus1_Nplus1_col_end, :) = ...
		  patchNminus1_Nplus1(Nminus1_Nplus1_row_start:Nminus1_Nplus1_row_end, ...
				      Nminus1_Nplus1_col_start:Nminus1_Nplus1_col_end, :) + ...
		  patchNminus1_Nplus1_array{N_Nplus1_patch_row, N_Nplus1_patch_col};
	    endfor %% N_Nplus1_patch_col
	  endfor %% N_Nplus1_patch_row

	  %% get shrunken patch (if last level, but only do once)
	  if 1 %%i_layerN_Nplus1 < num_layersN_Nplus1_list-2
	    patch_tmp3 = patchNminus1_Nplus1;
	    weightsNminus1_Nplus1_nyp_shrunken = weightsNminus1_Nplus1_nyp;
	    weightsNminus1_Nplus1_nxp_shrunken = weightsNminus1_Nplus1_nxp;
	  elseif kf_preN_Nplus1_rank == 1 
	    patch_tmp3 = patchNminus1_Nplus1;
	    weightsNminus1_Nplus1_nyp_shrunken = size(patch_tmp3, 1);
	    patch_tmp4 = patch_tmp3(1, :, :);
	    while ~any(patch_tmp4(:)) %% && ((weights0_2_nyp - weights0_2_nyp_shrunken) <= max_shrinkage/2)
	      weightsNminus1_Nplus1_nyp_shrunken = weightsNminus1_Nplus1_nyp_shrunken - 1;
	      patch_tmp3 = patch_tmp3(2:weightsNminus1_Nplus1_nyp_shrunken, :, :);
	      patch_tmp4 = patch_tmp3(1, :, :);
	    endwhile
	    weightsNminus1_Nplus1_nyp_shrunken = size(patch_tmp3, 1);
	    patch_tmp4 = patch_tmp3(weightsNminus1_Nplus1_nyp_shrunken, :, :);
	    while ~any(patch_tmp4(:))
	      weightsNminus1_Nplus1_nyp_shrunken = weightsNminus1_Nplus1_nyp_shrunken - 1;
	      patch_tmp3 = patch_tmp3(1:weightsNminus1_Nplus1_nyp_shrunken, :, :);
	      patch_tmp4 = patch_tmp3(weightsNminus1_Nplus1_nyp_shrunken, :, :);
	    endwhile
	    weightsNminus1_Nplus1_nxp_shrunken = size(patch_tmp3, 2);
	    patch_tmp4 = patch_tmp3(:, 1, :);
	    while ~any(patch_tmp4(:)) %% && ((weightsNminus1_Nplus1_nyp - weightsNminus1_Nplus1_nyp_shrunken) <= max_shrinkage/2)
	      weightsNminus1_Nplus1_nxp_shrunken = weightsNminus1_Nplus1_nxp_shrunken - 1;
	      patch_tmp3 = patch_tmp3(:, 2:weightsNminus1_Nplus1_nxp_shrunken, :);
	      patch_tmp4 = patch_tmp3(:, 1, :);
	    endwhile
	    weightsNminus1_Nplus1_nxp_shrunken = size(patch_tmp3, 2);
	    patch_tmp4 = patch_tmp3(:, weightsNminus1_Nplus1_nxp_shrunken, :);
	    while ~any(patch_tmp4(:))
	      weightsNminus1_Nplus1_nxp_shrunken = weightsNminus1_Nplus1_nxp_shrunken - 1;
	      patch_tmp3 = patch_tmp3(:, 1:weightsNminus1_Nplus1_nxp_shrunken, :);
	      patch_tmp4 = patch_tmp3(:, weightsNminus1_Nplus1_nxp_shrunken, :);
	    endwhile
	  else
	    nyp_shift = floor((weightsNminus1_Nplus1_nyp - weightsNminus1_Nplus1_nyp_shrunken)/2);
	    nxp_shift = floor((weightsNminus1_Nplus1_nxp - weightsNminus1_Nplus1_nxp_shrunken)/2);
	    patch_tmp3 = ...
		patchNminus1_Nplus1(nyp_shift+1:weightsNminus1_Nplus1_nyp_shrunken, ...
				    nxp_shift+1:weightsNminus1_Nplus1_nxp_shrunken, :);
	  endif  %% kf_preN_Nplus1_rank == 1

	  %% rescale patch
	  min_patch = min(patch_tmp3(:));
	  max_patch = max(patch_tmp3(:));
	  patch_tmp5 = ...
	      uint8((flipdim(patch_tmp3,1) - min_patch) * 255 / ...
		    (max_patch - min_patch + ((max_patch - min_patch)==0)));
	  
	  if plotNminus1_Nplus1_flag
	    imagesc(patch_tmp5); 
	    if weightsNminus1_N_nfp == 1
	      colormap(gray);
	    endif
	    box off
	    axis off
	    axis image
	    if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
	      [~, max_label] = max(squeeze(labelWeights_vals(:,kf_preN_Nplus1)));
	      text(weightsNminus1_Nplus1_nyp_shrunken/2, ...
		   -weightsNminus1_Nplus1_nxp_shrunken/6, ...
		   num2str(max_label-1), "color", [1 0 0]);
	    endif %% ~empty(labelWeights_vals)
	    %%drawnow;
	  endif %% plotNminus1_Nplus1_flag 

	  if isempty(weightsNminus1_Nplus1_array)
	    weightsNminus1_Nplus1_array = ...
		zeros(num_patchesN_Nplus1_rows*weightsNminus1_Nplus1_nyp_shrunken, ...
		      num_patchesN_Nplus1_cols*weightsNminus1_Nplus1_nxp_shrunken, ...
		      weightsNminus1_N_nfp);
	    weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1+1} = ...
		zeros(weightsNminus1_Nplus1_nyp_shrunken, ...
		      weightsNminus1_Nplus1_nxp_shrunken, ...
		      weightsNminus1_N_nfp, ...
		      num_patchesN_Nplus1);
	  endif
	  if  kf_preN_Nplus1_rank <= max_patches
	    col_ndx = 1 + mod(kf_preN_Nplus1_rank-1, num_patchesN_Nplus1_cols);
	    row_ndx = 1 + floor((kf_preN_Nplus1_rank-1) / num_patchesN_Nplus1_cols);
	    weightsNminus1_Nplus1_array((1+(row_ndx-1)*weightsNminus1_Nplus1_nyp_shrunken):...
					(row_ndx*weightsNminus1_Nplus1_nyp_shrunken), ...
					(1+(col_ndx-1)*weightsNminus1_Nplus1_nxp_shrunken): ...
					(col_ndx*weightsNminus1_Nplus1_nxp_shrunken),:) = ...
		patch_tmp5;
	  endif

	  %% set weightsN_Nplus1_vals to patch_tmp3
	  weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1+1}(:, :, :, kf_preN_Nplus1_rank) = ...
	      patch_tmp3;

	  %% Plot the average movie weights for each label %%
	  if ~isempty(labelWeights_vals) && ...
		plot_flag && ...
		i_checkpoint == max_checkpoint && ...
		i_layerN_Nplus1 == num_layersN_Nplus1_list-2 
	    if ~isempty(labeledWeightsNminus1_Nplus1{max_label})
	      labeledWeightsNminus1_Nplus1{max_label} = ...
		  labeledWeightsNminus1_Nplus1{max_label} + double(patch_tmp5);
	    else
	      labeledWeightsNminus1_Nplus1{max_label} = double(patch_tmp5);
	    endif
	  endif %%  ~isempty(labelWeights_vals) 

	endfor %% kf_preN_Nplus1_rank

	if plot_flag && ...
	      i_checkpoint == max_checkpoint && ...
	      i_layerN_Nplus1 == num_layersN_Nplus1_list-2 
	  saveas(weightsNminus1_Nplus1_fig, [weightsN_Nplus1_dir, filesep, weightsN_Nplus1_name, ".png"], "png");
	  if ~isempty(labelWeights_vals) 
	    for label = 1:size(labelWeights_vals,1)
	      labeledWeights_str = ...
		  ["labeledWeightsFig_", ...
		   weightsN_Nplus1_list{i_weightN_Nplus1,1}, weightsN_Nplus1_list{i_weightN_Nplus1,2}, ...
		   "_", num2str(label, "%d"), "_", ...
		   num2str(weightsN_Nplus1_time, "%08d")];
	      labeledWeights_fig = figure;
	      title(labeledWeights_str);
	      imagesc(squeeze(labeledWeightsNminus1_Nplus1{label}));
	      saveas(labeledWeights_fig,  [weightsN_Nplus1_dir, filesep, labeledWeights_str, ".png"], "png");
	    endfor %% label = 1:size(labelWeights_vals,1)
	  endif %%  ~isempty(labelWeights_vals) 
	endif
	imwrite(uint8(weightsNminus1_Nplus1_array), ...
		[weightsN_Nplus1_movie_dir, filesep, weightsN_Nplus1_name, ".png"], "png");
	

	%% make histogram of all weights
	if plot_flag && ...
	      i_checkpoint == max_checkpoint && ...
	      i_layerN_Nplus1 == num_layersN_Nplus1_list-2 
	  weightsNminus1_Nplus1_hist_fig = figure;
	  [weightsN_Nplus1_hist, weightsN_Nplus1_hist_bins] = ...
	      hist(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1+1}(:), 100);
	  bar(weightsN_Nplus1_hist_bins, log(weightsN_Nplus1_hist+1));
	  set(weightsNminus1_Nplus1_hist_fig, "name", ...
	      ["Hist_", ...
	       weightsN_Nplus1_list{i_weightN_Nplus1,1}, weightsN_Nplus1_list{i_weightN_Nplus1,2}, "_", ...
	       num2str(weightsN_Nplus1_time, "%08d")]);
	  saveas(weightsNminus1_Nplus1_hist_fig, ...
		 [weightsN_Nplus1_dir, filesep, "weightsN_Nplus1_hist_", ...
		  weightsN_Nplus1_list{i_weightN_Nplus1,1}, weightsN_Nplus1_list{i_weightN_Nplus1,2}, "_", ...
		  num2str(weightsN_Nplus1_time, "%08d")], "png");
	endif %% plotNminus1_Nplus1_flag

	%% plot average labelWeights for each label
	if ~isempty(labelWeights_vals) && ...
	      ~isempty(labelWeights_time) && ...
	      plot_flag && ...
	      i_checkpoint == max_checkpoint && ...
	      i_layerN_Nplus1 == num_layersN_Nplus1_list-2 

	  %% plot label weights as matrix of column vectors
	  ranked_labelWeights = labelWeights_vals(:, pre_hist_rank(1:num_patchesN_Nplus1));
	  [~, max_label] = max(ranked_labelWeights,[],1);
	  [max_label_sorted, max_label_ndx] = sort(max_label);
	  label_weights_fig = figure;
	  imagesc(ranked_labelWeights(:,max_label_ndx))
	  label_weights_str = ...
	      ["LabelWeights_", ...
	       weightsN_Nplus1_list{i_weightN_Nplus1,1}, labelWeights_list{i_weightN_Nplus1,2}, "_", ...
	       num2str(labelWeights_time, "%08d")];
	  %%title(label_weights_fig, label_weights_str);
	  figure(label_weights_fig); title(label_weights_str);
	  saveas(label_weights_fig, [weightsN_Nplus1_dir, filesep, label_weights_str, ".png"] , "png");
	endif  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

      endfor %% i_checkpoint

    endfor %% i_layerN_Nplus1

  endfor %% i_weightN_Nplus1
  
endif  %% plot_weights



