
%%clear all;
%%close all;
setenv("GNUTERM","X11")
if ismac
  workspace_path = "/Users/garkenyon/workspace";
  output_dir = "/Users/garkenyon/workspace/HyPerHLCA2/output_animal1200000_color_deep"; 
elseif isunix
  workspace_path = "/home/gkenyon/workspace";
  output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_12x12x512_lambda_05X2_color_deep"; 
  %%output_dir = "/nh/compneuro/Data/KITTI/LCA/2011_09_26_drive_0005_sync"; 
  %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_16x16x1024_Overlap_lambda_05X2"; 
  %%output_dir = "/nh/compneuro/Data/vine/LCA/detail/output_16x16x1024_overlap_lambda_05X2_errorthresh_005"; 
endif
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
last_checkpoint_ndx = 10000;
checkpoint_path = [output_dir, filesep, "Checkpoints", filesep,  "Checkpoint", num2str(last_checkpoint_ndx, "%i")]; %% "Last"];%%

%% plot Reconstructions
plot_Recon = true;
if plot_Recon
  num_Recon_default = 16;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Recon_list = ...
      {["a1_"], ["Retina"];
       ["a3_"], ["Ganglion"];
       ["a6_"], ["Recon"];
       ["a9_"], ["Recon2"];
       ["a12_"], ["ReconInfra"];
       ["a12_"], ["ReconInfra"]};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KITTI list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Recon_list = ...
%%      {["a1_"], ["LeftRetina"];
%%       ["a3_"], ["LeftGanglion"];
%%       ["a5_"], ["LeftRecon"];
%%       ["a7_"], ["RightRetina"];
%%       ["a9_"], ["RightGanglion"];
%%       ["a11_"], ["RightRecon"]};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  num_Recon_list = size(Recon_list,1);
  num_Recon_frames = repmat(num_Recon_default, 1, num_Recon_list);

%% list of layers to unwhiten
  unwhiten_list = zeros(num_Recon_list,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  unwhiten_list([2,3,5,6]) = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KITTI list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  unwhiten_list([2,3,5,6]) = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% list of layers to use as a normalization reference for unwhitening
%% default to self
  normalize_list = 1:num_Recon_list;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  normalize_list(3) = 2;
  normalize_list(5) = 2;
  normalize_list(6) = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KITTI list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  normalize_list(3) = 2;
%%  normalize_list(6) = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% list of (previous) layers to sum with current layer
%% default to empty
  sum_list = cell(num_Recon_list,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sum_list{6} = 3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %%keyboard;
  Recon_dir = [output_dir, filesep, "Recon"];
  mkdir(Recon_dir);
  
  %% parse center/surround pre-processing filters
  plot_DoG_kernel = 1;
  if plot_DoG_kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    blur_center_path = [checkpoint_path, filesep, "RetinaToBipolarCenter_W.pvp"];
    DoG_center_path = [checkpoint_path, filesep, "BipolarToGanglionCenter_W.pvp"];
    DoG_surround_path = [checkpoint_path, filesep, "BipolarToGanglionSurround_W.pvp"];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KITTI list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    blur_center_path = [checkpoint_path, filesep, "LeftRetinaToLeftBipolarCenter_W.pvp"];
%%    DoG_center_path = [checkpoint_path, filesep, "LeftBipolarToLeftGanglionCenter_W.pvp"];
%%    DoG_surround_path = [checkpoint_path, filesep, "LeftBipolarToLeftGanglionSurround_W.pvp"];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [blur_weights] = get_Blur_weights(blur_center_path);
    [DoG_weights] = get_DoG_weights(DoG_center_path, DoG_surround_path);
  endif

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
      error(["file does not exist: ", Recon_file]);
    endif
    Recon_fid(i_Recon) = fopen(Recon_file);
    Recon_hdr{i_Recon} = readpvpheader(Recon_fid(i_Recon));
    fclose(Recon_fid(i_Recon));
    tot_Recon_frames(i_Recon) = Recon_hdr{i_Recon}.nbands;
    %% TODO:: set num_Recon_frames_skip to the number of existing frames in recon dir
%%    num_Recon_frames(i_Recon) = tot_Recon_frames(i_Recon) - num_Recon_frames(i_Recon);
%%    if i_Recon == 2
%%      num_Recon_frames(i_Recon) = 4000;
%%    elseif i_Recon == 1
%%      num_Recon_frames(i_Recon) = 16;
%%    endif
    progress_step = ceil(tot_Recon_frames(i_Recon) / 10);
    [Recon_struct, Recon_hdr_tmp] = ...
	readpvpfile(Recon_file, ...
		    progress_step, ...
		    tot_Recon_frames(i_Recon), ... %% num_Recon_frames(i_Recon), ... %%
		    tot_Recon_frames(i_Recon)-num_Recon_frames(i_Recon)+1); %% 1); %% 
    Recon_fig(i_Recon) = figure;
    num_Recon_colors = Recon_hdr{i_Recon}.nf;
    mean_unwhitened_Recon{i_Recon,1} = zeros(num_Recon_colors,num_Recon_frames(i_Recon));
    std_unwhitened_Recon{i_Recon, 1} = ones(num_Recon_colors, num_Recon_frames(i_Recon));
    max_unwhitened_Recon{i_Recon, 1} = ones(num_Recon_colors, num_Recon_frames(i_Recon));
    min_unwhitened_Recon{i_Recon, 1} = zeros(num_Recon_colors,num_Recon_frames(i_Recon));
    Recon_vals{i_Recon} = cell(num_Recon_frames(i_Recon),1);
    Recon_times{i_Recon} = zeros(num_Recon_frames(i_Recon),1);
    if plot_DoG_kernel && unwhiten_list(i_Recon)
      unwhitened_Recon_fig(i_Recon) = figure;
      unwhitened_Recon_DoG{i_Recon} = cell(num_Recon_frames(i_Recon),1);
    endif
    for i_frame = 1 : num_Recon_frames(i_Recon)
      Recon_time{i_Recon}(i_frame) = Recon_struct{i_frame}.time;
      Recon_vals{i_Recon}{i_frame} = Recon_struct{i_frame}.values;
      figure(Recon_fig(i_Recon));
      Recon_fig_name{i_Recon} = Recon_list{i_Recon,2};
      num_sum_list = length(sum_list{i_Recon});
      for i_sum = 1 : num_sum_list
	sum_ndx = sum_list{i_Recon}(i_sum);
	Recon_vals{i_Recon}{i_frame} = Recon_vals{i_Recon}{i_frame} + ...
	    Recon_vals{sum_ndx}{i_frame};
	Recon_fig_name{i_Recon} = [Recon_fig_name{i_Recon}, "_", Recon_list{sum_ndx,2}];
      endfor %% i_sum
      mean_Recon_tmp = mean(Recon_vals{i_Recon}{i_frame}(:));
      std_Recon_tmp = std(Recon_vals{i_Recon}{i_frame}(:));
      Recon_mean(i_Recon) = Recon_mean(i_Recon) + mean_Recon_tmp;
      Recon_std(i_Recon) = Recon_std(i_Recon) + std_Recon_tmp;
      Recon_fig_name{i_Recon} = [Recon_fig_name{i_Recon}, "_", num2str(Recon_time{i_Recon}(i_frame), "%07d")];
      set(Recon_fig(i_Recon), "name", Recon_fig_name{i_Recon});
      imagesc(permute(Recon_vals{i_Recon}{i_frame},[2,1,3])); 
      if num_Recon_colors == 1
	colormap(gray); 
      endif
      box off; axis off; axis image;
      saveas(Recon_fig(i_Recon), ...
	     [Recon_dir, filesep, Recon_list{i_Recon,2}, "_", num2str(Recon_time{i_Recon}(i_frame), "%07d"), ".png"], "png");
      if plot_DoG_kernel && unwhiten_list(i_Recon)
	unwhitened_Recon_DoG{i_Recon}{i_frame} = zeros(size(permute(Recon_vals{i_Recon}{i_frame},[2,1,3])));
	for i_color = 1 : num_Recon_colors
	  tmp_Recon = ...
	      deconvolvemirrorbc(squeeze(Recon_vals{i_Recon}{i_frame}(:,:,i_color))', DoG_weights);
	  mean_unwhitened_Recon{i_Recon}(i_color, i_frame) = mean(tmp_Recon(:));
 	  std_unwhitened_Recon{i_Recon}(i_color, i_frame) = std(tmp_Recon(:));
	  j_frame = ceil(i_frame * tot_Recon_frames(normalize_list(i_Recon)) / tot_Recon_frames(i_Recon));
	  tmp_Recon = ...
	      (tmp_Recon - mean_unwhitened_Recon{i_Recon}(i_color, j_frame)) * ...
	      (std_unwhitened_Recon{normalize_list(i_Recon)}(i_color, j_frame) / ...
	       (std_unwhitened_Recon{i_Recon}(i_color, j_frame) + (std_unwhitened_Recon{i_Recon}(i_color, j_frame)==0))) + ...
	      mean_unwhitened_Recon{normalize_list(i_Recon)}(i_color, j_frame); 
	  max_unwhitened_Recon{i_Recon}(i_color, i_frame) = max(tmp_Recon(:));
	  min_unwhitened_Recon{i_Recon}(i_color, i_frame) = min(tmp_Recon(:));
	  [unwhitened_Recon_DoG{i_Recon}{i_frame}(:,:,i_color)] = tmp_Recon;
	endfor
	figure(unwhitened_Recon_fig(i_Recon));
	for i_sum = 1 : num_sum_list
	  sum_ndx = sum_list{i_Recon}(i_sum);
	  unwhitened_Recon_DoG{i_Recon}{i_frame} = unwhitened_Recon_DoG{i_Recon}{i_frame} + ...
	      unwhitened_Recon_DoG{sum_ndx}{i_frame};
	endfor %% i_sum
	set(unwhitened_Recon_fig(i_Recon), "name", ["unwhitened_", Recon_fig_name{i_Recon}]); 
	imagesc(squeeze(unwhitened_Recon_DoG{i_Recon}{i_frame})); 
	if num_Recon_colors == 1
	  colormap(gray); 
	endif
	box off; axis off; axis image;
	saveas(unwhitened_Recon_fig(i_Recon), ...
	       [Recon_dir, filesep, "unwhitened_", Recon_fig_name{i_Recon}, ".png"], "png");
	drawnow
      endif %% plot_DoG_kernel
    endfor   %% i_frame
    Recon_mean(i_Recon) = Recon_mean(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    Recon_std(i_Recon) = Recon_std(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    disp([ Recon_fig_name{i_Recon}, "_Recon_mean = ", num2str(Recon_mean(i_Recon)), " +/- ", num2str(Recon_std(i_Recon))]);
    
  endfor %% i_Recon
endif %% plot_Recon

%%keyboard;
plot_StatsProbe_vs_time = true;
if plot_StatsProbe_vs_time
  StatsProbe_plot_lines = 5000;
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KITTI list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  StatsProbe_list = ...
%%      {["LeftError"],["_Stats.txt"]; ...
%%       ["RightError"],["_Stats.txt"]; ...
%%       ["BinocularV1"],["_Stats.txt"]};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  StatsProbe_vs_time_dir = [output_dir, filesep, "StatsProbe_vs_time"];
  mkdir(StatsProbe_vs_time_dir);
  num_StatsProbe_list = size(StatsProbe_list,1);
  StatsProbe_sigma_flag = ones(1,num_StatsProbe_list);
  StatsProbe_sigma_flag([2,4,6]) = 0;
  StatsProbe_nnz_flag = ~StatsProbe_sigma_flag;
  for i_StatsProbe = 1 : num_StatsProbe_list
    StatsProbe_file = [output_dir, filesep, StatsProbe_list{i_StatsProbe,1}, StatsProbe_list{i_StatsProbe,2}]
    if ~exist(StatsProbe_file,"file")
      error(["StatsProbe_file does not exist: ", StatsProbe_file]);
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
    StatsProbe_vs_time_fig(i_StatsProbe) = figure;
    if StatsProbe_nnz_flag(i_StatsProbe)
      StatsProbe_vs_time_hndl = plot(StatsProbe_time_vals, StatsProbe_nnz_vals/StatsProbe_N); axis tight;
      axis tight
      set(StatsProbe_vs_time_fig(i_StatsProbe), "name", [StatsProbe_list{i_StatsProbe,1}, " nnz"]);
      saveas(StatsProbe_vs_time_fig(i_StatsProbe), ...
	     [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
	      "_nnz_vs_time_", num2str(StatsProbe_time_vals(end), "%07d")], "png");
    else
      StatsProbe_vs_time_hndl = plot(StatsProbe_time_vals, StatsProbe_sigma_vals); axis tight;
      axis tight
      set(StatsProbe_vs_time_fig(i_StatsProbe), "name", [StatsProbe_list{i_StatsProbe,1}, " sigma"]);
      saveas(StatsProbe_vs_time_fig(i_StatsProbe), ...
	     [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
	      "_sigma_vs_time_", num2str(StatsProbe_time_vals(end), "%07d")], "png");
    endif %% 
    drawnow;
  endfor %% i_StatsProbe
endif  %% plot_StatsProbe_vs_time

plot_Sparse = true;
if plot_Sparse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Sparse_list = ...
      {["a5_"], ["V1"]; ...
       ["a8_"], ["V2"]};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KITTI list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Sparse_list = ...
%%      {["a12_"], ["BinocularV1"]};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  num_Sparse_list = size(Sparse_list,1);
  Sparse_hdr = cell(num_Sparse_list,1);
  Sparse_hist_rank = cell(num_Sparse_list,1);
  Sparse_dir = [output_dir, filesep, "Sparse"];
  mkdir(Sparse_dir);
  for i_Sparse = 1 : num_Sparse_list
    Sparse_file = [output_dir, filesep, Sparse_list{i_Sparse,1}, Sparse_list{i_Sparse,2}, ".pvp"]
    if ~exist(Sparse_file, "file")
      error(["file does not exist: ", Sparse_file]);
    endif
    Sparse_fid = fopen(Sparse_file);
    Sparse_hdr{i_Sparse} = readpvpheader(Sparse_fid);
    fclose(Sparse_fid);
    tot_Sparse_frames = Sparse_hdr{i_Sparse}.nbands;
    num_Sparse = tot_Sparse_frames;
    progress_step = ceil(tot_Sparse_frames / 10);
    [Sparse_struct, Sparse_hdr_tmp] = ...
	readpvpfile(Sparse_file, progress_step, tot_Sparse_frames, tot_Sparse_frames-num_Sparse+1);
    nx_Sparse = Sparse_hdr{i_Sparse}.nx;
    ny_Sparse = Sparse_hdr{i_Sparse}.ny;
    nf_Sparse = Sparse_hdr{i_Sparse}.nf;
    n_Sparse = nx_Sparse * ny_Sparse * nf_Sparse;
    num_frames = size(Sparse_struct,1);
    Sparse_hist = zeros(nf_Sparse+1,1);
    Sparse_hist_edges = [0:1:nf_Sparse]+0.5;
    Sparse_current = zeros(n_Sparse,1);
    Sparse_abs_change = zeros(num_frames,1);
    Sparse_percent_change = zeros(num_frames,1);
    Sparse_current_active = 0;
    Sparse_tot_active = zeros(num_frames,1);
    Sparse_times = zeros(num_frames,1);
    for i_frame = 1 : 1 : num_frames
      Sparse_times(i_frame) = squeeze(Sparse_struct{i_frame}.time);
      Sparse_active_ndx = squeeze(Sparse_struct{i_frame}.values);
      Sparse_previous = Sparse_current;
      Sparse_current = full(sparse(Sparse_active_ndx+1,1,1,n_Sparse,1,n_Sparse));
      Sparse_abs_change(i_frame) = sum(Sparse_current(:) ~= Sparse_previous(:));
      Sparse_previous_active = Sparse_current_active;
      Sparse_current_active = nnz(Sparse_current(:));
      Sparse_tot_active(i_frame) = Sparse_current_active;
      Sparse_max_active = max(Sparse_current_active, Sparse_previous_active);
      Sparse_percent_change(i_frame) = ...
	  Sparse_abs_change(i_frame) / (Sparse_max_active + (Sparse_max_active==0));
      Sparse_active_kf = mod(Sparse_active_ndx, nf_Sparse) + 1;
      if Sparse_max_active > 0
	Sparse_hist_frame = histc(Sparse_active_kf, Sparse_hist_edges);
      else
	Sparse_hist_frame = zeros(nf_Sparse+1,1);
      endif
      Sparse_hist = Sparse_hist + Sparse_hist_frame;
    endfor %% i_frame
    Sparse_hist = Sparse_hist(1:nf_Sparse);
    Sparse_hist = Sparse_hist / (num_frames * nx_Sparse * ny_Sparse); %% (sum(Sparse_hist(:)) + (nnz(Sparse_hist)==0));
    [Sparse_hist_sorted, Sparse_hist_rank{i_Sparse}] = sort(Sparse_hist, 1, "descend");

    Sparse_hist_fig = figure;
    Sparse_hist_bins = 1:nf_Sparse;
    Sparse_hist_hndl = bar(Sparse_hist_bins, Sparse_hist_sorted); axis tight;
    set(Sparse_hist_fig, "name", ["Hist_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_frames), "%i")]);
    saveas(Sparse_hist_fig, ...
	   [Sparse_dir, filesep, ...
	    "Hist_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_frames), "%i")], "png");
    
    Sparse_abs_change_fig = figure;
    Sparse_abs_change_hndl = plot(Sparse_times, Sparse_abs_change); axis tight;
    set(Sparse_abs_change_fig, "name", ["abs_change_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_frames), "%i")]);
    saveas(Sparse_abs_change_fig, ...
	   [Sparse_dir, filesep, "abs_change_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_frames), "%i")], "png");
    
    Sparse_percent_change_fig = figure;
    Sparse_percent_change_hndl = plot(Sparse_times, Sparse_percent_change); axis tight;
    set(Sparse_percent_change_fig, ...
	"name", ["percent_change_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_frames), "%07d")]);
    saveas(Sparse_percent_change_fig, ...
	   [Sparse_dir, filesep, ...
	    "percent_change_", Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_frames), "%07d")], "png");
    
    Sparse_tot_active_fig = figure;
    Sparse_tot_active_hndl = plot(Sparse_times, Sparse_tot_active/n_Sparse); axis tight;
    set(Sparse_tot_active_fig, "name", ["tot_active_", Sparse_list{i_Sparse,2}, "_", ...
					num2str(Sparse_times(num_frames), "%07d")]);
    saveas(Sparse_tot_active_fig, ...
	   [Sparse_dir, filesep, "tot_active_", Sparse_list{i_Sparse,2}, "_", ...
	    num2str(Sparse_times(num_frames), "%07d")], "png");
    
    Sparse_mean_active = mean(Sparse_tot_active(:)/n_Sparse);
    disp([Sparse_list{i_Sparse,2}, "_", num2str(Sparse_times(num_frames), "%i"), ...
	  " mean_active = ", num2str(Sparse_mean_active)]);
  endfor  %% i_Sparse
endif %% plot_Sparse

  

plot_nonSparse = true;
if plot_nonSparse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  nonSparse_list = ...
      {["a4_"], ["Error"]; ...
       ["a7_"], ["Error2"]; ...
       ["a10_"], ["Error1_2"]};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KITTI list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  nonSparse_list = ...
%%      {["a4_"], ["LeftError"]; ...
%%       ["a10_"], ["RightError"]};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  num_nonSparse_list = size(nonSparse_list,1);

%% num frames to skip between stored frames, default is 
  nonSparse_skip = repmat(1, num_nonSparse_list, 1);
  nonSparse_skip(1) = 10;
  nonSparse_skip(2) = 10;
  nonSparse_skip(3) = 10;
  nonSparse_hdr = cell(num_nonSparse_list,1);
  nonSparse_dir = [output_dir, filesep, "nonSparse"];
  mkdir(nonSparse_dir);
  for i_nonSparse = 1 : num_nonSparse_list
    nonSparse_file = [output_dir, filesep, nonSparse_list{i_nonSparse,1}, nonSparse_list{i_nonSparse,2}, ".pvp"]
    if ~exist(nonSparse_file, "file")
      error(["file does not exist: ", nonSparse_file]);
    endif
    nonSparse_fid = fopen(nonSparse_file);
    nonSparse_hdr{i_nonSparse} = readpvpheader(nonSparse_fid);
    fclose(nonSparse_fid);
    tot_nonSparse_frames = nonSparse_hdr{i_nonSparse}.nbands;
	       
    num_nonSparse = tot_nonSparse_frames;
    progress_step = ceil(tot_nonSparse_frames / 10);
    [nonSparse_struct, nonSparse_hdr_tmp] = ...
	readpvpfile(nonSparse_file, progress_step, tot_nonSparse_frames, tot_nonSparse_frames-num_nonSparse+1, ...
		    nonSparse_skip(i_nonSparse));
    nx_nonSparse = nonSparse_hdr{i_nonSparse}.nx;
    ny_nonSparse = nonSparse_hdr{i_nonSparse}.ny;
    nf_nonSparse = nonSparse_hdr{i_nonSparse}.nf;
    n_nonSparse = nx_nonSparse * ny_nonSparse * nf_nonSparse;
    num_frames = size(nonSparse_struct,1);
    nonSparse_times = zeros(num_frames,1);
    nonSparse_RMS = zeros(num_frames,1);
    for i_frame = 1 : 1 : num_frames
      if ~isempty(nonSparse_struct{i_frame})
	nonSparse_times(i_frame) = squeeze(nonSparse_struct{i_frame}.time);
	nonSparse_vals = squeeze(nonSparse_struct{i_frame}.values);
	nonSparse_RMS(i_frame) = std(nonSparse_vals(:));
      else
	num_frames = i_frame - 1;
	nonSparse_times = nonSparse_times(1:num_frames);
	nonSparse_RMS = nonSparse_RMS(1:num_frames);
	break;
      endif
    endfor %% i_frame
    
    nonSparse_RMS_fig = figure;
    nonSparse_RMS_hndl = plot(nonSparse_times, nonSparse_RMS); axis tight;
    set(nonSparse_RMS_fig, "name", ["RMS_", nonSparse_list{i_nonSparse,2}, "_", num2str(nonSparse_times(num_frames), "%07d")]);
    saveas(nonSparse_RMS_fig, ...
	   [nonSparse_dir, filesep, ...
	    "RMS_", nonSparse_list{i_nonSparse,2}, "_", num2str(nonSparse_times(num_frames), "%07d")], "png");
    
    nonSparse_mean_active = median(nonSparse_RMS(:));
    disp([nonSparse_list{i_nonSparse,2}, "_", num2str(nonSparse_times(num_frames), "%i"), ...
	  " median RMS = ", num2str(nonSparse_mean_active)]);
  endfor  %% i_nonSparse
endif %% plot_nonSparse



plot_weights = true;
if plot_weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  weights_list = ...
      {["w5_"], ["V1ToError"]; ...
       ["w9_"], ["V2ToError2"]};
  pre_list = ...
      {["a5_"], ["V1"]; ...
       ["a8_"], ["V2"]};
  sparse_ndx = [1; 2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KITTI list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  weights_list = ...
%%      {["w10_"], ["BinocularV1ToLeftError"]; ...
%%       ["w13_"], ["BinocularV1ToRightError"]};
%%  pre_list = ...
%%      {["a12_"], ["BinocularV1"]; ...
%%       ["a12_"], ["BinocularV1"]};
%%  sparse_ndx = [1; 1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  num_weights_list = size(weights_list,1);
  weights_hdr = cell(num_weights_list,1);
  pre_hdr = cell(num_weights_list,1);
  weights_dir = [output_dir, filesep, "weights"];
  mkdir(weights_dir);
  for i_weights = 1 : num_weights_list
    weights_file = [output_dir, filesep, weights_list{i_weights,1}, weights_list{i_weights,2}, ".pvp"]
    if ~exist(weights_file, "file")
      error(["file does not exist: ", weights_file]);
    endif
    weights_fid = fopen(weights_file);
    weights_hdr{i_weights} = readpvpheader(weights_fid);    
    fclose(weights_fid);
    weights_filedata = dir(weights_file);
    weights_framesize = weights_hdr{i_weights}.recordsize*weights_hdr{i_weights}.numrecords+weights_hdr{i_weights}.headersize;
    tot_weights_frames = weights_filedata(1).bytes/weights_framesize;

    %%  
    i_pre = i_weights;
    pre_file = [output_dir, filesep, pre_list{i_pre,1}, pre_list{i_pre,2}, ".pvp"]
    if ~exist(pre_file, "file")
      error(["file does not exist: ", pre_file]);
    endif
    pre_fid = fopen(pre_file);
    pre_hdr{i_pre} = readpvpheader(pre_fid);
    fclose(pre_fid);

    num_weights = 1;
    progress_step = ceil(tot_weights_frames / 10);
    [weights_struct, weights_hdr_tmp] = ...
	readpvpfile(weights_file, progress_step, tot_weights_frames, tot_weights_frames-num_weights+1);
    i_frame = num_weights;
    i_arbor = 1;
    weight_vals = squeeze(weights_struct{i_frame}.values{i_arbor});
    weight_time = squeeze(weights_struct{i_frame}.time);
    if plot_Sparse
      pre_hist_rank = Sparse_hist_rank{sparse_ndx(i_weights)};
    else
      pre_hist_rank = (1:pre_hdr.nf);
    endif

    %% make tableau of all patches
    %%keyboard;
    i_patch = 1;
    num_weights_dims = ndims(weight_vals);
    num_patches = size(weight_vals, num_weights_dims);
    num_patches_rows = floor(sqrt(num_patches));
    num_patches_cols = ceil(num_patches / num_patches_rows);
    num_weights_colors = 1;
    if num_weights_dims == 4
      num_weights_colors = size(weight_vals,3);
    endif
    weights_fig = figure;
    set(weights_fig, "name", ["Weights_", weights_list{i_weights,2}, "_", num2str(weight_time, "%07d")]);
    for j_patch = 1  : num_patches
      i_patch = pre_hist_rank(j_patch);
      subplot(num_patches_rows, num_patches_cols, j_patch); 
      if num_weights_colors == 1
	patch_tmp = squeeze(weight_vals(:,:,i_patch));
      else
	patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
      endif
      patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
      min_patch = min(patch_tmp2(:));
      max_patch = max(patch_tmp2(:));
      patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
      patch_tmp2 = uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
      imagesc(patch_tmp2); 
      if num_weights_colors == 1
	colormap(gray);
      endif
      box off
      axis off
      axis image
      %%drawnow;
    endfor
    weights_dir = [output_dir, filesep, "weights"];
    mkdir(weights_dir);
    saveas(weights_fig, [weights_dir, filesep, "Weights_", weights_list{i_weights,2}, "_", num2str(weight_time, "%07d")], "png");


    %% make histogram of all weights
    weights_hist_fig = figure;
    [weights_hist, weights_hist_bins] = hist(weight_vals(:), 100);
    bar(weights_hist_bins, log(weights_hist+1));
    set(weights_hist_fig, "name", ["weights_Histogram_", weights_list{i_weights,2}, "_", num2str(weight_time, "%07d")]);
    saveas(weights_hist_fig, [weights_dir, filesep, "weights_hist_", num2str(weight_time)], "png");

  endfor %% i_weights
    
endif  %% plot_weights


plot_weights1_2 = true;
if plot_weights1_2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% deep list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% list of weights from layer2 to layer1
  weights1_2_list = ...
      {["w13_"], ["V2ToError1_2"]};
  post1_2_list = ...
      {["a5_"], ["V1"]};
  pre1_2_list = ...
      {["a8_"], ["V2"]};
  %% list of weights from layer1 to image
  weights0_1_list = ...
      {["w5_"], ["V1ToError"]};
  image_list = ...
      {["a1_"], ["Retina"]};
  %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
  sparse_ndx = [1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %% get image header (to get image dimensions)
  i_image = 1;
  image_file = [output_dir, filesep, image_list{i_image,1}, image_list{i_image,2}, ".pvp"]
  if ~exist(image_file, "file")
    error(["file does not exist: ", image_file]);
  endif
  image_fid = fopen(image_file);
  image_hdr = readpvpheader(image_fid);
  fclose(image_fid);

  num_weights1_2_list = size(weights1_2_list,1);
  weights1_2_hdr = cell(num_weights1_2_list,1);
  pre1_2_hdr = cell(num_weights1_2_list,1);
  post1_2_hdr = cell(num_weights1_2_list,1);

  weights1_2_dir = [output_dir, filesep, "weights1_2"];
  mkdir(weights1_2_dir);
  for i_weights1_2 = 1 : num_weights1_2_list

    %% get weight 2->1 file
    weights1_2_file = [output_dir, filesep, weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, ".pvp"]
    if ~exist(weights1_2_file, "file")
      error(["file does not exist: ", weights1_2_file]);
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

    %% get weight 1->0 file
    i_weights0_1 = i_weights1_2;
    weights0_1_file = [output_dir, filesep, weights0_1_list{i_weights0_1,1}, weights0_1_list{i_weights0_1,2}, ".pvp"]
    if ~exist(weights0_1_file, "file")
      error(["file does not exist: ", weights0_1_file]);
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

    %% get pre header (to get pre layer dimensions)
    i_pre1_2 = i_weights1_2;
    pre1_2_file = [output_dir, filesep, pre1_2_list{i_pre1_2,1}, pre1_2_list{i_pre1_2,2}, ".pvp"]
    if ~exist(pre1_2_file, "file")
      error(["file does not exist: ", pre1_2_file]);
    endif
    pre1_2_fid = fopen(pre1_2_file);
    pre1_2_hdr{i_pre1_2} = readpvpheader(pre1_2_fid);
    fclose(pre1_2_fid);

    %% get post header (to get post layer dimensions)
    i_post1_2 = i_weights1_2;
    post1_2_file = [output_dir, filesep, post1_2_list{i_post1_2,1}, post1_2_list{i_post1_2,2}, ".pvp"]
    if ~exist(post1_2_file, "file")
      error(["file does not exist: ", post1_2_file]);
    endif
    post1_2_fid = fopen(post1_2_file);
    post1_2_hdr{i_post1_2} = readpvpheader(post1_2_fid);
    fclose(post1_2_fid);
    post1_2_nf = post1_2_hdr{i_post1_2}.nf;

    %% read 2 -> 1 weights
    num_weights1_2 = 1;
    progress_step = ceil(tot_weights1_2_frames / 10);
    [weights1_2_struct, weights1_2_hdr_tmp] = ...
	readpvpfile(weights1_2_file, progress_step, tot_weights1_2_frames, tot_weights1_2_frames-num_weights1_2+1);
    i_frame = num_weights1_2;
    i_arbor = 1;
    weights1_2_vals = squeeze(weights1_2_struct{i_frame}.values{i_arbor});
    weights1_2_time = squeeze(weights1_2_struct{i_frame}.time);
 
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
   if plot_Sparse
      pre_hist_rank = Sparse_hist_rank{sparse_ndx(i_weights1_2)};
    else
      pre_hist_rank = (1:pre1_2_hdr{i_pre1_2}.nf);
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
    num_patches1_2 = size(weights1_2_vals, num_weights1_2_dims);
    %% algorithms assumes weights1_2 are one to many
    num_patches_rows = floor(sqrt(num_patches1_2));
    num_patches_cols = ceil(num_patches1_2 / num_patches_rows);
    %% for one to many connections: dimensions of weights1_2 are:
    %% weights1_2(nxp, nyp, nf_post, nf_pre)
    weights1_2_fig = figure;
    set(weights1_2_fig, "name", ["Weights1_2_", weights1_2_list{i_weights1_2,2}, "_", num2str(weights1_2_time, "%07d")]);

    for kf_pre1_2_rank = 1  : num_patches1_2
      kf_pre1_2 = pre_hist_rank(kf_pre1_2_rank);
      subplot(num_patches_rows, num_patches_cols, kf_pre1_2_rank); 
      if ndims(weights1_2_vals) == 3
	patch1_2_tmp = squeeze(weights1_2_vals(:,:,kf_pre1_2));
	patch1_2_tmp = repmat(patch1_2_tmp, [1,1,1]);
      else
	patch1_2_tmp = squeeze(weights1_2_vals(:,:,:,kf_pre1_2));
      endif
      %% patch1_2_array stores the sum over all post layer 1 neurons, weighted by weights1_2, 
      %% of image patches for each columun of weights0_1 for pre layer 2 neuron kf_pre
      patch1_2_array = cell(size(weights1_2_vals,1),size(weights1_2_vals,2));
      %% patch1_2 stores the complete image patch of the layer 2 neuron kf_pre
      patch1_2 = zeros(weights0_2_nyp, weights0_2_nxp, weights0_1_nfp);
      %% loop over weights1_2 rows and columns
      for weights1_2_patch_row = 1 : weights1_2_nyp
	for weights1_2_patch_col = 1 : weights1_2_nxp
	  patch1_2_array{weights1_2_patch_row, weights1_2_patch_col} = ...
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
	    patch1_2_array{weights1_2_patch_row, weights1_2_patch_col} = ...
		patch1_2_array{weights1_2_patch_row, weights1_2_patch_col} + ...
		patch1_2_weight .* ...
		weights0_1_patch;
	  endfor %% kf_post1_2
	  row_start = 1+image2post_ny_ratio*(weights1_2_patch_row-1);
	  row_end = image2post_ny_ratio*(weights1_2_patch_row-1)+weights0_1_nyp;
	  col_start = 1+image2post_nx_ratio*(weights1_2_patch_col-1);
	  col_end = image2post_nx_ratio*(weights1_2_patch_col-1)+weights0_1_nxp;
	  patch1_2(row_start:row_end, col_start:col_end, :) = ...
	      patch1_2_array{weights1_2_patch_row, weights1_2_patch_col};
	endfor %% weights1_2_patch_col
      endfor %% weights1_2_patch_row
      min_patch = min(patch1_2(:));
      max_patch = max(patch1_2(:));
      patch_tmp2 = (patch1_2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
      patch_tmp2 = uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
      imagesc(patch_tmp2); 
      if weights0_1_nfp == 1
	colormap(gray);
      endif
      box off
      axis off
      axis image
      %%drawnow;

    endfor %% kf_pre1_2_ank
    saveas(weights1_2_fig, [weights1_2_dir, filesep, "Weights1_2_", weights1_2_list{i_weights1_2,2}, "_", ...
			    num2str(weights1_2_time, "%07d")], "png");


    %% make histogram of all weights
    weights1_2_hist_fig = figure;
    [weights1_2_hist, weights1_2_hist_bins] = hist(weights1_2_vals(:), 100);
    bar(weights1_2_hist_bins, log(weights1_2_hist+1));
    set(weights1_2_hist_fig, "name", ["weights1_2_Histogram_", weights1_2_list{i_weights1_2,2}, "_", ...
				      num2str(weights1_2_time, "%07d")]);
    saveas(weights1_2_hist_fig, [weights1_2_dir, filesep, "weights1_2_hist_", weights1_2_list{i_weights1_2,2}, "_", ...
				      num2str(weights1_2_time)], "png");

  endfor %% i_weights
    
endif  %% plot_weights