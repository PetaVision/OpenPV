
clear all;
close all;
setenv("GNUTERM","X11")
if ismac
  workspace_path = "/Users/garkenyon/workspace";
  output_dir = "/Users/garkenyon/workspace/HyPerHLCA2/output_animal1200000_color_deep"; 
  frame_duration = 1000;
elseif isunix
  workspace_path = "/home/gkenyon/workspace";
  output_dir = "/nh/compneuro/Data/KITTI/LCA/2011_09_26_drive_0002_sync"; 
  %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_16x16x1024_Overlap_lambda_05X2"; 
  %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_31/output_12x12x1024_lambda_05X2_color_deep"; 
  frame_duration = 5000;
endif
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
last_checkpoint_ndx = 1130000;
checkpoint_path = [output_dir, filesep, "Checkpoints", filesep,  "Checkpoint", num2str(last_checkpoint_ndx, "%i")];
max_history = 140000;

%% plot Reconstructions
plot_Recon = 0;
if plot_Recon
  num_recon = 32;
  Recon_list = ...
      {["a1_"], ["LeftRetina"];
       ["a3_"], ["LeftGanglion"];
       ["a5_"], ["LeftRecon"];
       ["a7_"], ["RightRetina"];
       ["a9_"], ["RightGanglion"];
       ["a11_"], ["RightRecon"]};
  num_Recon_list = size(Recon_list,1);
  unwhiten_list = zeros(num_Recon_list,1);
  unwhiten_list([2,3,5,6]) = 1;
  normalize_list = 1:num_Recon_list;
  normalize_list(3) = 2;
  normalize_list(6) = 5;
  %%keyboard;
  Recon_dir = [output_dir, filesep, "Recon"];
  mkdir(Recon_dir);
  
  %% parse center/surround pre-processing filters
  plot_DoG_kernel = 0;
  if plot_DoG_kernel
    blur_center_path = [checkpoint_path, filesep, "LeftRetinaToLeftBipolarCenter_W.pvp"];
    [blur_weights] = get_Blur_weights(blur_center_path);
    DoG_center_path = [checkpoint_path, filesep, "LeftBipolarToLeftGanglionCenter_W.pvp"];
    DoG_surround_path = [checkpoint_path, filesep, "LeftBipolarToLeftGanglionSurround_W.pvp"];
    [DoG_weights] = get_DoG_weights(DoG_center_path, DoG_surround_path);
  endif

  %%keyboard;
  num_Recon_frames = zeros(num_Recon_list,1);
  Recon_hdr = cell(num_Recon_list,1);
  Recon_fig = zeros(num_Recon_list,1);
  unwhitened_Recon_fig = zeros(num_Recon_list,1);
  Recon_mean = zeros(num_Recon_list, 1);
  Recon_std = zeros(num_Recon_list, 1);
  mean_unwhitened_Recon = cell(num_Recon_list, num_recon);
  std_unwhitened_Recon = cell(num_Recon_list, num_recon);
  max_unwhitened_Recon = cell(num_Recon_list, num_recon);
  min_unwhitened_Recon = cell(num_Recon_list, num_recon);
  for i_Recon = 1 : num_Recon_list
    Recon_file = [output_dir, filesep, Recon_list{i_Recon,1}, Recon_list{i_Recon,2}, ".pvp"]
    if ~exist(Recon_file, "file")
      error(["file does not exist: ", Recon_file]);
    endif
    Recon_fid(i_Recon) = fopen(Recon_file);
    Recon_hdr{i_Recon} = readpvpheader(Recon_fid(i_Recon));
    fclose(Recon_fid(i_Recon));
    num_Recon_frames(i_Recon) = Recon_hdr{i_Recon}.nbands;
    progress_step = ceil(num_Recon_frames(i_Recon) / 10);
    [Recon_struct, Recon_hdr_tmp] = ...
	readpvpfile(Recon_file, progress_step, num_Recon_frames(i_Recon), num_Recon_frames(i_Recon)-num_recon+1);
    Recon_fig(i_Recon) = figure;
    num_Recon_colors = Recon_hdr{i_Recon}.nf;
    if plot_DoG_kernel
      unwhitened_Recon_fig(i_Recon) = figure;
    endif
    for i_frame = 1 : num_recon
      Recon_time = Recon_struct{i_frame}.time;
      Recon_vals = Recon_struct{i_frame}.values;
      mean_Recon_tmp = mean(Recon_vals(:));
      std_Recon_tmp = std(Recon_vals(:));
      Recon_mean(i_Recon) = Recon_mean(i_Recon) + mean_Recon_tmp;
      Recon_std(i_Recon) = Recon_std(i_Recon) + std_Recon_tmp;
      figure(Recon_fig(i_Recon));
      set(Recon_fig(i_Recon), "name", [Recon_list{i_Recon,2}, "_", num2str(Recon_time, "%0d")]);
      imagesc(permute(Recon_vals,[2,1,3])); 
      mean_unwhitened_Recon{i_Recon, i_frame} = zeros(num_Recon_colors,1);
      std_unwhitened_Recon{i_Recon, i_frame} = ones(num_Recon_colors,1);
      max_unwhitened_Recon{i_Recon, i_frame} = ones(num_Recon_colors,1);
      min_unwhitened_Recon{i_Recon, i_frame} = zeros(num_Recon_colors,1);
      if num_Recon_colors == 1
	colormap(gray); 
      endif
      box off; axis off; axis image;
      saveas(Recon_fig(i_Recon), [Recon_dir, filesep, Recon_list{i_Recon,2}, "_", num2str(Recon_time, "%0d")], "png");
      if plot_DoG_kernel && unwhiten_list(i_Recon)
	unwhitened_Recon_DoG = zeros(size(permute(Recon_vals,[2,1,3])));
	for i_color = 1 : num_Recon_colors
	  tmp_Recon = ...
	      deconvolvemirrorbc(squeeze(Recon_vals(:,:,i_color))', DoG_weights);
	  mean_unwhitened_Recon{i_Recon, i_frame}(i_color) = mean(tmp_Recon(:));
 	  std_unwhitened_Recon{i_Recon, i_frame}(i_color) = std(tmp_Recon(:));
	  tmp_Recon = ...
	      (tmp_Recon - mean_Recon_tmp) * ...
	      (std_unwhitened_Recon{normalize_list(i_Recon), i_frame}(i_color) / std_Recon_tmp) + ...
	      mean_unwhitened_Recon{normalize_list(i_Recon), i_frame}(i_color); 
	  max_unwhitened_Recon(i_color) = max(tmp_Recon(:));
	  min_unwhitened_Recon(i_color) = min(tmp_Recon(:));
	  [unwhitened_Recon_DoG(:,:,i_color)] = tmp_Recon;
	endfor
	figure(unwhitened_Recon_fig(i_Recon));
	set(unwhitened_Recon_fig(i_Recon), "name", ["unwhitened ", Recon_list{i_Recon,2}, " ", num2str(Recon_time, "%0d")]);
	imagesc(squeeze(unwhitened_Recon_DoG)); 
	if num_Recon_colors == 1
	  colormap(gray); 
	endif
	box off; axis off; axis image;
	saveas(unwhitened_Recon_fig(i_Recon), ...
	       [Recon_dir, filesep, "unwhitened_", Recon_list{i_Recon,2}, "_", num2str(Recon_time, "%0d")], "png");
	drawnow
      endif %% plot_DoG_kernel
    endfor   %% i_frame
    Recon_mean(i_Recon) = Recon_mean(i_Recon) / (num_recon + (num_recon == 0));
    Recon_std(i_Recon) = Recon_std(i_Recon) / (num_recon + (num_recon == 0));
    disp(["Recon_mean = ", num2str(Recon_mean(i_Recon)), " +/- ", num2str(Recon_std(i_Recon))]);
    
  endfor %% i_Recon
endif %% plot_Recon

%%keyboard;
plot_StatsProbe_vs_time = 1;
if plot_StatsProbe_vs_time
  StatsProbe_plot_lines = 100000;
  StatsProbe_list = ...
      {["LeftError"],["_Stats.txt"]; ...
       ["RightError"],["_Stats.txt"]; ...
       ["BinocularV1"],["_Stats.txt"]};
  StatsProbe_vs_time_dir = [output_dir, filesep, "StatsProbe_vs_time"];
  mkdir(StatsProbe_vs_time_dir);
  num_StatsProbe_list = size(StatsProbe_list,1);
  StatsProbe_sigma_flag = ones(1,num_StatsProbe_list);
  StatsProbe_sigma_flag([3]) = 0;
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
	      "_nnz_vs_time_", num2str(StatsProbe_time_vals(end), "%i")], "png");
    else
      StatsProbe_vs_time_hndl = plot(StatsProbe_time_vals, StatsProbe_sigma_vals); axis tight;
      axis tight
      set(StatsProbe_vs_time_fig(i_StatsProbe), "name", [StatsProbe_list{i_StatsProbe,1}, " sigma"]);
      saveas(StatsProbe_vs_time_fig(i_StatsProbe), ...
	     [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
	      "_sigma_vs_time_", num2str(StatsProbe_time_vals(end), "%i")], "png");
    endif %% 
    drawnow;
  endfor %% i_StatsProbe
endif  %% plot_StatsProbe_vs_time
