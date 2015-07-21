
clear all;
close all;
%setenv("GNUTERM","X11")

workspace_path = "/home/slundquist/workspace";
output_dir = "/nh/compneuro/Data/Depth/LCA/depth/"; 
%output_dir = "/nh/compneuro/Data/Depth/LCA/arbortest/"; 

addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
addpath([workspace_path, filesep, "/PetaVision/mlab/HyPerLCA"]);
last_checkpoint_ndx = 2000;
checkpoint_path = [output_dir, filesep, "Checkpoints", filesep,  "Checkpoint", num2str(last_checkpoint_ndx, '%i')]; %% 
max_history = 196000;
numarbors = 1;

%%keyboard;
plot_StatsProbe_vs_time = 1;
if plot_StatsProbe_vs_time
  first_StatsProbe_line = 1; %%max([(last_StatsProbe_line - StatsProbe_plot_lines), 1]);
  StatsProbe_plot_lines = 1999;
%%  StatsProbe_list = ...
%%      {["Error"],["_Stats.txt"]; ...
%%       ["V1"],["_Stats.txt"]};
  StatsProbe_list = ...
       {["BinocularV1S1"],["_Stats.txt"]; ...
       ["BinocularV1S2"],["_Stats.txt"]; ...
       ["LeftError1"],["_Stats.txt"]; ...
       ["LeftError2"],["_Stats.txt"]; ...
       ["RightError1"],["_Stats.txt"]; ...
       ["RightError2"],["_Stats.txt"]; ...
       ["LeftDepthError"],["_Stats.txt"]; ...
       ["RightDepthError"],["_Stats.txt"]; ...
       ["PosError"],["_Stats.txt"]; ...
       };
  StatsProbe_vs_time_dir = [output_dir, filesep, "StatsProbe_vs_time"];
  mkdir(StatsProbe_vs_time_dir);
  num_StatsProbe_list = size(StatsProbe_list,1);
  StatsProbe_sigma_flag = ones(1,num_StatsProbe_list);
  StatsProbe_sigma_flag([1]) = 0;
  StatsProbe_sigma_flag([2]) = 0;
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
    StatsProbe_time_vals = [];
    StatsProbe_sigma_vals = [];
    StatsProbe_nnz_vals = [];
    skip_StatsProbe_line = 1; %%2000 per time update
    last_StatsProbe_line = StatsProbe_plot_lines; %% StatsProbe_num_lines - 2;
    num_lines = floor((last_StatsProbe_line - first_StatsProbe_line)/ skip_StatsProbe_line);

    %StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    %StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);
    %StatsProbe_time_vals = zeros(1,StatsProbe_plot_lines);

    for i_line = 1:first_StatsProbe_line-1
      StatsProbe_line = fgets(StatsProbe_fid);
    endfor
    %% extract N
    StatsProbe_N_ndx1 = strfind(StatsProbe_line, "N==");
    StatsProbe_N_ndx2 = strfind(StatsProbe_line, "Total==");
    StatsProbe_N_str = StatsProbe_line(StatsProbe_N_ndx1+3:StatsProbe_N_ndx2-2);
    StatsProbe_N = str2num(StatsProbe_N_str);
    for i_line = 1:num_lines
      %Skip lines based on how many was skipped
      for s_line = 1:skip_StatsProbe_line
         StatsProbe_line = fgets(StatsProbe_fid);
      endfor
      %% extract time
      StatsProbe_time_ndx1 = strfind(StatsProbe_line, "t==");
      StatsProbe_time_ndx2 = strfind(StatsProbe_line, "N==");
      StatsProbe_time_str = StatsProbe_line(StatsProbe_time_ndx1+3:StatsProbe_time_ndx2-2);
      StatsProbe_time_vals(i_line) = str2num(StatsProbe_time_str);
      %% extract sigma
      StatsProbe_sigma_ndx1 = strfind(StatsProbe_line, "sigma==");
      StatsProbe_sigma_ndx2 = strfind(StatsProbe_line, "nnz==");
      StatsProbe_sigma_str = StatsProbe_line(StatsProbe_sigma_ndx1+7:StatsProbe_sigma_ndx2-2);
      StatsProbe_sigma_vals(i_line) = str2num(StatsProbe_sigma_str);
      %% extract nnz
      StatsProbe_nnz_ndx1 = strfind(StatsProbe_line, "nnz==");
      StatsProbe_nnz_ndx2 = length(StatsProbe_line); 
      StatsProbe_nnz_str = StatsProbe_line(StatsProbe_nnz_ndx1+5:StatsProbe_nnz_ndx2-1);
      StatsProbe_nnz_vals(i_line) = str2num(StatsProbe_nnz_str);
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

%Prints weights from checkpoints
plot_Sparse = 1;
if plot_Sparse 
    Sparse_list = ...
       {["a42_"], ["BinocularV1S1"]; ...
        ["a43_"], ["BinocularV1S2"]; ...
        };

    load_Sparse_flag = 0;
    plot_Sparse_flag = 1;
    fraction_Sparse_frames_read = 1;
    min_Sparse_skip = 1;
    fraction_Sparse_progress = 1;
    num_procs = nproc;

    [Sparse_hdr, ...
          Sparse_hist_rank_array, ...
          Sparse_times_array, ...
          Sparse_percent_active_array, ...
          Sparse_percent_change_array, ...
          Sparse_std_array] = ...
          analyzeSparsePVP(Sparse_list, ...
                output_dir, ...
                load_Sparse_flag, ...
                plot_Sparse_flag, ...
                fraction_Sparse_frames_read, ...
                min_Sparse_skip, ...
                fraction_Sparse_progress, ...
                num_procs);
endif

plot_weights = 1;
if plot_weights
   weights_list = ...
       { ...
        ["BinocularV1S1ToLeftError1_W"]; ...
        ["BinocularV1S2ToLeftError2_W"]; ...
        ["BinocularV1S1ToRightError1_W"]; ...
        ["BinocularV1S2ToRightError2_W"]; ...
        ["BinocularV1S1ToLeftDepthError_W"]; ...
        ["BinocularV1S2ToLeftDepthError_W"]; ...
        ["BinocularV1S1ToRightDepthError_W"]; ...
        ["BinocularV1S2ToRightDepthError_W"]; ...
        ["BinocularV1S1ToPosError_W"]; ...
        ["BinocularV1S2ToPosError_W"]; ...
        };
   pre_list = ...
       { ...
        ["BinocularV1S1_A"]; ...
        ["BinocularV1S2_A"]; ...
        ["BinocularV1S1_A"]; ...
        ["BinocularV1S2_A"]; ...
        ["BinocularV1S1_A"]; ...
        ["BinocularV1S2_A"]; ...
        ["BinocularV1S1_A"]; ...
        ["BinocularV1S2_A"]; ...
        ["BinocularV1S1_A"]; ...
        ["BinocularV1S2_A"]; ...
        };
   sparse_ndx = ...
        {   ...
        1;  ...
        2;  ...
        1;  ...
        2;  ...
        1;  ...
        2;  ...
        1;  ...
        2;  ...
        1;  ...
        2;  ...
        };
   num_weights_list = size(weights_list, 1);
   weights_hdr = cell(num_weights_list, 1);
   pre_hdr = cell(num_weights_list, 1);
   weights_dir = [output_dir, filesep, "weights"];
   mkdir(weights_dir);
   for i_weights = 1:num_weights_list
      weights_file = [checkpoint_path, filesep, weights_list{i_weights}, '.pvp'];
      if ~exist(weights_file, "file")
        error(["file does not exist: ", weights_file]);
      endif
      i_pre = i_weights;
      pre_file = [checkpoint_path, filesep, pre_list{i_pre}, '.pvp'];
      if ~exist(pre_file, "file")
        error(["file does not exist: ", pre_file]);
      endif
      pre_fid = fopen(pre_file);
      pre_hdr{i_pre} = readpvpheader(pre_fid);
      fclose(pre_fid);
      [weights_struct, weights_hdr_tmp] = ...
      readpvpfile(weights_file);
      %i_arbor = 1;
      for i_arbor = 1:numarbors
         i_frame = 1;
         if weights_hdr_tmp.nfp == 1
            weight_vals = squeeze(weights_struct{i_frame}.values{i_arbor});
         else
            weight_vals = weights_struct{i_frame}.values{i_arbor};
            %stepval = 1/weights_hdr_tmp.nfp;
            %rangeval = stepval/2;
            %[maxmat, idx] = max(weight_vals, [], 3);
            %idx = squeeze(idx);
            %weight_vals = (idx - 1) * stepval + rangeval;
         endif
         weight_time = squeeze(weights_struct{i_frame}.time);
         if plot_Sparse
           pre_hist_rank = Sparse_hist_rank_array{sparse_ndx{i_weights}};
         else
           pre_hist_rank = (1:pre_hdr{1}.nf);
         endif
       %% make tableau of all patches
       %%keyboard;
       i_patch = 1;
       num_weights_dims = ndims(weight_vals);
       %num_patches = size(weight_vals, num_weights_dims);
       num_patches = 64;
       num_patches_rows = floor(sqrt(num_patches));
       num_patches_cols = ceil(num_patches / num_patches_rows);
       num_weights_colors = 1;
       if num_weights_dims == 4
         num_weights_colors = size(weight_vals,3);
       endif
       weights_fig = figure;
       set(weights_fig, "name", ["Weights_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)]);
       for j_patch = 1  : num_patches
         i_patch = pre_hist_rank(j_patch);
         subplot(num_patches_rows, num_patches_cols, j_patch);

         if num_weights_colors == 1 patch_tmp = squeeze(weight_vals(:,:,i_patch));
         else patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
         endif
         if weights_hdr_tmp.nfp == 1
            patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
            min_patch = min(patch_tmp2(:));
            max_patch = max(patch_tmp2(:));
            patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
            patch_tmp2 = uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
            imagesc(patch_tmp2); 
            colormap(gray);
            box off
            axis off
            axis image
            %%drawnow;
         else
            plotme = squeeze(weight_vals(:, :, :, i_patch)); 
            assert(length(size(plotme)) == 2);
            [nf, temp] = size(plotme);
            plotme = repmat(plotme, 1, nf);
            
            box off
            axis on
            axis nolabel
            imagesc(plotme);
            colormap(gray);
            axis image
         endif
       endfor
       weights_dir = [output_dir, filesep, "weights"];
       mkdir(weights_dir);
       saveas(weights_fig, [weights_dir, filesep, "Weights_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)], "png");
       %% make histogram of all weights
       weights_hist_fig = figure;
       [weights_hist, weights_hist_bins] = hist(weight_vals(:), 100);
       bar(weights_hist_bins, log(weights_hist+1));
       set(weights_hist_fig, "name", ["weights_Histogram_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)]);
       saveas(weights_hist_fig, [weights_dir, filesep, "weights_hist_", weights_list{i_weights}, "_", num2str(i_arbor), "_", num2str(weight_time)], "png");
    endfor %% i_arbors
  endfor %% i_weights
endif  %% plot_weights

%% plot Reconstructions
plot_Recon = 0;
if plot_Recon
  num_Recon_default = 197;
%%  Recon_list = ...
%%      {["a3_"], ["Ganglion"]};
%%       {["a4_"], ["Recon"]};
  Recon_list = ...
      {["LeftRetina_A"];
       ["LeftGanglion_A"];
       ["LeftRecon_A"];
       %["LeftDepthRecon_A"];
       ["RightRetina_A"];
       ["RightGanglion_A"];
       ["RightRecon_A"];
       %["RightDepthRecon_A"];
       };
  num_Recon_list = size(Recon_list,1);
  num_Recon_frames = repmat(num_Recon_default, 1, num_Recon_list);
  unwhiten_list = zeros(num_Recon_list,1);
%%  unwhiten_list([2,3,5,6]) = 1;
  unwhiten_list([1]) = 1;
  normalize_list = 1:num_Recon_list;
%%  normalize_list(2) = 1;
%%  normalize_list(3) = 2;
%%  normalize_list(6) = 5;
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
  Recon_hdr = cell(num_Recon_list,1);
  Recon_fig = zeros(num_Recon_list,1);
  unwhitened_Recon_fig = zeros(num_Recon_list,1);
  Recon_mean = zeros(num_Recon_list, 1);
  Recon_std = zeros(num_Recon_list, 1);
  mean_unwhitened_Recon = cell(num_Recon_list,1);
  std_unwhitened_Recon = cell(num_Recon_list, 1);
  max_unwhitened_Recon = cell(num_Recon_list, 1);
  min_unwhitened_Recon = cell(num_Recon_list, 1);
  for i_Recon = 1 : num_Recon_list
    %Recon_file = [output_dir, filesep, Recon_list{i_Recon,1}, Recon_list{i_Recon,2}, ".pvp"]
    Recon_file = [checkpoint_path, filesep, Recon_list{i_Recon}, ".pvp"]
    if ~exist(Recon_file, "file")
     error(["file does not exist: ", Recon_file]);
    endif
    Recon_fid(i_Recon) = fopen(Recon_file);
    Recon_hdr{i_Recon} = readpvpheader(Recon_fid(i_Recon));
    fclose(Recon_fid(i_Recon));
    tot_Recon_frames = Recon_hdr{i_Recon}.nbands;
    %% TODO:: set num_Recon_frames_skip to the number of existing frames in recon dir
%%    num_Recon_frames(i_Recon) = tot_Recon_frames(i_Recon) - num_Recon_frames(i_Recon);
%%    if i_Recon == 2
%%      num_Recon_frames(i_Recon) = 4000;
%%    elseif i_Recon == 1
%%      num_Recon_frames(i_Recon) = 16;
%%    endif
    progress_step = ceil(tot_Recon_frames / 10);
    [Recon_struct, Recon_hdr_tmp] = ...
	readpvpfile(Recon_file, ...
		    progress_step);
		    %tot_Recon_frames, ... %% num_Recon_frames(i_Recon), ... %%
		    %tot_Recon_frames-num_Recon_frames(i_Recon)+1); %% 1); %% 
    Recon_fig(i_Recon) = figure;
    num_Recon_colors = Recon_hdr{i_Recon}.nf;
    if plot_DoG_kernel
      unwhitened_Recon_fig(i_Recon) = figure;
    endif
    mean_unwhitened_Recon{i_Recon,1} = zeros(num_Recon_colors,num_Recon_frames(i_Recon));
    std_unwhitened_Recon{i_Recon, 1} = ones(num_Recon_colors, num_Recon_frames(i_Recon));
    max_unwhitened_Recon{i_Recon, 1} = ones(num_Recon_colors, num_Recon_frames(i_Recon));
    min_unwhitened_Recon{i_Recon, 1} = zeros(num_Recon_colors,num_Recon_frames(i_Recon));

    i_frame = 1;
    %for i_frame = 1 : num_Recon_frames(i_Recon)
      Recon_time = Recon_struct{i_frame}.time;
      Recon_vals = Recon_struct{i_frame}.values;
      mean_Recon_tmp = mean(Recon_vals(:));
      std_Recon_tmp = std(Recon_vals(:));
      Recon_mean(i_Recon) = Recon_mean(i_Recon) + mean_Recon_tmp;
      Recon_std(i_Recon) = Recon_std(i_Recon) + std_Recon_tmp;
      figure(Recon_fig(i_Recon));
      set(Recon_fig(i_Recon), "name", [Recon_list{i_Recon}, "_", num2str(i_frame, "%05d")]);
      imagesc(permute(Recon_vals,[2,1,3])); 
      if num_Recon_colors == 1
	colormap(gray); 
      endif
      box off; axis off; axis image;
      saveas(Recon_fig(i_Recon), [Recon_dir, filesep, Recon_list{i_Recon}, "_", num2str(i_frame, "%05d")], "png");
      if plot_DoG_kernel && unwhiten_list(i_Recon)
	unwhitened_Recon_DoG = zeros(size(permute(Recon_vals,[2,1,3])));
	for i_color = 1 : num_Recon_colors
     cimage = squeeze(Recon_vals(:,:,i_color))';
	  tmp_Recon = ...
	      deconvolvemirrorbc(cimage, DoG_weights);
	  mean_unwhitened_Recon{i_Recon}(i_color, i_frame) = mean(tmp_Recon(:));
 	  std_unwhitened_Recon{i_Recon}(i_color, i_frame) = std(tmp_Recon(:));
	  j_frame = ceil(i_frame * tot_Recon_frames(normalize_list(i_Recon)) / tot_Recon_frames(i_Recon));
	  tmp_Recon = ...
	      (tmp_Recon - mean_Recon_tmp) * ...
	      (std_unwhitened_Recon{normalize_list(i_Recon)}(i_color, j_frame) / std_Recon_tmp) + ...
	      mean_unwhitened_Recon{normalize_list(i_Recon)}(i_color, j_frame); 
	  max_unwhitened_Recon{i_Recon}(i_color, i_frame) = max(tmp_Recon(:));
	  min_unwhitened_Recon{i_Recon}(i_color, i_frame) = min(tmp_Recon(:));
	  [unwhitened_Recon_DoG(:,:,i_color)] = tmp_Recon;
	endfor
	figure(unwhitened_Recon_fig(i_Recon));
	set(unwhitened_Recon_fig(i_Recon), "name", ["unwhitened ", Recon_list{i_Recon}, "_", num2str(i_frame, "%05d")]);
	imagesc(squeeze(unwhitened_Recon_DoG)); 
	if num_Recon_colors == 1
	  colormap(gray); 
	endif
	box off; axis off; axis image;
	saveas(unwhitened_Recon_fig(i_Recon), ...
	       [Recon_dir, filesep, "unwhitened_", Recon_list{i_Recon}, "_", num2str(i_frame, "%05d")], "png");
	drawnow
      endif %% plot_DoG_kernel
    %endfor   %% i_frame
    Recon_mean(i_Recon) = Recon_mean(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    Recon_std(i_Recon) = Recon_std(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    disp(["Recon_mean = ", num2str(Recon_mean(i_Recon)), " +/- ", num2str(Recon_std(i_Recon))]);
    
  endfor %% i_Recon
endif %% plot_Recon




  

plot_nonSparse = 0;
if plot_nonSparse
%%  nonSparse_list = ...
%%      {["a5_"], ["Error"]};%% 
  nonSparse_list = ...
      {["a4_"], ["LeftError"]; ...
       ["a8_"], ["LeftDepthError"]; ...
       ["a14_"], ["RightError"]; ...
       ["a18_"], ["RightDepthError"]};
  num_nonSparse_list = size(nonSparse_list,1);
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
	readpvpfile(nonSparse_file, progress_step, tot_nonSparse_frames, tot_nonSparse_frames-num_nonSparse+1);
    nx_nonSparse = nonSparse_hdr{i_nonSparse}.nx;
    ny_nonSparse = nonSparse_hdr{i_nonSparse}.ny;
    nf_nonSparse = nonSparse_hdr{i_nonSparse}.nf;
    n_nonSparse = nx_nonSparse * ny_nonSparse * nf_nonSparse;
    num_frames = size(nonSparse_struct,1);
    nonSparse_times = zeros(num_frames,1);
    nonSparse_RMS = zeros(num_frames,1);
    for i_frame = 1 : 1 : num_frames
      nonSparse_times(i_frame) = squeeze(nonSparse_struct{i_frame}.time);
      nonSparse_vals = squeeze(nonSparse_struct{i_frame}.values);
      nonSparse_RMS(i_frame) = std(nonSparse_vals(:));
    endfor %% i_frame
    
    nonSparse_RMS_fig = figure;
    nonSparse_RMS_hndl = plot(nonSparse_times, nonSparse_RMS); axis tight;
    set(nonSparse_RMS_fig, "name", ["RMS_", nonSparse_list{i_nonSparse,2}, "_", num2str(nonSparse_times(num_frames), "%i")]);
    saveas(nonSparse_RMS_fig, ...
	   [nonSparse_dir, filesep, "RMS_", nonSparse_list{i_nonSparse,2}, "_", num2str(nonSparse_times(num_frames), "%i")], "png");
    
    nonSparse_mean_active = median(nonSparse_RMS(:));
    disp([nonSparse_list{i_nonSparse,2}, "_", num2str(nonSparse_times(num_frames), "%i"), " median RMS = ", num2str(nonSparse_mean_active)]);
  endfor  %% i_nonSparse
endif %% plot_nonSparse
