clear all;
close all;
more off
global plot_flag %% if true, plot graphical output to screen, else do not generate graphical output
plot_flag = true;
global load_Sparse_flag %% if true, then load "saved" data structures rather than computing them 
load_Sparse_flag = false;
%if plot_flag
%   setenv("GNUTERM","X11")
%end
no_clobber = false;

%% machine/run_type environment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ismac
elseif isunix
   %%workspace_path = "/home/wshainin/workspace";
   workspace_path = "/home/ec2-user/workspace";

   %%run_type = "Stack";
   %%run_type = "Shuffle";
   %%run_type = "Shuffle_DS";
   %%run_type = "Stack_16";
   run_type = "Dylan";

   if strcmp(run_type, "Stack")
      output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_PASCAL";
      %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_vine_const_DW";
      %%output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_animal_init";
      checkpoint_dir = output_dir;
      checkpoint_parent = "/nh/compneuro/Data/vine/LCA";
      checkpoint_children = {"2013_01_24_2013_02_01/output_stack_PASCAL"};
      %%checkpoint_children = {"2013_01_24_2013_02_01/output_stack_vine_const_DW"};
      %%checkpoint_children = {"2013_01_24_2013_02_01/output_stack_animal_init"};

  elseif strcmp(run_type, "Stack_16")
     output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_PASCAL";
     checkpoint_dir = output_dir;
     checkpoint_parent = "/nh/compneuro/Data/vine/LCA";
     checkpoint_children = {"2013_01_24_2013_02_01/output_stack_PASCAL"};

  elseif strcmp(run_type, "Shuffle")
     output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_PASCAL_shuffle_1_2";
     checkpoint_dir = output_dir;
     checkpoint_parent = "/nh/compneuro/Data/vine/LCA";
     checkpoint_children = {"2013_01_24_2013_02_01/output_stack_PASCAL_shuffle_1_2"};

  elseif strcmp(run_type, "Shuffle_DS")
     output_dir = "/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_PASCAL_shuffle_2_2";
     checkpoint_dir = output_dir;
     checkpoint_parent = "/nh/compneuro/Data/vine/LCA";
     checkpoint_children = {"2013_01_24_2013_02_01/output_stack_PASCAL_shuffle_2_2"};
     
      %%%%
%%%%%%%%%%%
%%%%%%%%%%%%
  elseif strcmp(run_type, "Dylan")
     output_dir = "/home/ec2-user/mountData/AlexNetTest/output/fixed/longerFixedTest";
     checkpoint_dir = output_dir;
     checkpoint_parent = "/home/ec2-user/mountData/AlexNetTest/output/fixed/longerFixedTest";
     checkpoint_children = {""};
 end %% run_type
end %% isunix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([workspace_path, filesep, "/PetaVision/mlab/util"]);
%% default paths
if ~exist("output_dir") || isempty(output_dir)
   warning("using default output dir");
   output_dir = pwd
end
DoG_path = [];
unwhiten_flag = false;  %% set to true if DoG filtering used and dewhitening of reconstructions is desired
if unwhiten_flag && (~exist("DoG_path") || isempty(DoG_path))
   DoG_path = output_dir;
end

max_patches = 600;  %% maximum number of weight patches to plot, typically ordered from most to least active if Sparse_flag == true
checkpoint_weights_movie = true; %% make movie of weights over time using list of checkpoint folders getCheckpointList(checkpoint_parent, checkpoint_children)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot Reconstructions                   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
analyze_Recon = false;
if analyze_Recon
   if strcmp(run_type, "Stack_16") 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% Stack_16 list
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      Recon_list = ...
      {["a0_"], ["Image"];
      ["a6_"], ["Recon_S2"];
      ["a7_"], ["Recon_S4"];
      ["a8_"], ["Recon_S8"];
      ["a9_"], ["Recon_S16"];
      ["a10_"], ["Recon_SA"]};
      %% list of layers to unwhiten
      num_Recon_list = size(Recon_list,1);
      Recon_unwhiten_list = zeros(num_Recon_list,1);
      %%Recon_unwhiten_list([2,3,5,6]) = 1;
      %% list of layers to use as a normalization reference for unwhitening
      Recon_normalize_list = 1:num_Recon_list;
      %% list of (previous) layers to sum with current layer
      Recon_sum_list = cell(num_Recon_list,1);

  elseif strcmp(run_type, "Stack") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Stack list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Recon_list = ...
     {["a0_"], ["Image"];
     ["a5_"], ["Recon_S2"];
     ["a6_"], ["Recon_S4"];
     ["a7_"], ["Recon_S8"];
     ["a8_"], ["Recon_SA"]};
     %% list of layers to unwhiten
     num_Recon_list = size(Recon_list,1);
     Recon_unwhiten_list = zeros(num_Recon_list,1);
     %%Recon_unwhiten_list([2,3,5,6]) = 1;
     %% list of layers to use as a normalization reference for unwhitening
     Recon_normalize_list = 1:num_Recon_list;
     %% list of (previous) layers to sum with current layer
     Recon_sum_list = cell(num_Recon_list,1);

  elseif strcmp(run_type, "Shuffle") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Shuffle list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Recon_list = ...
     {["a0_"], ["Image"];
     ["a5_"], ["Recon_S2"];
     ["a6_"], ["Recon_S4"];
     ["a7_"], ["Recon_S8"];
     ["a8_"], ["Recon_SA"];
     ["a12_"], ["Shuffle_Recon_S2"];
     ["a13_"], ["Shuffle_Recon_S4"];
     ["a14_"], ["Shuffle_Recon_S8"];
     ["a15_"], ["Shuffle_Recon_SA"]};
     %% list of layers to unwhiten
     num_Recon_list = size(Recon_list,1);
     Recon_unwhiten_list = zeros(num_Recon_list,1);
     %%Recon_unwhiten_list([2,3,5,6]) = 1;
     %% list of layers to use as a normalization reference for unwhitening
     Recon_normalize_list = 1:num_Recon_list;
     %% list of (previous) layers to sum with current layer
     Recon_sum_list = cell(num_Recon_list,1);

  elseif strcmp(run_type, "Shuffle_DS") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Shuffle dataset list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Recon_list = ...
     {["a0_"], ["Image"];
     ["a15_"], ["Shuffle_Recon_SA"]};
     %% list of layers to unwhiten
     num_Recon_list = size(Recon_list,1);
     Recon_unwhiten_list = zeros(num_Recon_list,1);
     %%Recon_unwhiten_list([2,3,5,6]) = 1;
     %% list of layers to use as a normalization reference for unwhitening
     Recon_normalize_list = 1:num_Recon_list;
     %% list of (previous) layers to sum with current layer
     Recon_sum_list = cell(num_Recon_list,1);

  elseif strcmp(run_type, "Dylan") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Brian's MLP 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Recon_list = ...
     {["a3_"], ["HiddenLayer1"];
      ["a7_"], ["Error1"]};
     %% list of layers to unwhiten
     num_Recon_list = size(Recon_list,1);
     Recon_unwhiten_list = zeros(num_Recon_list,1);
     %%Recon_unwhiten_list([2,3,5,6]) = 1;
     %% list of layers to use as a normalization reference for unwhitening
     Recon_normalize_list = 1:num_Recon_list;
     %% list of (previous) layers to sum with current layer
     Recon_sum_list = cell(num_Recon_list,1);

  end %% run_type


  %% parse center/surround pre-processing filters
  DoG_weights = [];
  if unwhiten_flag
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% deep/lateral/noPulvinar list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     blur_center_path = [DoG_path, filesep, "ImageToBipolarCenter_W.pvp"];
     DoG_center_path = [DoG_path, filesep, "BipolarToGanglionCenter_W.pvp"];
     DoG_surround_path = [DoG_path, filesep, "BipolarToGanglionSurround_W.pvp"];
     [blur_weights] = get_Blur_weights(blur_center_path);
     [DoG_weights] = get_DoG_weights(DoG_center_path, DoG_surround_path);
  end  %% unwhiten_flag

  if strcmp(run_type, "Shuffle") || strcmp(run_type, "Shuffle_DS")
     num_Recon_frames_per_layer = 8;
     Recon_LIFO_flag = true;
  else 
     num_Recon_frames_per_layer = 2;
     Recon_LIFO_flag = true;
  end

  [Recon_hdr, ...
  Recon_fig, ...
  Recon_fig_name, ...
  Recon_vals, ...
  Recon_time, ...
  Recon_mean, ...
  Recon_std, ...
  unwhitened_Recon_fig, ...
  unwhitened_Recon_vals] = ...
  analyzeReconPVP(Recon_list, ...
  num_Recon_frames_per_layer, ...
  output_dir, ...
  plot_flag, ...
  Recon_sum_list, ...
  DoG_weights, ...
  Recon_unwhiten_list, ...
  Recon_normalize_list, ...
  Recon_LIFO_flag);
  drawnow;

end %% analyze_Recon





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot Stats Probe vs Time               %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_StatsProbe_vs_time = false;
if plot_StatsProbe_vs_time && plot_flag
   StatsProbe_plot_lines = 20000;
   max_StatsProbe_line   = Inf;
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
  elseif strcmp(run_type, "noPulvinar") || strcmp(run_type, "TopDown")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% noPulvinar list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     StatsProbe_list = ...
     {["Error"],["_Stats.txt"]; ...
     ["V1"],["_Stats.txt"];
     ["V2"],["_Stats.txt"];
     ["Error1_2"],["_Stats.txt"]; ...
     ["V1Infra"],["_Stats.txt"]};
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
     ["V2"],["_Stats.txt"]; ...
     ["V2"],["_Stats.txt"]};
  elseif strcmp(run_type, "Stack")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Stack List
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     StatsProbe_list = ...
     {["Error"],["_Stats.txt"]; ...
     ["V1_S2"],["_Stats.txt"];
     ["V1_S4"],["_Stats.txt"];
     ["V1_S8"],["_Stats.txt"]};
  elseif strcmp(run_type, "CIFAR_deep")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% MNIST/CIFAR list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     StatsProbe_list = ...
     {["LabelError"],["_Stats.txt"]; ...
     ["Error"],["_Stats.txt"]; ...
     ["Error1_2"],["_Stats.txt"]; ...
     ["V1"],["_Stats.txt"];...
     ["V2"],["_Stats.txt"]};
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  end %% run_type
  StatsProbe_vs_time_dir = [output_dir, filesep, "StatsProbe_vs_time"]
  [status, msg, msgid] = mkdir(StatsProbe_vs_time_dir);
  if status ~= 1
     warning(["mkdir(", StatsProbe_vs_time_dir, ")", " msg = ", msg]);
  end 
  num_StatsProbe_list = size(StatsProbe_list,1);

  StatsProbe_sigma_flag = ones(1,num_StatsProbe_list);
  if strcmp(run_type, "color_deep")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% deep list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     StatsProbe_sigma_flag([2,4,6]) = 0;
  elseif strcmp(run_type, "noPulvinar") || strcmp(run_type, "TopDown")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% noPulvinar list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     StatsProbe_sigma_flag([2,3,5]) = 0;
  elseif strcmp(run_type, "lateral")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% lateral list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     StatsProbe_sigma_flag([2,4,6,7]) = 0;
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "Stack")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Stack list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     StatsProbe_sigma_flag([2,3,4]) = 0;
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif  strcmp(run_type, "CIFAR_deep") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% MNIST/CIFAR list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     StatsProbe_sigma_flag([4]) = 0;
     StatsProbe_sigma_flag([5]) = 0;
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  end %% run_type
  StatsProbe_nnz_flag = ~StatsProbe_sigma_flag;
  for i_StatsProbe = 1 : num_StatsProbe_list
     StatsProbe_file = [output_dir, filesep, StatsProbe_list{i_StatsProbe,1}, StatsProbe_list{i_StatsProbe,2}]
     if ~exist(StatsProbe_file,"file")
        warning(["StatsProbe_file does not exist: ", StatsProbe_file]);
        continue;
    end
    [status, wc_output] = system(["cat ",StatsProbe_file," | wc"], true, "sync");
    if status ~= 0
       error(["system call to compute num lines failed in file: ", StatsProbe_file, " with status: ", num2str(status)]);
    end
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
    end
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
    end %%i_line
    fclose(StatsProbe_fid);
    if plot_flag
       StatsProbe_vs_time_fig(i_StatsProbe) = figure;
    end
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
      end
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
      end
      save("-mat", ...
      [StatsProbe_vs_time_dir, filesep, StatsProbe_list{i_StatsProbe,1}, ...
      "_sigma_vs_time_", num2str(StatsProbe_time_vals(end), "%08d"), ".mat"], ...
      "StatsProbe_time_vals", "StatsProbe_sigma_vals");
   end %% 
   drawnow;
end %% i_StatsProbe
end  %% plot_StatsProbe_vs_time







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot Sparse Layers                     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
analyze_Sparse_flag = false;
if analyze_Sparse_flag
   Sparse_frames_list = [];
   
   if strcmp(run_type, "Stack_16") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Stack_16 list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Sparse_list = ...
     {["a2_"], ["V1_S2"]; ...
     ["a3_"], ["V1_S4"]; ...
     ["a4_"], ["V1_S8"]; ...
     ["a5_"], ["V1_S16"]};
  
   elseif strcmp(run_type, "Stack") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Stack list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Sparse_list = ...
     {["a2_"], ["V1_S2"]; ...
     ["a3_"], ["V1_S4"]; ...
     ["a4_"], ["V1_S8"]};
   
   elseif strcmp(run_type, "Shuffle") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Shuffle list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Sparse_list = ...
     {["a2_"], ["V1_S2"]; ...
     ["a3_"], ["V1_S4"]; ...
     ["a4_"], ["V1_S8"]};
   
   elseif strcmp(run_type, "Dylan") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Shuffle list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Sparse_list = ...
     {["a2_"], ["L1"]};
   
   end %% run_type

  fraction_Sparse_frames_read = 1;
  min_Sparse_skip = 1;
  fraction_Sparse_progress = 10;
  num_epochs = 4;
  if load_Sparse_flag
     num_procs = 1;
  else
     num_procs = 8;
  end
  [Sparse_hdr, ...
  Sparse_hist_rank_array, ...
  Sparse_times_array, ...
  Sparse_percent_active_array, ...
  Sparse_percent_change_array, ...
  Sparse_std_array, ...
  Sparse_struct_array] = ...
  analyzeSparseEpochsPVP2(Sparse_list, ...
  output_dir, ...
  load_Sparse_flag, ...
  plot_flag, ...
  fraction_Sparse_frames_read, ...
  min_Sparse_skip, ...
  fraction_Sparse_progress, ...
  Sparse_frames_list, ...
  num_procs, ...
  num_epochs);
  drawnow;

end %% plot_Sparse_flag





analyze_nonSparse_flag = false;
if analyze_nonSparse_flag
   if strcmp(run_type, "Stack") || strcmp(run_type, "Stack_16") || strcmp(run_type, "Shuffle")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Stack/Shuffle
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     nonSparse_list = ...
     {["a1_"], ["Error"]};
     num_nonSparse_list = size(nonSparse_list,1);
     nonSparse_skip = repmat(1, num_nonSparse_list, 1);
     nonSparse_skip(1) = 10;
     nonSparse_norm_list = ...
     {["a0_"], ["Image"]};
     nonSparse_norm_strength = ones(num_nonSparse_list,1);
     nonSparse_norm_strength(1) = ...
     1/sqrt(18*18);
     Sparse_std_ndx = 0;

  elseif strcmp(run_type, "Dylan")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Dylan
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     nonSparse_list = ...
     {["a1_"], ["Residual"]};
     num_nonSparse_list = size(nonSparse_list,1);
     nonSparse_skip = repmat(1, num_nonSparse_list, 1);
     nonSparse_skip(1) = 10;
     nonSparse_norm_list = ...
     {["a0_"], ["Input"]};
     nonSparse_norm_strength = ones(num_nonSparse_list,1);
     nonSparse_norm_strength(1) = 1/sqrt(8*8);
     Sparse_std_ndx = 0;

  end %% run_type
  
  if ~exist("Sparse_std_ndx")
     Sparse_std_ndx = zeros(num_nonSparse_list,1);
  end
  if ~exist("nonSparse_norm_strength")
     nonSparse_norm_strength = ones(num_nonSparse_list,1);
  end

  fraction_nonSparse_frames_read = 1;
  min_nonSparse_skip = 1;
  fraction_nonSparse_progress = 10;
  [nonSparse_times_array, ...
  nonSparse_RMS_array, ...
  nonSparse_norm_RMS_array, ...
  nonSparse_RMS_fig] = ...
  analyzeNonSparsePVP(nonSparse_list, ...
  nonSparse_skip, ...
  nonSparse_norm_list, ...
  nonSparse_norm_strength, ...
  Sparse_times_array, ...
  Sparse_std_array, ...
  Sparse_std_ndx, ...
  output_dir, ...
  plot_flag, ...
  fraction_nonSparse_frames_read, ...
  min_nonSparse_skip, ...
  fraction_nonSparse_progress);

end %% analyze_nonSparse_flag





plot_ReconError = false && analyze_nonSparse_flag;
ReconError_RMS_fig_ndx = [];
if plot_ReconError
   if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown") 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% deep/noTopDown list
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ReconError_list = ...
      {["a5_"], ["Recon"]; ...
      ["a11_"], ["ReconInfra"]};
      num_ReconError_list = size(ReconError_list,1);
      ReconError_skip = repmat(1, num_ReconError_list, 1);
      ReconError_skip(1) = 1;
      ReconError_skip(2) = 1;
      ReconError_norm_list = ...
      {["a2_"], ["Ganglion"]; ...
      ["a2_"], ["Ganglion"]};
      ReconError_norm_strength = ones(num_ReconError_list,1);
      ReconError_RMS_fig_ndx = [1 1];  %% causes recon error to be overlaid on specified  nonSparse (Error) figure
  elseif strcmp(run_type, "noPulvinar")  || strcmp(run_type, "TopDown")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% noPulvinar/TopDown list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     ReconError_list = ...
     {["a5_"], ["Recon"]; ...
     ["a9_"], ["ReconInfra"]};
     num_ReconError_list = size(ReconError_list,1);
     ReconError_skip = repmat(1, num_ReconError_list, 1);
     ReconError_skip(1) = 1;
     ReconError_skip(2) = 1;
     ReconError_norm_list = ...
     {["a2_"], ["Ganglion"]; ...
     ["a2_"], ["Ganglion"]};
     ReconError_norm_strength = ones(num_ReconError_list,1);
     ReconError_RMS_fig_ndx = [1 1];  %% causes recon error to be overlaid on specified  nonSparse (Error) figure
  elseif strcmp(run_type, "V1")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% V1 list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     ReconError_list = ...
     {["a5_"], ["Recon"]};
     num_ReconError_list = size(ReconError_list,1);
     ReconError_skip = repmat(1, num_ReconError_list, 1);
     ReconError_skip(1) = 1;
     ReconError_norm_list = ...
     {["a2_"], ["Ganglion"]};
     ReconError_norm_strength = ones(num_ReconError_list,1);
     ReconError_RMS_fig_ndx = [1];  %% causes recon error to be overlaid on specified  nonSparse (Error) figure
  elseif strcmp(run_type, "Heli_DPT") || strcmp(run_type, "Heli_C1") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Heli_DPT/Heli_C1 list
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
     ReconError_norm_strength = ...
     [1/sqrt(18*18); 1/sqrt(18*18)];
     ReconError_RMS_fig_ndx = [1 1];  %% causes recon error to be overlaid on specified  nonSparse (Error) figure
  elseif strcmp(run_type, "Heli_D") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Heli_D list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     ReconError_list = ...
     {["a3_"], ["Recon"]; ...
     ["a7_"], ["ReconInfra"]};
     num_ReconError_list = size(ReconError_list,1);
     ReconError_skip = repmat(1, num_ReconError_list, 1);
     ReconError_skip(1) = 1;
     ReconError_skip(2) = 1;
     ReconError_norm_list = ...
     {["a0_"], ["Image"]; ...
     ["a0_"], ["Image"]};
     ReconError_norm_strength = ones(num_ReconError_list,1);
     ReconError_norm_strength = ...
     [1/sqrt(18*18); 1/sqrt(18*18)];
     ReconError_RMS_fig_ndx = [1 1];  %% causes recon error to be overlaid on specified  nonSparse (Error) figure
  elseif strcmp(run_type, "Stack_16") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Stack_16 list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     ReconError_list = ...
     {["a6_"], ["Recon_S2"]; ...
     ["a7_"], ["Recon_S4"]; ...
     ["a8_"], ["Recon_S8"]; ...
     ["a9_"], ["Recon_S16"]};
     num_ReconError_list = size(ReconError_list,1);
     ReconError_skip = repmat(1, num_ReconError_list, 1);
     ReconError_skip(1) = 1;
     ReconError_skip(2) = 1;
     ReconError_norm_list = ...
     {["a0_"], ["Image"]; ...
     ["a0_"], ["Image"]; ...
     ["a0_"], ["Image"]; ...
     ["a0_"], ["Image"]};
     ReconError_norm_strength = ones(num_ReconError_list,1);
     ReconError_norm_strength = ...
     [1/sqrt(18*18); 1/sqrt(18*18); 1/sqrt(18*18); 1/sqrt(18*18)];
     ReconError_RMS_fig_ndx = [1 1 1 1];  %% causes recon error to be overlaid on specified  nonSparse (Error) figure %%%%Added value. If broken, see Heli_D%%%%
  elseif strcmp(run_type, "Stack") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% Stack list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     ReconError_list = ...
     {["a5_"], ["Recon_S2"]; ...
     ["a6_"], ["Recon_S4"]; ...
     ["a7_"], ["Recon_S8"]};
     num_ReconError_list = size(ReconError_list,1);
     ReconError_skip = repmat(1, num_ReconError_list, 1);
     ReconError_skip(1) = 1;
     ReconError_skip(2) = 1;
     ReconError_norm_list = ...
     {["a0_"], ["Image"]; ...
     ["a0_"], ["Image"]; ...
     ["a0_"], ["Image"]};
     ReconError_norm_strength = ones(num_ReconError_list,1);
     ReconError_norm_strength = ...
     [1/sqrt(18*18); 1/sqrt(18*18); 1/sqrt(18*18)];
     ReconError_RMS_fig_ndx = [1 1 1];  %% causes recon error to be overlaid on specified  nonSparse (Error) figure %%%%Added value. If broken, see Heli_D%%%%
  elseif strcmp(run_type, "CIFAR_deep") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% CIFAR_deep list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     ReconError_list = ...
     {["a3_"],  ["Recon"]; ...
     ["a7_"],  ["ReconInfra"]};
     num_ReconError_list = size(ReconError_list,1);
     ReconError_skip = repmat(1, num_ReconError_list, 1);
     ReconError_skip(1) = 1;
     ReconError_skip(2) = 1;
     ReconError_norm_list = ...
     {["a0_"], ["Image"]; ...
     ["a0_"], ["Image"]};
     ReconError_norm_strength = ...
     [1/sqrt(32*32); 1/sqrt(32*32)];
     ReconError_RMS_fig_ndx = [1 1];
  elseif strcmp(run_type, "CIFAR_C1") 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% CIFAR_C1 list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     ReconError_list = ...
     {["a3_"],  ["Recon"]; ...
     ["a8_"],  ["ReconC1"]};
     num_ReconError_list = size(ReconError_list,1);
     ReconError_skip = repmat(1, num_ReconError_list, 1);
     ReconError_skip(1) = 1;
     ReconError_skip(2) = 1;
     ReconError_norm_list = ...
     {["a0_"], ["Image"]; ...
     ["a0_"], ["Image"]};
     ReconError_norm_strength = ...
     [1/sqrt(32*32); 1/sqrt(32*32)];
     ReconError_RMS_fig_ndx = [1 1];
  elseif strcmp(run_type, "lateral")
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% lateral list
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     ReconError_list = ...
     {["a5_"], ["Recon"]; ...
     ["a8_"], ["Recon2"]; ...
     ["a11_"], ["ReconInfra"]; ...
     ["a13_"], ["Recon2PlusReconInfra"]; ...
     ["a16_"], ["ReconMaxPoolingV2X05"]};
     num_ReconError_list = size(ReconError_list,1);
     ReconError_skip = repmat(1, num_ReconError_list, 1);
     ReconError_skip(1) = 1;
     ReconError_skip(2) = 1;
     ReconError_skip(3) = 1;
     ReconError_norm_list = ...
     {["a2_"], ["Ganglion"]; ...
     ["a2_"], ["Ganglion"]; ...
     ["a2_"], ["Ganglion"]; ...
     ["a2_"], ["Ganglion"]; ...
     ["a2_"], ["Ganglion"]};
     ReconError_norm_strength = ones(num_ReconError_list,1);
     ReconError_norm_strength(4) = 2;
     ReconError_RMS_fig_ndx = [1 1 1 1 1];
  end %% run_type


  [nonSparse_times_array, ...
  nonSparse_RMS_array, ...
  nonSparse_norm_RMS_array, ...
  ReconError_RMS_fig] = ...
  analyzeReconErrorPVP(ReconError_list, ...
  ReconError_skip, ...
  ReconError_norm_list, ...
  ReconError_norm_strength, ...
  nonSparse_times_array, ...
  nonSparse_RMS_array, ...
  nonSparse_norm_RMS_array, ...
  nonSparse_RMS_fig, ...
  ReconError_RMS_fig_ndx, ...
  output_dir, ...
  plot_flag, ...
  fraction_nonSparse_frames_read, ...
  min_nonSparse_skip, ...
  fraction_nonSparse_progress);
  drawnow;
end %% plot_ReconError





plot_ErrorVsSparse = false && analyze_Sparse_flag && analyze_nonSparse_flag;
if plot_ErrorVsSparse
   ErrorVsSparse_list = [nonSparse_list; ReconError_list];
   num_ErrorVsSparse_list = size(ErrorVsSparse_list,1);
   Sparse_axis_index = ones(num_ErrorVsSparse_list,1);
   if strcmp(run_type, "color_deep")
      Sparse_axis_index(1) = [1,2];  %% combine sparsity of V1 + V2
      Sparse_axis_index(2) = 2;
      Sparse_axis_index(3) = 2;
      Sparse_axis_index(4) = 1;
      Sparse_axis_index(5) = 2;
      Sparse_axis_index(6) = 2;
  elseif strcmp(run_type, "noTopDown") 
     Sparse_axis_index(2) = 2;
     Sparse_axis_index(3) = 2;
     Sparse_axis_index(4) = 1;
     Sparse_axis_index(5) = 2;
     Sparse_axis_index(6) = 2;
  elseif strcmp(run_type, "Heli_DPT") || strcmp(run_type, "Heli_C1") 
     Sparse_axis_index(2) = 2;
     Sparse_axis_index(3) = 2;
     Sparse_axis_index(4) = 1;
     Sparse_axis_index(5) = 2;
     Sparse_axis_index(6) = 2;
  elseif strcmp(run_type, "noPulvinar") || strcmp(run_type, "TopDown")
     Sparse_axis_index(2) = 2;
     Sparse_axis_index(3) = 1;
     Sparse_axis_index(4) = 2;
     Sparse_axis_index(5) = 2;
     Sparse_axis_index(6) = 2;
  elseif strcmp(run_type, "V1")
     Sparse_axis_index(2) = 1;
  elseif strcmp(run_type, "CIFAR_deep")
     Sparse_axis_index(2) = 2;
     Sparse_axis_index(3) = 2;
     Sparse_axis_index(4) = 1;
     Sparse_axis_index(5) = 2;
  elseif strcmp(run_type, "CIFAR_C1")
     Sparse_axis_index(2) = 2;
     Sparse_axis_index(3) = 2;
     Sparse_axis_index(4) = 1;
     Sparse_axis_index(5) = 2;
  elseif strcmp(run_type, "lateral")
     Sparse_axis_index(2) = 2;
     Sparse_axis_index(3) = 2;
     Sparse_axis_index(4) = 1;
     Sparse_axis_index(5) = 2;
     Sparse_axis_index(6) = 2;
     Sparse_axis_index(7) = 2;
  end



  ErrorVsSparse_dir = [output_dir, filesep, "ErrorVsSparse"]
  [status, msg, msgid] = mkdir(ErrorVsSparse_dir);
  if status ~= 1
     warning(["mkdir(", ErrorVsSparse_dir, ")", " msg = ", msg]);
  end 
  for i_ErrorVsSparse = 1 : num_ErrorVsSparse_list
     nonSparse_times = nonSparse_times_array{i_ErrorVsSparse};
     nonSparse_RMS = nonSparse_RMS_array{i_ErrorVsSparse};
     nonSparse_norm_RMS = nonSparse_norm_RMS_array{i_ErrorVsSparse};
     num_nonSparse_frames = length(nonSparse_times);
     if num_nonSparse_frames < 2
        continue;
    end



    %% get percent active bins
    i_Sparse = Sparse_axis_index(i_ErrorVsSparse,1);
    Sparse_times = Sparse_times_array{i_Sparse};
    Sparse_percent_active = Sparse_percent_active_array{i_Sparse};
    num_Sparse_neurons = ...
    Sparse_hdr{i_Sparse}.nxGlobal * Sparse_hdr{i_Sparse}.nyGlobal * Sparse_hdr{i_Sparse}.nf;
    Sparse_sum_norm = 1;
    for i_Sparse_sum = 2 : size(Sparse_axis_index,2)
       i_Sparse = Sparse_axis_index(i_ErrorVsSparse,i_Sparse_sum);
       Sparse_times_sum = Sparse_times_array{i_Sparse};
       if any(Sparse_times_sum ~= Sparse_times)
          continue;
      end
      num_Sparse_neurons_sum = ...
      Sparse_hdr{i_Sparse}.nxGlobal * Sparse_hdr{i_Sparse}.nyGlobal * Sparse_hdr{i_Sparse}.nf;
      Sparse_percent_active = Sparse_percent_active + (num_Sparse_neurons_sum / num_Sparse_neurons) * Sparse_percent_active_array{i_Sparse};
      Sparse_sum_norm = Sparse_sum_norm + (num_Sparse_neurons_sum / num_Sparse_neurons);
   end
   Sparse_percent_active = Sparse_percent_active / Sparse_sum_norm;
   first_nonSparse_time = nonSparse_times(2);
   second_nonSparse_time = nonSparse_times(3);
   last_nonSparse_time = nonSparse_times(end);    
   [first_Sparse_ndx1, ~, first_Sparse_diff1] = ...
   find((Sparse_times - first_nonSparse_time) >= 0, 1, "first");
   [first_Sparse_ndx2, ~, first_Sparse_diff2] = ...
   find((Sparse_times - first_nonSparse_time) <= 0, 1, "last");
   if abs(first_Sparse_diff1) < abs(first_Sparse_diff2)
      first_Sparse_ndx = first_Sparse_ndx1;
      first_Sparse_diff = first_Sparse_diff1;
   else
      first_Sparse_ndx = first_Sparse_ndx2;
      first_Sparse_diff = first_Sparse_diff2;
   end      
   [second_Sparse_ndx1, ~, second_Sparse_diff1] = ...
   find(Sparse_times - second_nonSparse_time >= 0, 1, "first");
   [second_Sparse_ndx2, ~, second_Sparse_diff2] = ...
   find(Sparse_times - second_nonSparse_time <= 0, 1, "last");
   if abs(second_Sparse_diff1) < abs(second_Sparse_diff2)
      second_Sparse_ndx = second_Sparse_ndx1;
      second_Sparse_diff = second_Sparse_diff1;
   else
      second_Sparse_ndx = second_Sparse_ndx2;
      second_Sparse_diff = second_Sparse_diff2;
   end      
   if max(Sparse_times(:)) >= last_nonSparse_time
      [last_Sparse_ndx, ~, last_Sparse_diff] = ...
      find(Sparse_times - last_nonSparse_time < 0, 1, "last");
      %%last_nonSparse_ndx = num_nonSparse_frames;
   else
      %%[last_nonSparse_ndx, ~, last_nonSparse_diff] = find(nonSparse_times - Sparse_times(end) < 0, 1, "last");
      last_Sparse_ndx = length(Sparse_times);
   end
   skip_Sparse_ndx = max(second_Sparse_ndx - first_Sparse_ndx, 1);
   Sparse_vals = 1-Sparse_percent_active(first_Sparse_ndx:skip_Sparse_ndx:last_Sparse_ndx);
   num_Sparse_vals = length(Sparse_vals);
   if num_Sparse_vals < 1
      continue;
   end
   num_Sparse_bins = 5; %%10;
   min_Sparse_val = min(Sparse_vals(:));
   max_Sparse_val = max(Sparse_vals(:));
   skip_Sparse_val = (max_Sparse_val - min_Sparse_val) / num_Sparse_bins;
   if skip_Sparse_val == 0
      skip_Sparse_val = 1;
   end
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
      end
   end %% i_Sparse_bin
   last_nonSparse_ndx = length(Sparse_vals);
   ErrorVsSparse_name = ...
   ["ErrorVsSparse_", ErrorVsSparse_list{i_ErrorVsSparse,1}, ErrorVsSparse_list{i_ErrorVsSparse,2}, "_", ...
   num2str(nonSparse_times(num_nonSparse_frames), "%08d")];
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
      end
      hold on
      eh = errorbar(Sparse_bins+skip_Sparse_val/2, mean_nonSparse_RMS, std_nonSparse_RMS);
      set(eh, "color", [0 0 0]);
      set(eh, "linewidth", 1.5);
      set(ErrorVsSparse_fig, "name", ...
      ErrorVsSparse_name);
      saveas(ErrorVsSparse_fig, ...
      [ErrorVsSparse_dir, filesep, ...
      ErrorVsSparse_name, "png"]);
   end %% plot_flag
   save("-mat", ...
   [ErrorVsSparse_dir, filesep, ErrorVsSparse_name, ".mat"], ...
   "nonSparse_times", "Sparse_vals", "nonSparse_RMS", "nonSparse_norm_RMS", ...
   "Sparse_bins", "mean_nonSparse_RMS", "std_nonSparse_RMS");	 
end  %% i_ErrorVsSparse
drawnow;
end %% plot_ErrorVsSparse






%%keyboard;
plot_weights = true;
if plot_weights
   weights_list = {};
   labelWeights_list = {};

   if strcmp(run_type, "Stack_16")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list STACK_16
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_ndx = [1; 2; 3; 4];
    weights_list = ...
    {["V1_S2ToError"], "_W"; ...
    ["V1_S4ToError"], "_W"; ...
    ["V1_S8ToError"], "_W"; ...
    ["V1_S16ToError"], "_W"};
    checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    num_checkpoints = size(checkpoints_list,1);

 elseif strcmp(run_type, "Stack")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list STACK 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_ndx = [1; 2; 3];
    weights_list = ...
    {["V1_S2ToError"], "_W"; ...
    ["V1_S4ToError"], "_W"; ...
    ["V1_S8ToError"], "_W"};
    checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    num_checkpoints = size(checkpoints_list,1);
    
 elseif strcmp(run_type, "Shuffle")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list Shuffle 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_ndx = [1; 2; 3];
    weights_list = ...
    {["ShuffleV1_S2ToError"], "_W"; ...
    ["ShuffleV1_S4ToError"], "_W"; ...
    ["ShuffleV1_S8ToError"], "_W"};
    checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    num_checkpoints = size(checkpoints_list,1);

 elseif strcmp(run_type, "Dylan")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Dylan's weights
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_ndx = [1];
    weights_list = ...
    {["W1T"], "_W"};
    checkpoint_parent
    checkpoint_children
     checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children)
    num_checkpoints = size(checkpoints_list,1);
 end %% run_type


 num_weights_list = size(weights_list,1);
 weights_hdr = cell(num_weights_list,1);
 pre_hdr = cell(num_weights_list,1);
 if checkpoint_weights_movie
    weights_movie_dir = [output_dir, filesep, "weights_movie"]
    [status, msg, msgid] = mkdir(weights_movie_dir);
    if status ~= 1
       warning(["mkdir(", weights_movie_dir, ")", " msg = ", msg]);
    end 
 end
 weights_dir = [output_dir, filesep, "weights"]
 [status, msg, msgid] = mkdir(weights_dir);
 if status ~= 1
    warning(["mkdir(", weights_dir, ")", " msg = ", msg]);
 end 
 for i_weights = 1 : num_weights_list
    max_weight_time = 0;
    max_checkpoint = 0;
    for i_checkpoint = 1 : num_checkpoints
       checkpoint_dir = checkpoints_list{i_checkpoint,:};
       weights_file = [checkpoint_dir, filesep, weights_list{i_weights,1}, weights_list{i_weights,2}, ".pvp"];
       if ~exist(weights_file, "file")
          warning(["file does not exist: ", weights_file]);
          continue;
      end
      weights_fid = fopen(weights_file);
      weights_hdr{i_weights} = readpvpheader(weights_fid);    
      fclose(weights_fid);

      weight_time = weights_hdr{i_weights}.time;
      if weight_time > max_weight_time
         max_weight_time = weight_time;
         max_checkpoint = i_checkpoint;
      end
   end %% i_checkpoint

   for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights_file = [checkpoint_dir, filesep, weights_list{i_weights,1}, weights_list{i_weights,2}, ".pvp"];
      if ~exist(weights_file, "file")
         warning(["file does not exist: ", weights_file]);
         continue;
      end
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
      size(weight_vals)
      weight_time = squeeze(weights_struct{i_frame}.time);
      weights_name =  [weights_list{i_weights,1}, weights_list{i_weights,2}, "_", num2str(weight_time, "%08d")];
      if no_clobber && exist([weights_movie_dir, filesep, weights_name, ".png"]) && i_checkpoint ~= max_checkpoint
         continue;
      end
      tmp_ndx = sparse_ndx(i_weights);
      if analyze_Sparse_flag
         tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
         tmp_rank = [];
      end
      if analyze_Sparse_flag && ~isempty(tmp_rank)
         pre_hist_rank = tmp_rank;
      else
         pre_hist_rank = (1:weights_hdr{i_weights}.nf);
      end

      if length(labelWeights_list) >= i_weights && ...
         ~isempty(labelWeights_list{i_weights}) && ...
         plot_flag && ...
         i_checkpoint == max_checkpoint
         labelWeights_file = ...
         [checkpoint_dir, filesep, labelWeights_list{i_weights,1}, labelWeights_list{i_weights,2}, ".pvp"]
         if ~exist(labelWeights_file, "file")
            warning(["file does not exist: ", labelWeights_file]);
            continue;
   end
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
end

%% make tableau of all patches
%%keyboard;
i_patch = 1;
disp('num_weights_dims')
num_weights_dims = ndims(weight_vals)
num_patches = size(weight_vals, num_weights_dims)
num_patches = min(num_patches, max_patches)
num_patches_rows = floor(sqrt(num_patches))
num_patches_cols = ceil(num_patches / num_patches_rows)
num_weights_colors = 1;
if num_weights_dims == 4
   num_weights_colors = size(weight_vals,3)
end
if plot_flag && i_checkpoint == max_checkpoint
   weights_fig = figure;
   set(weights_fig, "name", weights_name);
end
weight_patch_array = [];
for j_patch = 1  : num_patches
   i_patch = pre_hist_rank(j_patch)
   if plot_flag && i_checkpoint == max_checkpoint
      subplot(num_patches_rows, num_patches_cols, j_patch); 
   end
   if num_weights_colors == 1
      i_patch
      size(weight_vals)
      patch_tmp = squeeze(weight_vals(:,:,i_patch));
      i_patch
   else
      patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
   end
   patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
   min_patch = min(patch_tmp2(:));
   max_patch = max(patch_tmp2(:));
   patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch - min_patch)==0));
   patch_tmp2 = uint8(permute(patch_tmp2, [2,1,3])); %% uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));
   if plot_flag && i_checkpoint == max_checkpoint
      imagesc(patch_tmp2); 
      if num_weights_colors == 1
         colormap(gray);
     end
     box off
     axis off
     axis image
     if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
        [~, max_label] = max(squeeze(labelWeights_vals(:,i_patch)));
        text(size(weight_vals,1)/2, -size(weight_vals,2)/6, num2str(max_label-1), "color", [1 0 0]);
     end %% ~empty(labelWeights_vals)
     %%drawnow;
  end %% plot_flag
  if isempty(weight_patch_array)
     weight_patch_array = ...
     zeros(num_patches_rows*size(patch_tmp2,1), num_patches_cols*size(patch_tmp2,2), size(patch_tmp2,3));
  end
  col_ndx = 1 + mod(j_patch-1, num_patches_cols);
  row_ndx = 1 + floor((j_patch-1) / num_patches_cols);
  weight_patch_array(((row_ndx-1)*size(patch_tmp2,1)+1):row_ndx*size(patch_tmp2,1), ...
  ((col_ndx-1)*size(patch_tmp2,2)+1):col_ndx*size(patch_tmp2,2),:) = ...
  patch_tmp2;
      end  %% j_patch
      if plot_flag && i_checkpoint == max_checkpoint
         if ~isempty(labelWeights_vals)
            saveas(weights_fig, [weights_dir, filesep, [weights_name,"_labeled"], ".png"], "png");
   else
      saveas(weights_fig, [weights_dir, filesep, weights_name, ".png"], "png");
   end
end
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
end

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
       ["LabelWeights_", labelWeights_list{i_weights,1}, labelWeights_list{i_weights,2}, ...
       "_", num2str(labelWeights_time, "%08d")];
       %%title(label_weights_fig, label_weights_str);
       figure(label_weights_fig, "name", label_weights_str); title(label_weights_str);
       saveas(label_weights_fig, [weights_dir, filesep, label_weights_str, ".png"] , "png");

       %% Plot the average movie weights for a label %%
       labeledWeights_str = ...
       ["labeledWeights_", ...
       weights_list{i_weights,1}, weights_list{i_weights,2}, "_", ...
       num2str(weight_time, "%08d")];
       labeledWeights_fig = figure("name", labeledWeights_str);
       title(labeledWeights_str);
       rows_labeledWeights = ceil(sqrt(size(labelWeights_vals,1)));
       cols_labeledWeights = ceil(size(labelWeights_vals,1) / rows_labeledWeights);
       for label = 0 : size(labelWeights_vals,1)-1 %% anything 0:0
          subplot(rows_labeledWeights, cols_labeledWeights, label+1);
          if num_weights_colors == 1
             imagesc(squeeze(mean(weight_vals(:,:,maxind(maxnum==(label+1))),3))')
     else
        imagesc(permute(squeeze(mean(weight_vals(:,:,:,1+mod(maxind(maxnum==(label+1))-1,size(weight_vals,4))),4)),[2,1,3]));
     end
     labeledWeights_subplot_str = ...
     [num2str(label, "%d")];
     title(labeledWeights_subplot_str);
     axis off
  end %% label
  saveas(labeledWeights_fig,  [weights_dir, filesep, labeledWeights_str, ".png"], "png");
      end  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

   end %% i_checkpoint
end %% i_weights
end  %% plot_weights




%%keyboard;
plot_labelRecon = true;
labels_list = {};
labelRecon_list = {};
if plot_labelRecon
   if strcmp(run_type, "CIFAR_deep") 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% MNIST/CIFAR list
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      labels_list = ...
      {["a6_"], ["gt"]};
      labelRecon_list = ...
      {["a5_"], ["FinalLayer"]};
  end %% run_type
end
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
      end
      label_time(i_frame) = squeeze(labels_struct{i_frame}.time);
   end

   labelRecon_file = ...
   [output_dir, filesep, labelRecon_list{i_label,1}, labelRecon_list{i_label,2}, ".pvp"]
   if ~exist(labelRecon_file, "file")
      warning(["does not exist: ", labelRecon_file]);
      break;
   end
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
      end
      labelRecon_time(i_frame) = squeeze(labelRecon_struct{i_frame}.time);
   end
   delta_frames = 1;
   [max_label_vals, max_label_ndx] = max(label_vals);
   [max_labelRecon_vals, max_labelRecon_ndx] = max(labelRecon_vals);
   for i_shift = 0:2 %% correct i_shift should be 1 but if simulation is running during analysis could be off
      accuracy = ...
      sum(max_label_ndx(1:end-i_shift)==max_labelRecon_ndx(i_shift+1:end)) / ...
      (numel(max_label_vals)-i_shift)
   end

end
end  %% plot_weightLabels


%%keyboard;
plot_weights0_2 = false; %%(true && ~strcmp(run_type, "MNIST"));
plot_weights0_2_flag = false;
plot_labelWeights_flag = false;
if plot_weights0_2
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
    end %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 elseif strcmp(run_type, "Heli_DPT") || strcmp(run_type, "Heli_D") 
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
    end %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 elseif strcmp(run_type, "Heli_C1")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% deep list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
       checkpoints_list = {output_dir};
       weights1_2_list = ...
       {["w_"], ["C1ToError2"]; ...
       ["w10_"], ["C1ToError1_2"]};
       post1_2_list = ...
       {["a2_"], ["S1"]; ...
       ["a2_"], ["S1"]};
       %% list of weights from layer1 to image
       weights0_1_list = ...
       {["w4_"], ["S1ToError"]; ...
       ["w4_"], ["S1ToError"]};
       image_list = ...
       {["a0_"], ["Image"]; ...
       ["a0_"], ["Image"]};
    else
       checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
       weights1_2_list = ...
       {["C1ToError2"], ["_W"]; ...
       ["C1ToError1_2"], ["_W"]};
       post1_2_list = ...
       {["S1"], ["_A"]; ...
       ["S1"], ["_A"]};
       %% list of weights from layer1 to image
       weights0_1_list = ...
       {["S1ToError"], ["_W"]; ...
       ["S1ToError"], ["_W"]};
       %%      image_list = ...
       %%          {["a1_"], ["Image"]};
       image_list = ...
       {["Image"], ["_A"]; ...
       ["Image"], ["_A"]};
    end %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2, 2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 elseif strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR_deep list
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
       labelWeights_list = ...
       {["V2ToLabelError"], ["_W"]};
    end %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 elseif strcmp(run_type, "CIFAR_C1") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR_C1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
       checkpoints_list = {output_dir};
       weights1_2_list = ...
       {["w9_"], ["S2ToErrorS1C1Local"]; ...
       ["w9_"], ["S2ToErrorS1C1Lateral"]};
       post1_2_list = ...
       {["a2_"], ["S1"]; ...
       ["a2_"], ["S1"]};
       %% list of weights from layer1 to image
       weights0_1_list = ...
       {["w1_"], ["S1ToError"]; ...
       ["w1_"], ["S1ToError"]};
       image_list = ...
       {["a0_"], ["Image"]; ...
       ["a0_"], ["Image"]};
    else
       checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
       weights1_2_list = ...
       {["C1ToErrorS1C1Local"], ["_W"]; ...
       ["C1ToErrorS1C1Lateral"], ["_W"]};
       post1_2_list = ...
       {["S1"], ["_A"]; ...
       ["S1"], ["_A"]};
       %% list of weights from layer1 to image
       weights0_1_list = ...
       {["S1ToError"], ["_W"]; ...
       ["S1ToError"], ["_W"]};
       %%      image_list = ...
       %%          {["a1_"], ["Image"]};
       image_list = ...
       {["Image"], ["_A"]; ...
       ["Image"], ["_A"]};
       %%      labelWeights_list = ...
       %%	  {["V2ToLabelError"], ["_W"]};
    end %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2,2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 elseif strcmp(run_type, "noPulvinar") || strcmp(run_type, "TopDown")
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
    end %% checkpoint_weights_movie
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
    end %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_ndx = [2,2];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 end %% run_type
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 num_weights1_2_list = size(weights1_2_list,1);
 if num_weights1_2_list == 0
    break;
 end

 %% get image header (to get image dimensions)
 i_image = 1;
 image_file = ...
 [output_dir, filesep, image_list{i_image,1}, image_list{i_image,2}, ".pvp"]
 if ~exist(image_file, "file")
    i_checkpoint = 1;
    image_file = ...
    [checkpoints_list{i_checkpoint,:}, filesep, image_list{i_image,1}, image_list{i_image,2}, ".pvp"]
 end
 if ~exist(image_file, "file")
    error(["file does not exist: ", image_file]);
 end
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
    end 
 end
 weights1_2_dir = [output_dir, filesep, "weights1_2"]
 [status, msg, msgid] = mkdir(weights1_2_dir);
 if status ~= 1
    warning(["mkdir(", weights1_2_dir, ")", " msg = ", msg]);
 end 
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
      end
      weights1_2_fid = fopen(weights1_2_file);
      weights1_2_hdr{i_weights1_2} = readpvpheader(weights1_2_fid);    
      fclose(weights1_2_fid);

      weight1_2_time = weights1_2_hdr{i_weights1_2}.time;
      if weight1_2_time > max_weight1_2_time
         max_weight1_2_time = weight1_2_time;
         max_checkpoint = i_checkpoint;
      end
   end %% i_checkpoint

   for i_checkpoint = 1 : num_checkpoints
      if i_checkpoint ~= max_checkpoint 
         %%continue;
      end
      checkpoint_dir = checkpoints_list{i_checkpoint,:};
      weights1_2_file = ...
      [checkpoint_dir, filesep, weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, ".pvp"]
      if ~exist(weights1_2_file, "file")
         warning(["file does not exist: ", weights1_2_file]);
         continue;
      end
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
      end

      %% get weight 1->0 file
      i_weights0_1 = i_weights1_2;
      weights0_1_file = ...
      [checkpoint_dir, filesep, weights0_1_list{i_weights0_1,1}, weights0_1_list{i_weights0_1,2}, ".pvp"]
      if ~exist(weights0_1_file, "file")
         warning(["file does not exist: ", weights0_1_file]);
         continue;
      end
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
      end
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
      if analyze_Sparse_flag
         tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
         tmp_rank = [];
      end
      if analyze_Sparse_flag && ~isempty(tmp_rank)
         pre_hist_rank = tmp_rank;
      else
         pre_hist_rank = (1:weights1_2_hdr{i_weights1_2}.nf);
      end

      if exist("labelWeights_list") && length(labelWeights_list) >= i_weights1_2 && ...
         ~isempty(labelWeights_list{i_weights1_2}) && ...
         plot_labelWeights_flag && ...
         i_checkpoint == max_checkpoint
         labelWeights_file = ...
         [checkpoint_dir, filesep, ...
         labelWeights_list{i_weights1_2,1}, labelWeights_list{i_weights1_2,2}, ".pvp"]
         if ~exist(labelWeights_file, "file")
            warning(["file does not exist: ", labelWeights_file]);
            continue;
   end
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
end


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
if plot_weights0_2_flag && i_checkpoint == max_checkpoint
   weights1_2_fig = figure;
   set(weights1_2_fig, "name", weights1_2_name);
end
max_shrinkage = 8; %% 
weight_patch0_2_array = [];
for kf_pre1_2_rank = 1  : num_patches0_2
   kf_pre1_2 = pre_hist_rank(kf_pre1_2_rank);
   if plot_weights0_2_flag && i_checkpoint == max_checkpoint
      subplot(num_patches0_2_rows, num_patches0_2_cols, kf_pre1_2_rank); 
   end
   if ndims(weights1_2_vals) == 4
      patch1_2_tmp = squeeze(weights1_2_vals(:,:,:,kf_pre1_2));
   elseif ndims(weights1_2_vals) == 3
      patch1_2_tmp = squeeze(weights1_2_vals(:,:,kf_pre1_2));
      patch1_2_tmp = reshape(patch1_2_tmp, [1,1,1,size(weights1_2_vals,2)]);
   elseif ndims(weights1_2_vals) == 2
      patch1_2_tmp = squeeze(weights1_2_vals(:,kf_pre1_2));
      patch1_2_tmp = reshape(patch1_2_tmp, [1,1,1,size(weights1_2_vals,2)]);
   end
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
         end
         if weights0_1_nfp == 1
            weights0_1_patch = squeeze(weights0_1_vals(:,:,kf_post1_2));
         else
            weights0_1_patch = squeeze(weights0_1_vals(:,:,:,kf_post1_2));
         end
         %%  store weights0_1_patch by column
         patch0_2_array{weights1_2_patch_row, weights1_2_patch_col} = ...
         patch0_2_array{weights1_2_patch_row, weights1_2_patch_col} + ...
         patch1_2_weight .* ...
         weights0_1_patch;
      end %% kf_post1_2
      row_start = 1+image2post_ny_ratio*(weights1_2_patch_row-1);
      row_end = image2post_ny_ratio*(weights1_2_patch_row-1)+weights0_1_nyp;
      col_start = 1+image2post_nx_ratio*(weights1_2_patch_col-1);
      col_end = image2post_nx_ratio*(weights1_2_patch_col-1)+weights0_1_nxp;
      patch0_2(row_start:row_end, col_start:col_end, :) = ...
      patch0_2(row_start:row_end, col_start:col_end, :) + ...
      patch0_2_array{weights1_2_patch_row, weights1_2_patch_col};
   end %% weights1_2_patch_col
end %% weights1_2_patch_row
patch_tmp2 = flipdim(permute(patch0_2, [2,1,3]),1);
patch_tmp3 = patch_tmp2;
weights0_2_nyp_shrunken = size(patch_tmp3, 1);
patch_tmp4 = patch_tmp3(1, :, :);
while ~any(patch_tmp4(:)) %% && ((weights0_2_nyp - weights0_2_nyp_shrunken) <= max_shrinkage/2)
     weights0_2_nyp_shrunken = weights0_2_nyp_shrunken - 1;
     patch_tmp3 = patch_tmp3(2:weights0_2_nyp_shrunken, :, :);
     patch_tmp4 = patch_tmp3(1, :, :);
  end
  weights0_2_nyp_shrunken = size(patch_tmp3, 1);
  patch_tmp4 = patch_tmp3(weights0_2_nyp_shrunken, :, :);
  while ~any(patch_tmp4(:))
        weights0_2_nyp_shrunken = weights0_2_nyp_shrunken - 1;
        patch_tmp3 = patch_tmp3(1:weights0_2_nyp_shrunken, :, :);
        patch_tmp4 = patch_tmp3(weights0_2_nyp_shrunken, :, :);
     end
     weights0_2_nxp_shrunken = size(patch_tmp3, 2);
     patch_tmp4 = patch_tmp3(:, 1, :);
     while ~any(patch_tmp4(:)) %% && ((weights0_2_nyp - weights0_2_nyp_shrunken) <= max_shrinkage/2)
           weights0_2_nxp_shrunken = weights0_2_nxp_shrunken - 1;
           patch_tmp3 = patch_tmp3(:, 2:weights0_2_nxp_shrunken, :);
           patch_tmp4 = patch_tmp3(:, 1, :);
        end
        weights0_2_nxp_shrunken = size(patch_tmp3, 2);
        patch_tmp4 = patch_tmp3(:, weights0_2_nxp_shrunken, :);
        while ~any(patch_tmp4(:))
              weights0_2_nxp_shrunken = weights0_2_nxp_shrunken - 1;
              patch_tmp3 = patch_tmp3(:, 1:weights0_2_nxp_shrunken, :);
              patch_tmp4 = patch_tmp3(:, weights0_2_nxp_shrunken, :);
           end
           min_patch = min(patch_tmp3(:));
           max_patch = max(patch_tmp3(:));
           patch_tmp5 = ...
           uint8((flipdim(patch_tmp3,1) - min_patch) * 255 / ...
           (max_patch - min_patch + ((max_patch - min_patch)==0)));

           if plot_weights0_2_flag && i_checkpoint == max_checkpoint
                 imagesc(patch_tmp5); 
                 if weights0_1_nfp == 1
                    colormap(gray);
     end
     box off
     axis off
     axis image
  end
  if plot_labelWeights_flag && i_checkpoint == max_checkpoint
     if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
        [~, max_label] = max(squeeze(labelWeights_vals(:,kf_pre1_2)));
        text(weights0_2_nyp_shrunken/2, -weights0_2_nxp_shrunken/6, num2str(max_label-1), "color", [1 0 0]);
     end %% ~empty(labelWeights_vals)
     %%drawnow;
  end %% plot_weights0_2_flag && i_checkpoint == max_checkpoint

  if isempty(weight_patch0_2_array)
     weight_patch0_2_array = ...
     zeros(num_patches0_2_rows*weights0_2_nyp_shrunken, ...
     num_patches0_2_cols*weights0_2_nxp_shrunken, weights0_1_nfp);
  end
  col_ndx = 1 + mod(kf_pre1_2_rank-1, num_patches0_2_cols);
  row_ndx = 1 + floor((kf_pre1_2_rank-1) / num_patches0_2_cols);
  weight_patch0_2_array((1+(row_ndx-1)*weights0_2_nyp_shrunken):(row_ndx*weights0_2_nyp_shrunken), ...
  (1+(col_ndx-1)*weights0_2_nxp_shrunken):(col_ndx*weights0_2_nxp_shrunken),:) = ...
  patch_tmp5;

  %% Plot the average movie weights for a label %%
  if plot_labelWeights_flag && i_checkpoint == max_checkpoint
     if ~isempty(labelWeights_vals) 
        if ~isempty(labeledWeights0_2{max_label})
           labeledWeights0_2{max_label} = labeledWeights0_2{max_label} + double(patch_tmp5);
       else
          labeledWeights0_2{max_label} = double(patch_tmp5);
       end
    end %%  ~isempty(labelWeights_vals) 
 end %% plot_weights0_2_flag && i_checkpoint == max_checkpoint

      end %% kf_pre1_2_ank

      if plot_weights0_2_flag && i_checkpoint == max_checkpoint
         saveas(weights1_2_fig, [weights1_2_dir, filesep, weights1_2_name, ".png"], "png");
      end
      if plot_labelWeights_flag && i_checkpoint == max_checkpoint && ~isempty(labelWeights_vals) 
         labeledWeights_str = ...
         ["labeledWeights_", ...
         weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, ...
         "_", num2str(weight_time, "%08d")];
         labeledWeights_fig = figure("name", labeledWeights_str);
         rows_labeledWeights = ceil(sqrt(size(labelWeights_vals,1)));
         cols_labeledWeights = ceil(size(labelWeights_vals,1) / rows_labeledWeights);
         for label = 1:size(labelWeights_vals,1)
            subplot(rows_labeledWeights, cols_labeledWeights, label);
            labeledWeights_subplot_str = ...
            [num2str(label, "%d")];
            imagesc(squeeze(labeledWeights0_2{label}));
            axis off
            title(labeledWeights_subplot_str);
   end %% label = 1:size(labelWeights_vals,1)
   saveas(labeledWeights_fig,  [weights_dir, filesep, labeledWeights_str, ".png"], "png");
end %%  ~isempty(labelWeights_vals) 

imwrite(uint8(weight_patch0_2_array), [weights1_2_movie_dir, filesep, weights1_2_name, ".png"], "png");
if i_checkpoint == max_checkpoint
   save("-mat", ...
   [weights1_2_movie_dir, filesep, weights1_2_name, ".mat"], ...
   "weight_patch0_2_array");
end


%% make histogram of all weights
if plot_weights0_2_flag && i_checkpoint == max_checkpoint
   weights1_2_hist_fig = figure;
   [weights1_2_hist, weights1_2_hist_bins] = hist(weights1_2_vals(:), 100);
   bar(weights1_2_hist_bins, log(weights1_2_hist+1));
   set(weights1_2_hist_fig, "name", ...
   ["Hist_", weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, "_", ...
   num2str(weights1_2_time, "%08d")]);
   saveas(weights1_2_hist_fig, ...
   [weights1_2_dir, filesep, "weights1_2_hist_", weights1_2_list{i_weights1_2,1}, weights1_2_list{i_weights1_2,2}, "_", ...
   num2str(weights1_2_time, "%08d")], "png");
end

%% plot average labelWeights for each label
if ~isempty(labelWeights_vals) && ...
       ~isempty(labelWeights_time) && ...
       plot_weights0_2_flag && ...
       i_checkpoint == max_checkpoint

       %% plot label weights as matrix of column vectors
       ranked_labelWeights = labelWeights_vals(:, pre_hist_rank(1:num_patches0_2));
       [~, max_label] = max(ranked_labelWeights,[],1);
       [max_label_sorted, max_label_ndx] = sort(max_label);
       label_weights_fig = figure;
       imagesc(ranked_labelWeights(:,max_label_ndx))
       label_weights_str = ...
       ["LabelWeights_", labelWeights_list{i_weights1_2,1}, labelWeights_list{i_weights1_2,2}, ...
       "_", num2str(labelWeights_time, "%08d")];
       %%title(label_weights_fig, label_weights_str);
       figure(label_weights_fig, "name", label_weights_str); 
       title(label_weights_str);
       saveas(label_weights_fig, [weights_dir, filesep, label_weights_str, ".png"] , "png");

    end  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

 end %% i_checkpoint

  end %% i_weights

end  %% plot_weights





deRecon_flag = false; %%true && ~isempty(labelWeights_vals);
if deRecon_flag
   num_deRecon = 3;
   deRecon_sparse_ndx = 2;
   deRecon_struct = Sparse_struct_array{deRecon_sparse_ndx};
   num_deRecon_frames = size(deRecon_struct,1);
   Recon_dir = [output_dir, filesep, "Recon"];
   for i_deRecon_frame = 1 : num_deRecon_frames
      deRecon_time = deRecon_struct{i_deRecon_frame}.time
      deRecon_indices = deRecon_struct{i_deRecon_frame}.values(:,1);
      deRecon_vals = deRecon_struct{i_deRecon_frame}.values(:,2);
      [deRecon_vals_sorted, deRecon_vals_rank] = sort(deRecon_vals, "descend");
      deRecon_indices_sorted = deRecon_indices(deRecon_vals_rank)+1;
      num_deRecon_indices = length(deRecon_indices(:));
      deRecon_hist_rank = Sparse_hist_rank_array{deRecon_sparse_ndx}(:);
      for i_deRecon_index = 1 : min(num_deRecon, num_deRecon_indices)
         deRecon_rank = find(deRecon_hist_rank == deRecon_indices_sorted(i_deRecon_index))
         if deRecon_rank > num_patches0_2
            continue;
      end
      col_ndx = 1 + mod(deRecon_rank-1, num_patches0_2_cols);
      row_ndx = 1 + floor((deRecon_rank-1) / num_patches0_2_cols);
      row_indices = (1+(row_ndx-1)*weights0_2_nyp_shrunken):(row_ndx*weights0_2_nyp_shrunken);
      col_indices = (1+(col_ndx-1)*weights0_2_nxp_shrunken):(col_ndx*weights0_2_nxp_shrunken);
      deRecon_patch = weight_patch0_2_array(row_indices, col_indices, :);
      fh_deRecon = figure;
      imagesc(deRecon_patch);
      box off;
      axis off;
      deRecon_name = [Recon_list{3,2}, "_", num2str(deRecon_time, "%9i"), "_", num2str(i_deRecon_index)];
      set(fh_deRecon, "name", deRecon_name);
      saveas(fh_deRecon, [Recon_dir, filesep, deRecon_name, ".png"], "png");
      fh_deRecon_label = figure;
      bar(labelWeights_vals(:, deRecon_indices_sorted(i_deRecon_index)));
      set(fh_deRecon_label, "name", [deRecon_name, "_", "bar"]);
      saveas(fh_deRecon_label, [Recon_dir, filesep, deRecon_name, "_", "bar", ".png"], "png");
   end %% i_deRecon_index
   %%    disp(mat2str(labelWeights_vals(:,deRecon_rank(1: min(num_deRecon, num_deRecon_indices))));
   deRecon_labelWeights = labelWeights_vals(:,deRecon_indices_sorted);
   deRecon_label_activity = repmat(deRecon_vals_sorted(:)',[size(labelWeights_vals,1),1]);
   deRecon_label_prod = deRecon_labelWeights .* deRecon_label_activity;
   deRecon_vals_sorted
   sum_labelWeights = sum(deRecon_label_prod,2)
end %% i_deRecon
end %% deReconFlag







%%keyboard;
plot_weightsN_Nplus1 = false;
weightsN_Nplus1_list = {};
layersN_Nplus1_list = {};
if strcmp(run_type, "color_deep") || strcmp(run_type, "noTopDown")
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %% deep list
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(run_type, "CIFAR_deep") 
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %% CIFAR_noTask_deep list
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %% list of weights from layerN to layer0 (Image)
   if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      weightsN_Nplus1_list = ...
      {["w15_"], ["V4ToError2_4"], ["w9_"], ["V2ToError1_2"], ["w1_"], ["V1ToError"]};
      layersN_Nplus1_list = ...
      {["a12_"], ["V4"], ["a5_"], ["V2"], ["a2_"], ["V1"], ["a0_"], ["Image"]};
      labelWeights_list = ...
      {["w26_"], ["V4ToLabelError"]};
  else
     checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
     weightsN_Nplus1_list = ...
     {["V4ToError2_4"], ["_W"], ["V2ToError1_2"], ["_W"], ["V1ToError"], ["_W"]};
     layersN_Nplus1_list = ...
     {["V4"], ["_A"], ["V2"], ["_A"], ["V1"], ["_A"], ["Image"], ["_A"]};
     labelWeights_list = ...
     {["V4ToLabelError"], ["_W"]};
  end %% checkpoint_weights_movie
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
end %% run_type
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_weightsN_Nplus1_list = size(weightsN_Nplus1_list,1);
num_layersN_Nplus1_list = size(layersN_Nplus1_list,2)/2;
if num_weightsN_Nplus1_list == 0
   plot_weightsN_Nplus1 = false;
   warning(["num_weightsN_Nplus1_list == 0"]);  
elseif size(weightsN_Nplus1_list,2)/2 ~= num_layersN_Nplus1_list-1;
   plot_weightsN_Nplus1 = false;
   warning(["num_weightsN_Nplus1_list ~= num_layersN_Nplus1_list-1", ...
   ", num_weightsN_Nplus1_list = ", num2str(num_weightsN_Nplus1_list), ...
   ", num_layersN_Nplus1_list = ", num2str(num_layersN_Nplus1_list)]);
end
if num_weightsN_Nplus1_list == 0
   plot_weightsN_Nplus1 = false;
end

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
    end 
 end
 weightsN_Nplus1_dir = [output_dir, filesep, "weightsN_Nplus1"]
 [status, msg, msgid] = mkdir(weightsN_Nplus1_dir);
 if status ~= 1
    warning(["mkdir(", weightsN_Nplus1_dir, ")", " msg = ", msg]);
 end 
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
       if ~exist(weightsN_Nplus1_file, "file")
          warning(["file does not exist: ", weightsN_Nplus1_file]);
          continue;
      end
      weightsN_Nplus1_fid = fopen(weightsN_Nplus1_file);
      weightsN_Nplus1_hdr{i_weights1_2} = readpvpheader(weightsN_Nplus1_fid);    
      fclose(weightsN_Nplus1_fid);

      weightN_Nplus1_time = weightsN_Nplus1_hdr{i_weightN_Nplus1}.time;
      if weightN_Nplus1_time > max_weightN_Nplus1_time
         max_weightN_Nplus1_time = weightN_Nplus1_time;
         max_checkpoint = i_checkpoint;
      end
   end %% i_checkpoint

   %% get weights headers
   checkpoint_dir = checkpoints_list{max_checkpoint,:};
   for i_layerN_Nplus1 = 1 : num_layersN_Nplus1_list-1
      weightsN_Nplus1_file = ...
      [checkpoint_dir, filesep, ...
      weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2-1}, ...
      weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2}, ".pvp"]
      if ~exist(weightsN_Nplus1_file, "file")
         error(["file does not exist: ", weightsN_Nplus1_file]);
      end
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
   end %% i_layerN_Nplus1

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
      end
      layersN_Nplus1_fid = fopen(layersN_Nplus1_file);
      layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1} = readpvpheader(layersN_Nplus1_fid);
      fclose(layersN_Nplus1_fid);
      layersN_Nplus1_nx(i_weightN_Nplus1, i_layerN_Nplus1) = ...
      layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.nx;
      layersN_Nplus1_ny(i_weightN_Nplus1, i_layerN_Nplus1) = ...
      layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.ny;
      layersN_Nplus1_nf(i_weightN_Nplus1, i_layerN_Nplus1) = ...
      layersN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.nf;
   end %% i_layerN_Nplus1

   %% labels (if present)
   checkpoint_dir = checkpoints_list{max_checkpoint,:};
   if length(labelWeights_list) >= i_weightN_Nplus1 && ...
      ~isempty(labelWeights_list{i_weightN_Nplus1}) && ...
      plot_flag 
      labelWeights_file = ...
      [checkpoint_dir, filesep, ...
      labelWeights_list{i_weightN_Nplus1,1}, labelWeights_list{i_weightN_Nplus1,2}, ".pvp"]
      if ~exist(labelWeights_file, "file")
         warning(["file does not exist: ", labelWeights_file]);
         continue;
      end
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
   end %% labels


   %% get rank order of presynaptic elements
   i_layerN_Nplus1 = 1;
   tmp_ndx = sparse_ndx(i_weightN_Nplus1);
   if analyze_Sparse_flag
      tmp_rank = Sparse_hist_rank_array{tmp_ndx};
   else
      tmp_rank = [];
   end
   if analyze_Sparse_flag && ~isempty(tmp_rank)
      pre_hist_rank = tmp_rank;
   else
      pre_hist_rank = (1:weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.nf);
   end


   %% loop over checkpoints
   for i_checkpoint = 1 : num_checkpoints
      checkpoint_dir = checkpoints_list{i_checkpoint,:};

      %% re-initialize patch sizes throughout hierarchy since these are modified during recursive deconvolution
      for i_layerN_Nplus1 = 1 : num_layersN_Nplus1_list-1
         weightsN_Nplus1_file = ...
         [checkpoint_dir, filesep, ...
         weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2-1}, ...
         weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2}, ".pvp"]
         if ~exist(weightsN_Nplus1_file, "file")
            warning(["file does not exist: ", weightsN_Nplus1_file]);
            continue;
   end
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
end %% i_layerN_Nplus1

%% read the top layer of weights to initialize weightsN_Nplus1_vals
i_layerN_Nplus1 = 1;
weightsN_Nplus1_file = ...
[checkpoint_dir, filesep, ...
weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2-1}, ...
weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2}, ".pvp"]
if ~exist(weightsN_Nplus1_file, "file")
   warning(["file does not exist: ", weightsN_Nplus1_file]);
   continue;
end      
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
end
weightsN_Nplus1_time = squeeze(weightsN_Nplus1_struct{i_frame}.time);
weightsN_Nplus1_name = ...
[weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2-1}, ...
weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2}, ...
"_", num2str(weightsN_Nplus1_time, "%08d")];
if no_clobber && ...
       exist([weightsN_Nplus1_movie_dir, filesep, weightsN_Nplus1_name, ".png"]) && ...
       i_checkpoint ~= max_checkpoint
       continue;
    end
    num_weightsN_Nplus1_dims = ...
    ndims(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1});
    num_patchesN_Nplus1 = ...
    size(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1}, num_weightsN_Nplus1_dims);
    num_patchesN_Nplus1_rows = floor(sqrt(min(num_patchesN_Nplus1, max_patches)));
    num_patchesN_Nplus1_cols = ceil(min(num_patchesN_Nplus1, max_patches) / num_patchesN_Nplus1_rows);

    %% loop over lower layers
    for i_layerN_Nplus1 = 1 : num_layersN_Nplus1_list - 2  %% last layer is image

       %% reset patch sizes to reflect most recent deconvolution
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
   end
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
   end
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
   end %% plot_flag

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
     end

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
     end

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
      end
      if weightsNminus1_N_nfp == 1
         weightsNminus1_N_patch = squeeze(weightsNminus1_N_vals(:,:,kf_postN_Nplus1));
      else
         weightsNminus1_N_patch = squeeze(weightsNminus1_N_vals(:,:,:,kf_postN_Nplus1));
      end
      %%  store weightsNminus1_N_patch by column
      patchNminus1_Nplus1_array{N_Nplus1_patch_row, N_Nplus1_patch_col} = ...
      patchNminus1_Nplus1_array{N_Nplus1_patch_row, N_Nplus1_patch_col} + ...
      patchN_Nplus1_weight .* ...
      weightsNminus1_N_patch;
   end %% kf_postN_Nplus1
   Nminus1_Nplus1_row_start = 1+Nminus1_N_ny_ratio*(N_Nplus1_patch_row-1);
   Nminus1_Nplus1_row_end = Nminus1_N_ny_ratio*(N_Nplus1_patch_row-1)+weightsNminus1_N_nyp;
   Nminus1_Nplus1_col_start = 1+Nminus1_N_nx_ratio*(N_Nplus1_patch_col-1);
   Nminus1_Nplus1_col_end = Nminus1_N_nx_ratio*(N_Nplus1_patch_col-1)+weightsNminus1_N_nxp;
   patchNminus1_Nplus1(Nminus1_Nplus1_row_start:Nminus1_Nplus1_row_end, ...
   Nminus1_Nplus1_col_start:Nminus1_Nplus1_col_end, :) = ...
   patchNminus1_Nplus1(Nminus1_Nplus1_row_start:Nminus1_Nplus1_row_end, ...
   Nminus1_Nplus1_col_start:Nminus1_Nplus1_col_end, :) + ...
   patchNminus1_Nplus1_array{N_Nplus1_patch_row, N_Nplus1_patch_col};
end %% N_Nplus1_patch_col
     end %% N_Nplus1_patch_row

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
        end
        weightsNminus1_Nplus1_nyp_shrunken = size(patch_tmp3, 1);
        patch_tmp4 = patch_tmp3(weightsNminus1_Nplus1_nyp_shrunken, :, :);
        while ~any(patch_tmp4(:))
              weightsNminus1_Nplus1_nyp_shrunken = weightsNminus1_Nplus1_nyp_shrunken - 1;
              patch_tmp3 = patch_tmp3(1:weightsNminus1_Nplus1_nyp_shrunken, :, :);
              patch_tmp4 = patch_tmp3(weightsNminus1_Nplus1_nyp_shrunken, :, :);
           end
           weightsNminus1_Nplus1_nxp_shrunken = size(patch_tmp3, 2);
           patch_tmp4 = patch_tmp3(:, 1, :);
           while ~any(patch_tmp4(:)) %% && ((weightsNminus1_Nplus1_nyp - weightsNminus1_Nplus1_nyp_shrunken) <= max_shrinkage/2)
                 weightsNminus1_Nplus1_nxp_shrunken = weightsNminus1_Nplus1_nxp_shrunken - 1;
                 patch_tmp3 = patch_tmp3(:, 2:weightsNminus1_Nplus1_nxp_shrunken, :);
                 patch_tmp4 = patch_tmp3(:, 1, :);
              end
              weightsNminus1_Nplus1_nxp_shrunken = size(patch_tmp3, 2);
              patch_tmp4 = patch_tmp3(:, weightsNminus1_Nplus1_nxp_shrunken, :);
              while ~any(patch_tmp4(:))
                    weightsNminus1_Nplus1_nxp_shrunken = weightsNminus1_Nplus1_nxp_shrunken - 1;
                    patch_tmp3 = patch_tmp3(:, 1:weightsNminus1_Nplus1_nxp_shrunken, :);
                    patch_tmp4 = patch_tmp3(:, weightsNminus1_Nplus1_nxp_shrunken, :);
                 end
     else
        nyp_shift = floor((weightsNminus1_Nplus1_nyp - weightsNminus1_Nplus1_nyp_shrunken)/2);
        nxp_shift = floor((weightsNminus1_Nplus1_nxp - weightsNminus1_Nplus1_nxp_shrunken)/2);
        patch_tmp3 = ...
        patchNminus1_Nplus1(nyp_shift+1:weightsNminus1_Nplus1_nyp_shrunken, ...
        nxp_shift+1:weightsNminus1_Nplus1_nxp_shrunken, :);
     end  %% kf_preN_Nplus1_rank == 1

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
       end
       box off
       axis off
       axis image
       if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
          [~, max_label] = max(squeeze(labelWeights_vals(:,kf_preN_Nplus1)));
          text(weightsNminus1_Nplus1_nyp_shrunken/2, ...
          -weightsNminus1_Nplus1_nxp_shrunken/6, ...
          num2str(max_label-1), "color", [1 0 0]);
       end %% ~empty(labelWeights_vals)
       %%drawnow;
    end %% plotNminus1_Nplus1_flag 

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
    end
    if  kf_preN_Nplus1_rank <= max_patches
       col_ndx = 1 + mod(kf_preN_Nplus1_rank-1, num_patchesN_Nplus1_cols);
       row_ndx = 1 + floor((kf_preN_Nplus1_rank-1) / num_patchesN_Nplus1_cols);
       weightsNminus1_Nplus1_array((1+(row_ndx-1)*weightsNminus1_Nplus1_nyp_shrunken):...
       (row_ndx*weightsNminus1_Nplus1_nyp_shrunken), ...
       (1+(col_ndx-1)*weightsNminus1_Nplus1_nxp_shrunken): ...
       (col_ndx*weightsNminus1_Nplus1_nxp_shrunken),:) = ...
       patch_tmp5;
    end

    %% set weightsN_Nplus1_vals to patch_tmp3
    %%	  weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1+1}(:, :, :, kf_preN_Nplus1_rank) = ...
    %%	      patch_tmp3;
    weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1+1}(:, :, :, kf_preN_Nplus1) = ...
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
       end
    end %%  ~isempty(labelWeights_vals) 

 end %% kf_preN_Nplus1_rank

 if plot_flag && ...
         i_checkpoint == max_checkpoint && ...
         i_layerN_Nplus1 == num_layersN_Nplus1_list-2 
         saveas(weightsNminus1_Nplus1_fig, [weightsN_Nplus1_dir, filesep, weightsN_Nplus1_name, ".png"], "png");
         if ~isempty(labelWeights_vals) 
            labeledWeights_str = ...
            ["labeledWeights_", ...
            weightsN_Nplus1_list{i_weightN_Nplus1,1}, weightsN_Nplus1_list{i_weightN_Nplus1,2}, ...
            "_", num2str(weightsN_Nplus1_time, "%08d")];
            labeledWeights_fig = figure(labeledWeights_fig, "name", labeledWeights_str);
            title(labeledWeights_str);
            rows_labeledWeights = ceil(sqrt(size(labelWeights_vals,1)));
            cols_labeledWeights = ceil(size(labelWeights_vals,1) / rows_labeledWeights);
            for label = 1:size(labelWeights_vals,1)
               subplot(rows_labeledWeights, cols_labeledWeights, label);
               labeledWeights_subplot_str = ...
               [num2str(label, "%d")];
               imagesc(squeeze(labeledWeightsNminus1_Nplus1{label}));
               title(labeledWeights_subplot_str);
               axis off
            end %% label = 1:size(labelWeights_vals,1)
            saveas(labeledWeights_fig,  [weightsN_Nplus1_dir, filesep, labeledWeights_str, ".png"], "png");
    end %%  ~isempty(labelWeights_vals) 
 end
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
   end %% plotNminus1_Nplus1_flag

   %% plot average labelWeights for each label
   if ~isempty(labelWeights_vals) && ...
         ~isempty(labelWeights_time) && ...
         plot_flag && ...
         i_checkpoint == max_checkpoint && ...
         i_layerN_Nplus1 == num_layersN_Nplus1_list-2 

         %% plot label weights as matrix of column vectors
         ranked_labelWeights = labelWeights_vals(:, pre_hist_rank(1:num_patchesN_Nplus1));
         [~, max_label] = max(ranked_labelWeights,[],1);
         [max_label_sorted, max_label_ndx] = sort(max_label, "ascend");
         label_weights_str = ...
         ["LabelWeights_", ...
         labelWeights_list{i_weightN_Nplus1,1}, labelWeights_list{i_weightN_Nplus1,2}, "_", ...
         num2str(labelWeights_time, "%08d")];
         label_weights_fig = figure("name", label_weights_str);;
         imagesc(ranked_labelWeights(:,max_label_ndx))
         %%title(label_weights_fig, label_weights_str);
         title(label_weights_str);
         saveas(label_weights_fig, [weightsN_Nplus1_dir, filesep, label_weights_str, ".png"] , "png");
   end  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

end %% i_checkpoint

    end %% i_layerN_Nplus1

 end %% i_weightN_Nplus1

end  %% plot_weights
