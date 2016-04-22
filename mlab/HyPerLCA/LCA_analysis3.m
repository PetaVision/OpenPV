
clear all;
close all;
more off
pkg load all
global plot_flag %% if true, plot graphical output to screen, else do not generate graphical outputy
plot_flag = true; %%
global load_flag %% if true, then load "saved" data structures rather than computing them 
load_Sparse_flag = false;
if plot_flag
  setenv("GNUTERM","X11")
endif
no_clobber = false; %%true;

%% machine/run_type environment
if ismac
  workspace_path = "/Users/gkenyon/openpv";
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%run_type = "Grains";
  %%run_type = "ICA";
  %%ICA_subtype = "ICAX4";
  run_type = "scene";
  %%run_type = "experts";
  %%run_type = "MaxPool";
  %%run_type = "DCA";
  if strcmp(run_type, "Grains")
    output_dir = "/Volumes/mountData/Grains/Grains_S1_128/test3"; %%
    checkpoint_parent = "/Volumes/mountData/Grains/Grains_S1_128"; %%
    checkpoint_children = ...
    {"test3"};
  elseif strcmp(run_type, "scene")
    output_dir = "/Volumes/mountData/scene/scene_S1X4_1536_ICA/mount_rushmore1"; %%
    checkpoint_parent = "/Volumes/mountData/scene/scene_S1X4_1536_ICA"; %%
    checkpoint_children = ...
    {"mount_rushmore1"};
  elseif strcmp(run_type, "ICA")
    if ~exist("ICA_subtype", "var")
      output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_1536_ICA/VOC2007_landscape8";
      checkpoint_parent = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_1536_ICA";
      checkpoint_children = ...
      {"VOC2007_landscape8"};
    elseif strcmp(ICA_subtype, "ICAX4")
      output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1X4_1536_ICA/VOC2007_landscape1";
      checkpoint_parent = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1X4_1536_ICA";
      checkpoint_children = ...
      {"VOC2007_landscape1"};
    endif
  elseif strcmp(run_type, "experts")
    output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_16_8_4_experts/VOC2007_landscape";
    checkpoint_parent = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_16_8_4_experts";
    checkpoint_children = ...
    {"VOC2007_landscape"};
  elseif strcmp(run_type, "DCA")
    output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DCNN/VOC2007_landscape12";
    checkpoint_parent = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DCNN";
    checkpoint_children = {"VOC2007_landscape12"}; %%
  elseif strcmp(run_type, "MaxPool")
    output_dir = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_MaxPool/VOC2007_landscape6";
    checkpoint_parent = "/Volumes/mountData/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_MaxPool";
    checkpoint_children = ...
    {"VOC2007_landscape6"};
  endif
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif isunix
  workspace_path = "/nh/compneuro/Data/openpv";
  
  %%run_type = "Heli_DPTX3";
  %%run_type = "PASCAL_DPT";
  %%run_type = "MRI";
  %%run_type = "M1";
  %%run_type = "SLP";
  %%run_type = "PCA";
  %%run_type = "dSCANN";
  %%run_type = "DCNN";
  %%run_type = "CIFAR";
  %%run_type = "MaxPool";
  %%run_type = "DCA";
  %%run_type = "DCNNX3";
  %%run_type = "DBN";
  run_type = "experts";
  if strcmp(run_type, "experts") 
    output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_16_8_4_experts/VOC2007_landscape2";
    checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_16_8_4_experts";
    checkpoint_children = {"VOC2007_landscape2"}; %%
  elseif strcmp(run_type, "DCA")
    output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DCA/VOC2007_landscape2";
    checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DCA";
    checkpoint_children = {"VOC2007_landscape2"}; %%
  elseif strcmp(run_type, "MaxPool")
    output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_MaxPool/VOC2007_landscape7";
    checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_MaxPool";
    checkpoint_children = {"VOC2007_landscape7"}; %%
  elseif strcmp(run_type, "CIFAR")
    output_dir = "/nh/compneuro/Data/CIFAR/CIFAR_S1_48_S2_96_S3_48_DCA/CIFAR10_train2";
    checkpoint_parent = "/nh/compneuro/Data/CIFAR/CIFAR_S1_48_S2_96_S3_48_DCA";
    checkpoint_children = {"CIFAR10_train2"}; %%
  elseif strcmp(run_type, "SLP")
    output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_96_S2_1536/VOC2007_landscape2";
    checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_96_S2_1536";
    checkpoint_children = {"VOC2007_landscape26"}; %%
  elseif strcmp(run_type, "PCA")
    output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_96_S2_1536_PCA_4X3_1536/VOC2007_landscape16";
    checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_96_S2_1536_PCA_4X3_1536";
    checkpoint_children = {"VOC2007_landscape16"}; %%
  elseif strcmp(run_type, "dSCANN")
    output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_dSCANN/VOC2007_landscape2";
    checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_dSCANN";
    checkpoint_children = {"VOC2007_landscape2"}; %%
  elseif strcmp(run_type, "DBN")
    output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DBN/VOC2007_landscape";
    checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DBN";
    checkpoint_children = {"VOC2007_landscape"}; %%
  elseif strcmp(run_type, "DCNN")
    %%   output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DCNN/VOC2007_landscape7";
    %%   checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DCNN";
    %%   checkpoint_children = {"VOC2007_landscape7"}; %%
       output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DCNN/PASCAL_Vine11";
       checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1_128_S2_256_S3_512_DCNN";
       checkpoint_children = {"PASCAL_Vine11"}; %%
    %%    output_dir = "/nh/compneuro/Data/Grains/Grains_S1_128_S2_256_S3_512_DCNN/test4";
    %%    checkpoint_parent = "/nh/compneuro/Data/Grains/Grains_S1_128_S2_256_S3_512_DCNN";
    %%    checkpoint_children = {"test4"}; %%
    %%output_dir = "/nh/compneuro/Data/CIFAR/CIFAR_S1_128_S2_256_S3_512_DCNN/test5";
    %%checkpoint_parent = "/nh/compneuro/Data/CIFAR/CIFAR_S1_128_S2_256_S3_512_DCNN";
    %%checkpoint_children = {"test5"}; %%
  elseif strcmp(run_type, "DCNNX3")
    output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1X3_128_S2X2_256_S3_512_DCNN/VOC2007_landscape10";
    checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_S1X3_128_S2X2_256_S3_512_DCNN";
    checkpoint_children = {"VOC2007_landscape10"}; %%
  elseif strcmp(run_type, "MRI")
    output_dir = "/nh/compneuro/Data/MRI/5_subjects/3D";
    checkpoint_parent = "/nh/compneuro/Data/MRI/5_subjects";
    checkpoint_children = {"3D"}; %%
  elseif strcmp(run_type, "M1")
    output_dir = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_M1_160/VOC2007_landscape";
    checkpoint_parent = "/nh/compneuro/Data/PASCAL_VOC/PASCAL_M1_160/";
    checkpoint_children = {"VOC2007_landscape"}; %%
  endif
endif %% isunix
addpath(pwd);
addpath([workspace_path, filesep, "pv-core/mlab/util"]);

%% default paths
if ~exist("output_dir") || isempty(output_dir)
  warning("using default output dir");
  output_dir = pwd
endif
DoG_path = [];
unwhiten_flag = false;  %% set to true if DoG filtering used and dewhitening of reconstructions is desired
if unwhiten_flag && (~exist("DoG_path") || isempty(DoG_path))
  DoG_path = output_dir;
endif

max_patches = 1536/2; %%2*2*192; %%256; %%1024; %%  %% maximum number of weight patches to plot, typically ordered from most to least active if Sparse_flag == true
checkpoint_weights_movie = true; %% make movie of weights over time using list of checkpoint folders getCheckpointList(checkpoint_parent, checkpoint_children)







%% plot Reconstructions
analyze_Recon = true;
if analyze_Recon
  if strcmp(run_type, "SLP")  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SLP list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"],  ["Image"];
	 ["a3_"],  ["ImageReconS1"];
	 ["a7_"],  ["ImageReconS2"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    Recon_unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    Recon_normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    Recon_sum_list = cell(num_Recon_list,1);
  elseif  strcmp(run_type, "default") || strcmp(run_type, "ICA") || strcmp(run_type, "experts") || strcmp(run_type, "MaxPool") || strcmp(run_type, "DCA") || strcmp(run_type, "CIFAR")  || strcmp(run_type, "scene") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% default/glob generated list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_glob_list = glob([output_dir, filesep, "a*_Image*.pvp"]);
    num_Recon_list = length(Recon_glob_list);    
    Recon_list = cell(num_Recon_list,1);
    for i_Recon_list = 1 : num_Recon_list
      [Recon_list_dir, Recon_list_name, Recon_list_ext, Recon_list_ver] = fileparts(Recon_glob_list{i_Recon_list});
      Recon_underscore_ndx = strfind(Recon_list_name, "_");
      Recon_list{i_Recon_list,1} = Recon_list_name(1:Recon_underscore_ndx(1));
      Recon_list{i_Recon_list,2} = Recon_list_name(Recon_underscore_ndx(1)+1:length(Recon_list_name));
    endfor
    num_Recon_list = size(Recon_list,1);
    Recon_unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    Recon_normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    Recon_sum_list = cell(num_Recon_list,1);
  elseif strcmp(run_type, "dSCANN")  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% dSCANN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"],  ["Image"];
	 ["a1_"],  ["ImageDecon"];
	 ["a2_"],  ["ImageRecon"];
	 ["a7_"],  ["ImageDeconS1"];
	 ["a8_"],  ["ImageReconS1"];
	 ["a14_"],  ["ImageDeconS2"];
	 ["a15_"],  ["ImageReconS2"];
	 ["a23_"],  ["ImageReconS3"];
	 ["a24_"],  ["ImageDeconS3"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    Recon_unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    Recon_normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    Recon_sum_list = cell(num_Recon_list,1);
  elseif strcmp(run_type, "DBN")  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DBN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"],  ["Image"];
	 ["a1_"],  ["ImageRecon"];
	 ["a4_"],  ["ImageReconS1"];
	 ["a8_"],  ["ImageReconS2"];
	 ["a13_"],  ["ImageReconS3"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    Recon_unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    Recon_normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    Recon_sum_list = cell(num_Recon_list,1);
  elseif strcmp(run_type, "DCNN")  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"],  ["Image"];
	 ["a1_"],  ["ImageDecon"];
	 ["a5_"],  ["ImageDeconS1"];
	 ["a9_"],  ["ImageDeconS2"];
	 ["a14_"],  ["ImageDeconS3"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    Recon_unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    Recon_normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    Recon_sum_list = cell(num_Recon_list,1);
  elseif strcmp(run_type, "DCNNX3")  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNNX3 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"],  ["Image"];
	 ["a1_"],  ["ImageDecon"];
	 ["a7_"],  ["ImageDeconS1"];
	 ["a12_"],  ["ImageDeconS2"];
	 ["a17_"],  ["ImageDeconS3"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    Recon_unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    Recon_normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    Recon_sum_list = cell(num_Recon_list,1);
  elseif strcmp(run_type, "PCA")  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PCA list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"],  ["Image"];
	 ["a3_"],  ["ImageReconS1"];
	 ["a14_"],  ["ImageReconS2"];
	 ["a34_"],  ["ImageReconPCA"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    Recon_unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    Recon_normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    Recon_sum_list = cell(num_Recon_list,1);
  elseif strcmp(run_type, "MRI")  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MRI list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"],  ["Image0"];
	 ["a1_"],  ["Image1"];
	 ["a2_"],  ["Image2"];
	 ["a3_"],  ["Image3"];
	 ["a9_"],  ["Image0ReconS1"];
	 ["a10_"],  ["Image1ReconS1"];
	 ["a11_"],  ["Image2ReconS1"];
	 ["a12_"],  ["Image3ReconS1"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    Recon_unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    Recon_normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    Recon_sum_list = cell(num_Recon_list,1);
  elseif strcmp(run_type, "M1")  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% M1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Recon_list = ...
	{["a0_"],  ["Image"];
	 ["a5_"],  ["GroundTruthReconM1"]};
    %% list of layers to unwhiten
    num_Recon_list = size(Recon_list,1);
    Recon_unwhiten_list = zeros(num_Recon_list,1);
    %% list of layers to use as a normalization reference for unwhitening
    Recon_normalize_list = 1:num_Recon_list;
    %% list of (previous) layers to sum with current layer
    Recon_sum_list = cell(num_Recon_list,1);
  endif %% run_type
  
  
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
  endif  %% unwhiten_flag
  
  num_Recon_frames_per_layer = 1;
  %%plot_flag = false;
  Recon_LIFO_flag = true;
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
  %%plot_flag = true;
  
endif %% analyze_Recon







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
  elseif  strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    StatsProbe_sigma_flag([4]) = 0;
    StatsProbe_sigma_flag([5]) = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  StatsProbe_nnz_flag = ~StatsProbe_sigma_flag;
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







analyze_Sparse_flag = true;
if analyze_Sparse_flag
  Sparse_frames_list = [];
  if strcmp(run_type, "SLP")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SLP list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a2_"],  ["S1"]; ...
	 ["a5_"],  ["S2"]}; 
  elseif strcmp(run_type, "DCA") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCA list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a5_"],  ["S1"]; ...
	 ["a9_"],  ["S2"]; ...
	 ["a14_"],  ["S3"]; ...
	 ["a20_"],  ["GroundTruth"]}; 
  elseif strcmp(run_type, "MaxPool")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MaxPool list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a3_"],  ["S1"]; ...
	 ["a7_"],  ["S2"]; ...
	 ["a14_"],  ["S3"]; ...
	 ["a22_"],  ["GroundTruth"]}; 
  elseif strcmp(run_type, "CIFAR")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a5_"],  ["S1"]; ...
	 ["a9_"],  ["S2"]; ...
	 ["a14_"],  ["S3"]}; 
  elseif strcmp(run_type, "default") || strcmp(run_type, "ICA") || strcmp(run_type, "experts") || strcmp(run_type, "scene") 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ICA list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(run_type, "ICA")
      Sparse_glob_str = "a*_S*.pvp";
    elseif strcmp(run_type, "scene")
      Sparse_glob_str = "a*_S*.pvp";
    elseif strcmp(run_type, "experts")
      Sparse_glob_str = "a*_S*.pvp";
    elseif strcmp(run_type, "MaxPool")
      Sparse_glob_str = "a*_S?.pvp";
    endif
    Sparse_glob_list = glob([output_dir, filesep, Sparse_glob_str]);
    num_Sparse_list = length(Sparse_glob_list);    
    Sparse_list = cell(num_Sparse_list,1);
    for i_Sparse_list = 1 : num_Sparse_list
      [Sparse_list_dir, Sparse_list_name, Sparse_list_ext, Sparse_list_ver] = fileparts(Sparse_glob_list{i_Sparse_list});
      Sparse_underscore_ndx = strfind(Sparse_list_name, "_");
      Sparse_list{i_Sparse_list,1} = Sparse_list_name(1:Sparse_underscore_ndx(1));
      Sparse_list{i_Sparse_list,2} = Sparse_list_name(Sparse_underscore_ndx(1)+1:length(Sparse_list_name));
    endfor
  elseif strcmp(run_type, "dSCANN")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% dSCANN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a6_"],  ["S1"]; ...
	 ["a11_"],  ["S2"]; ...
	 ["a18_"],  ["S3"]}; 
  elseif strcmp(run_type, "DBN")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DBN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a3_"],  ["S1"]; ...
	 ["a6_"],  ["S2"]; ...
	 ["a10_"],  ["S3"]}; 
  elseif strcmp(run_type, "DCNN")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a4_"],  ["S1"]; ...
	 ["a7_"],  ["S2"]; ...
	 ["a11_"],  ["S3"]}; 
  elseif strcmp(run_type, "DCNNX3")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNNX3 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a6_"],  ["S1"]; ...
	 ["a10_"],  ["S2"]; ...
	 ["a14_"],  ["S3"]}; 
  elseif strcmp(run_type, "PCA")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PCA list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a2_"],  ["S1"]; ...
	 ["a5_"],  ["S201"]; ...
	 ["a6_"],  ["S202"]; ...
	 ["a7_"],  ["S203"]; ...
	 ["a8_"],  ["S204"]; ...
	 ["a9_"],  ["S205"]; ...
	 ["a10_"],  ["S206"]; ...
	 ["a11_"],  ["S207"]; ...
	 ["a12_"],  ["S208"]; ...
	 ["a24_"],  ["PCA"]}; 
 elseif strcmp(run_type, "MRI")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MRI list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a8_"],  ["S1"]}; 
 elseif strcmp(run_type, "M1")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% M1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sparse_list = ...
	{["a2_"],  ["GroundTruth"]; 
	 ["a4_"],  ["M1"]}; 
  endif %% run_type
  num_Sparse_list = length(Sparse_list);    

  fraction_Sparse_frames_read = 1;
  min_Sparse_skip = 1;
  fraction_Sparse_progress = 10;
  num_epochs = 1;
  if load_Sparse_flag
    num_procs = 1;
  else
    num_procs = 1;
  endif
  [Sparse_hdr, ...
   Sparse_hist_rank_array, ...
   Sparse_times_array, ...
   Sparse_percent_active_array, ...
   Sparse_percent_change_array, ...
   Sparse_std_array, ...
   Sparse_struct_array, ...
   Sparse_max_val_array, ...
   Sparse_min_val_array] = ...
  analyzeSparseEpochsPVP3(Sparse_list, ...
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

endif %% plot_Sparse_flag











%% Analyze Xcorr
%%keyboard
analyze_Xcorr = false; %%true;
if analyze_Xcorr
  num_Sparse_list = size(Sparse_list,1);
  if num_Sparse_list ==0
    warning(["analyze_Xcorr:num_Sparse_list == 0"]);
  endif
  xCorr_list = cell(num_Sparse_list,1);
  for i_sparse = 1 : num_Sparse_list
    xCorr_list{i_sparse} = [Sparse_list{i_sparse, 1}, Sparse_list{i_sparse,2}];
  endfor
  plot_corr = true;
  numprocs = 1;
  frames_calc = 0;
  [...
    iidx,             ... % Cell array, length of xCorr_list, contains vectors of point a
   jidx,             ... % Cell array, length of xCorr_list, contains vectors of point b
   finalCorr         ... % Cell array, length of xCorr_list, contains the corr value between a and b, ranked from most to least corr not including self
  ] = ...
  analyzeXCorr(...
		xCorr_list,       ... % A list of activity files to calculate the correlation of 
	       output_dir,       ... % The parent output directory of pv
	       plot_corr,        ... % Flag that determines if a plot should be made
	       frames_calc,      ... % The number of frames to calculate. Will calculate the lastest frame_calc frames. 0 is all frames
	       numprocs          ... % The number of processes to use for parallization
	      );
endif
%%keyboard




analyze_nonSparse_flag = true;
if analyze_nonSparse_flag
  if strcmp(run_type, "experts") 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% default/glog generated list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_glob_list = glob([output_dir, filesep, "a*_*Error*.pvp"]);
    num_nonSparse_list = length(nonSparse_glob_list);    
    nonSparse_list = cell(num_nonSparse_list,1);
    for i_nonSparse_list = 1 : num_nonSparse_list
      [nonSparse_list_dir, nonSparse_list_name, nonSparse_list_ext, nonSparse_list_ver] = fileparts(nonSparse_glob_list{i_nonSparse_list});
      nonSparse_underscore_ndx = strfind(nonSparse_list_name, "_");
      nonSparse_list{i_nonSparse_list,1} = nonSparse_list_name(1:nonSparse_underscore_ndx(1));
      nonSparse_list{i_nonSparse_list,2} = nonSparse_list_name(nonSparse_underscore_ndx(1)+1:length(nonSparse_list_name));
    endfor
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = cell(num_nonSparse_list, 2);
    if strcmp(run_type, "ICA")
      image_num_str = "1";
      if strcmp(ICA_subtype, "ICAX4")
	image_num_str = "11";
      endif
    elseif strcmp(run_type, "experts")
      image_num_str = "121";
    elseif strcmp(run_type, "MaxPool")
      image_num_str = "0";
    elseif strcmp(run_type, "scene")
      image_num_str = "3";
    endif
    for i_nonSparse_norm_list = 1 : num_nonSparse_list
      nonSparse_norm_list{i_nonSparse_norm_list,1} = ...
      ["a", image_num_str, "_"]; %%
      nonSparse_norm_list{i_nonSparse_norm_list,2} = ...
      ["Image"];
    endfor
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength = nonSparse_norm_strength./sqrt(18*18*3);
  elseif strcmp(run_type, "ICA") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ICA list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~exist("ICA_subtype", "var")
      nonSparse_list = ...
      {["a3_"], ["ImageReconS1Error"]; ...
       ["a6_"], ["GroundTruthReconS1Error"]};
    elseif strcmp(ICA_subtype, "ICAX4")
      nonSparse_list = ...
      {["a4_"], ["ImageReconS1Error"]; ...
       ["a1_"], ["GroundTruthReconS1Error"]};
    endif
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    if ~exist("ICA_subtype", "var")
      nonSparse_norm_list = ...
      {["a7_"], ["Image"]; ...
       ["a11_"], ["GroundTruth"]};
      Sparse_std_ndx = [0 1]; 
    elseif strcmp(ICA_subtype, "ICAX4")
      nonSparse_norm_list = ...
      {["a11_"], ["Image"]; ...
       ["a8_"], ["GroundTruth"]};
      Sparse_std_ndx = [0 2]; 
    endif      
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "SLP") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SLP list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a1_"], ["ImageReconS1Error"]; ...
         ["a8_"], ["ImageReconS2Error"]; ...
         ["a4_"], ["S1ReconS2Error"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a_"], ["Image"]; ...
         ["a22_"], ["GroundTruth"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    nonSparse_norm_strength(2) = 1/sqrt(18*18*3);
    Sparse_std_ndx = [0 0 1]; 
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DCA") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCA list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a2_"], ["ImageDeconError"]; ...
         ["a29_"], ["GroundTruthReconS1Error"]; ...
         ["a25_"], ["GroundTruthReconS2Error"]; ...
         ["a21_"], ["GroundTruthReconS3Error"]; ...
         ["a33_"], ["GroundTruthReconS1S2S3Error"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a20_"], ["GroundTruth"]; ...
         ["a20_"], ["GroundTruth"]; ...
         ["a20_"], ["GroundTruth"]; ...
         ["a20_"], ["GroundTruth"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    Sparse_std_ndx = [0 4 4 4 4]; 
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "MaxPool") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MaxPool list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a2_"], ["ImageReconS1Error"]; ...
         ["a6_"], ["S1MaxPoolReconS2Error"]; ...
         ["a13_"], ["S2MaxPoolReconS3Error"]; ...
         ["a31_"], ["GroundTruthReconS1Error"]; ...
         ["a27_"], ["GroundTruthReconS2Error"]; ...
         ["a23_"], ["GroundTruthReconS3Error"]; ...
         ["a35_"], ["GroundTruthReconS1S2S3Error"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a4_"], ["S1MaxPool"]; ...
         ["a11_"], ["S2MaxPool"]; ...
         ["a22_"], ["GroundTruth"]; ...
         ["a22_"], ["GroundTruth"]; ...
         ["a22_"], ["GroundTruth"]; ...
         ["a22_"], ["GroundTruth"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    Sparse_std_ndx = [0 0 0 4 4 4 4]; 
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "CIFAR") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a2_"], ["ImageDeconError"]; ...
         ["a28_"], ["GroundTruthReconS1Error"]; ...
         ["a24_"], ["GroundTruthReconS2Error"]; ...
         ["a20_"], ["GroundTruthReconS3Error"]; ...
         ["a32_"], ["GroundTruthReconS1S2S3Error"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a19_"], ["GroundTruth"]; ...
         ["a19_"], ["GroundTruth"]; ...
         ["a19_"], ["GroundTruth"]; ...
         ["a19_"], ["GroundTruth"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    Sparse_std_ndx = zeros(num_nonSparse_list,1); 
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "dSCANN") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% dSCANN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a3_"], ["ImageDeconError"]; ...
         ["a4_"], ["ImageReconError"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a0_"], ["Image"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    nonSparse_norm_strength(2) = 1/sqrt(18*18*3);
    Sparse_std_ndx = [0 0]; 
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DBN") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DBN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a2_"], ["ImageReconError"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    Sparse_std_ndx = [0]; 
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DCNN") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a2_"], ["ImageDeconError"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    Sparse_std_ndx = [0]; 
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DCNNX3") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNNX3 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a2_"], ["ImageDeconError"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    Sparse_std_ndx = [0]; 
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "PCA") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PCA list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a1_"], ["ImageReconS1Error"]; ...
         ["a15_"], ["ImageReconS2Error"]; ...
         ["a35_"], ["ImageReconPCAError"]; ...
         ["a4_"], ["S1ReconS2Error"]; ...
         ["a16_"], ["S201ReconPCAError"]; ...
         ["a17_"], ["S202ReconPCAError"]; ...
         ["a18_"], ["S203ReconPCAError"]; ...
         ["a19_"], ["S204ReconPCAError"]; ...
         ["a20_"], ["S205ReconPCAError"]; ...
         ["a21_"], ["S206ReconPCAError"]; ...
         ["a22_"], ["S207ReconPCAError"]; ...
         ["a23_"], ["S208ReconPCAError"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a0_"], ["Image"]; ...
         ["a2_"], ["S1"]; ...
         ["a5_"], ["S201"]; ...
         ["a6_"], ["S202"]; ...
         ["a7_"], ["S203"]; ...
         ["a8_"], ["S204"]; ...
         ["a9_"], ["S205"]; ...
         ["a10_"], ["S206"]; ...
         ["a11_"], ["S207"]; ...
         ["a12_"], ["S208"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength(1) = 1/sqrt(18*18*3);
    nonSparse_norm_strength(2) = 1/sqrt(18*18*3);
    Sparse_std_ndx = [0 0 1 2 3 4 5 6 7 8 9]; 
    %%%%%%%%%%%%%%%%%%%%%%%%e%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "MRI") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MRI list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a4_"], ["Image0ReconS1Error"];
	 ["a5_"], ["Image1ReconS1Error"];
	 ["a6_"], ["Image2ReconS1Error"];
	 ["a7_"], ["Image3ReconS1Error"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a0_"], ["Image0"];
	 ["a1_"], ["Image1"];
	 ["a2_"], ["Image2"];
	 ["a3_"], ["Image3"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength = nonSparse_norm_strength./sqrt(18*18*3);
    Sparse_std_ndx = [0 0 0 0]; 
  elseif strcmp(run_type, "M1") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% M1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nonSparse_list = ...
	{["a3_"], ["GroundTruthReconM1Error"]}; 
    num_nonSparse_list = size(nonSparse_list,1);
    nonSparse_skip = repmat(1, num_nonSparse_list, 1);
    nonSparse_norm_list = ...
        {["a2_"], ["GroundTruth"]};
%%    nonSparse_norm_list = {[],[]; [],[]};
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
    nonSparse_norm_strength = nonSparse_norm_strength./sqrt(18*18*3);
    Sparse_std_ndx = [1]; 
  endif %% run_type
  if ~exist("Sparse_std_ndx")
    Sparse_std_ndx = zeros(num_nonSparse_list,1);
  endif
  if ~exist("nonSparse_norm_strength")
    nonSparse_norm_strength = ones(num_nonSparse_list,1);
  endif

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

endif %% analyze_nonSparse_flag







plot_ReconError = false && analyze_nonSparse_flag;
ReconError_RMS_fig_ndx = [];
ReconError_list = [];
if plot_ReconError
  if strcmp(run_type, "Heli_DPTX3") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Heli_DPTX3 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ReconError_list = ...
        {["a3_"], ["ReconV1"]; ...
         ["a9_"], ["ReconInfraV2"]; ...
         ["a16_"], ["ReconInfraV4"]};
    num_ReconError_list = size(ReconError_list,1);
    ReconError_skip = repmat(1, num_ReconError_list, 1);
    ReconError_skip(1) = 1;
    ReconError_skip(2) = 1;
    ReconError_skip(3) = 1;
    ReconError_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a0_"], ["Image"]; ...
         ["a0_"], ["Image"]};
    ReconError_norm_strength = ones(num_ReconError_list,1);
    ReconError_norm_strength = ...
	[1/sqrt(18*18); 1/sqrt(18*18); 1/sqrt(18*18)];
    ReconError_RMS_fig_ndx = [1 1 1];  %% causes recon error to be overlaid on specified  nonSparse (Error) figure
  elseif strcmp(run_type, "PASCAL_DPT") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PASCAL_DPT list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ReconError_list = ...
        {["a3_"],  ["ImageReconS1"]; ...
         ["a8_"],  ["ImageReconS2"]; ...
         ["a13_"],  ["ImageReconS3"]; ...
         ["a10_"],  ["ImageReconS1ReconS2"]; ...
         ["a18_"],  ["ImageReconS2ReconS3"]}; 
    num_ReconError_list = size(ReconError_list,1);
    ReconError_skip = repmat(1, num_ReconError_list, 1);
    ReconError_norm_list = ...
        {["a0_"], ["Image"]; ...
         ["a0_"], ["Image"]; ...
         ["a0_"], ["Image"]; ...
         ["a0_"], ["Image"]; ...
         ["a0_"], ["Image"]}; 
    ReconError_norm_strength = ...
	[1/sqrt(18*18*3); 1/sqrt(18*18*3); 1/sqrt(18*18*3); 1/sqrt(18*18*3); 1/sqrt(18*18*3)];
    ReconError_RMS_fig_ndx = [1 1 1 1 1];%% 1];
  endif %% run_type
  
  if ~isempty(ReconError_list)
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
  endif
endif %% plot_ReconError







plot_ErrorVsSparse = false && analyze_Sparse_flag && analyze_nonSparse_flag;
if plot_ErrorVsSparse
  ErrorVsSparse_list = [nonSparse_list; ReconError_list];
  num_ErrorVsSparse_list = size(ErrorVsSparse_list,1);
  Sparse_axis_index = ones(num_ErrorVsSparse_list,1);
  if strcmp(run_type, "Heli_C1") 
    Sparse_axis_index(2) = 2;
    Sparse_axis_index(3) = 2;
    Sparse_axis_index(4) = 1;
    Sparse_axis_index(5) = 2;
    Sparse_axis_index(6) = 2;
    Sparse_axis_index(7) = 3;
  elseif strcmp(run_type, "Heli_DPTX3")
    Sparse_axis_index = zeros(num_ErrorVsSparse_list,3);
    Sparse_axis_index(1,1) = 1;
    Sparse_axis_index(2,1) = 2;
    Sparse_axis_index(3,1) = 2;
    Sparse_axis_index(4,1) = 3;
    Sparse_axis_index(5,1) = 3;
    Sparse_axis_index(6,1) = 1;
    Sparse_axis_index(7,1) = 2;
    Sparse_axis_index(8,1) = 3;
    Sparse_axis_index(:,2) = 2;
    Sparse_axis_index(:,3) = 3;
  elseif strcmp(run_type, "Heli_DTX3") || strcmp(run_type, "Heli_DX3")   
    Sparse_axis_index(2) = 2;
    Sparse_axis_index(3) = 3;
    Sparse_axis_index(4) = 1;
    Sparse_axis_index(5) = 2;
    Sparse_axis_index(6) = 3;
  elseif strcmp(run_type, "Heli_V1V2V4") 
    Sparse_axis_index(2) = 2;
    Sparse_axis_index(3) = 3;
    Sparse_axis_index(4) = 1;
    Sparse_axis_index(5) = 2;
    Sparse_axis_index(6) = 3;
  elseif strcmp(run_type, "CIFAR_C1")
    Sparse_axis_index(2) = 2;
    Sparse_axis_index(3) = 2;
    Sparse_axis_index(4) = 1;
    Sparse_axis_index(5) = 2;
  endif



  ErrorVsSparse_dir = [output_dir, filesep, "ErrorVsSparse"]
  [status, msg, msgid] = mkdir(ErrorVsSparse_dir);
  if status ~= 1
    warning(["mkdir(", ErrorVsSparse_dir, ")", " msg = ", msg]);
  endif 
  for i_ErrorVsSparse = 1 : num_ErrorVsSparse_list
    nonSparse_times = nonSparse_times_array{i_ErrorVsSparse};
    nonSparse_RMS = nonSparse_RMS_array{i_ErrorVsSparse};
    nonSparse_norm_RMS = nonSparse_norm_RMS_array{i_ErrorVsSparse};
    num_nonSparse_frames = length(nonSparse_times);
    if num_nonSparse_frames < 2
      continue;
    endif



    %% get percent active bins
    i_Sparse = Sparse_axis_index(i_ErrorVsSparse,1);
    Sparse_times = Sparse_times_array{i_Sparse};
    Sparse_percent_active = Sparse_percent_active_array{i_Sparse};
    num_Sparse_neurons = ...
	Sparse_hdr{i_Sparse}.nxGlobal * Sparse_hdr{i_Sparse}.nyGlobal * Sparse_hdr{i_Sparse}.nf;
    Sparse_sum_norm = 1;
    Sparse_sum_offset = 0;
    for i_Sparse_sum = 2 : size(Sparse_axis_index,2)
      i_Sparse = Sparse_axis_index(i_ErrorVsSparse,i_Sparse_sum);
      if i_Sparse <= 0
	continue;
      endif
      Sparse_times_sum = Sparse_times_array{i_Sparse};
      while(Sparse_times_sum(end-Sparse_sum_offset) > Sparse_times(end))
	Sparse_sum_offset = Sparse_sum_offset + 1;
      endwhile
      Sparse_times_sum = Sparse_times_sum(1:end-Sparse_sum_offset);
      %%Sparse_times_ndx = find(Sparse_times_sum ~= Sparse_times);
      %%Sparse_times_sum =Sparse_times_sum(Sparse_times_ndx);
      %%if any(Sparse_times_sum ~= Sparse_times)
      %%	keyboard;
      %%endif
      num_Sparse_neurons_sum = ...
	  Sparse_hdr{i_Sparse}.nxGlobal * Sparse_hdr{i_Sparse}.nyGlobal * Sparse_hdr{i_Sparse}.nf;
      if length(Sparse_percent_active) > length(Sparse_percent_active_array{i_Sparse}(1:end-Sparse_sum_offset))
	Sparse_percent_active = Sparse_percent_active(1+Sparse_sum_offset:end) + (num_Sparse_neurons_sum / num_Sparse_neurons) * Sparse_percent_active_array{i_Sparse}(1:end-Sparse_sum_offset);
      else
	Sparse_percent_active = Sparse_percent_active + (num_Sparse_neurons_sum / num_Sparse_neurons) * Sparse_percent_active_array{i_Sparse}(1:end-Sparse_sum_offset);
      endif      
      Sparse_sum_norm = Sparse_sum_norm + (num_Sparse_neurons_sum / num_Sparse_neurons);
    endfor
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
    endif      
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
    endif      
    if max(Sparse_times(:)) >= last_nonSparse_time
      [last_Sparse_ndx, ~, last_Sparse_diff] = ...
	  find(Sparse_times - last_nonSparse_time < 0, 1, "last");
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
    num_Sparse_bins = 5; %%10;
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
    ErrorVsSparse_name = ...
	["ErrorVsSparse_", ErrorVsSparse_list{i_ErrorVsSparse,1}, ErrorVsSparse_list{i_ErrorVsSparse,2}, "_", ...
	 num2str(nonSparse_times(num_nonSparse_frames), "%08d")];
    if plot_flag
      normalized_nonSparse_RMS = nonSparse_RMS(1:last_nonSparse_ndx) ./ ...
       	       (nonSparse_norm_RMS(1:last_nonSparse_ndx) + (nonSparse_norm_RMS(1:last_nonSparse_ndx) == 0));
      %%normalized_nonSparse_RMS = nonSparse_RMS(1:last_nonSparse_ndx);
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
	  ErrorVsSparse_name);
      saveas(ErrorVsSparse_fig, ...
	     [ErrorVsSparse_dir, filesep, ...
	      ErrorVsSparse_name, ".", "png"], "png");
    endif %% plot_flag
    save("-mat", ...
	 [ErrorVsSparse_dir, filesep, ErrorVsSparse_name, ".mat"], ...
	 "nonSparse_times", "Sparse_vals", "nonSparse_RMS", "nonSparse_norm_RMS", ...
	 "Sparse_bins", "mean_nonSparse_RMS", "std_nonSparse_RMS");	 
    %keyboard;
  endfor  %% i_ErrorVsSparse
  drawnow;
endif %% plot_ErrorVsSparse






%%keyboard;
analyze_weights = true;
plot_weights = plot_flag;
if analyze_weights
  weights_list = {};
  labelWeights_list = {};
  if strcmp(run_type, "ICA")  || strcmp(run_type, "experts")  || strcmp(run_type, "MaxPool")  || strcmp(run_type, "DCA") || strcmp(run_type, "CIFAR")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ICA; experts list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["S1ToImageReconS1Error"]};
      checkpoints_list = {output_dir};
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      if strcmp(run_type, "ICA")
	weights_glob_str = "S?ToImage*econ*Error*_W.pvp";
      elseif strcmp(run_type, "experts")
	weights_glob_str = "S*ToImageRecon*Error*_W.pvp";
      elseif strcmp(run_type, "MaxPool") 
	weights_glob_str = "S1ToImageReconS1Error*_W.pvp";
      elseif strcmp(run_type, "DCA") || strcmp(run_type, "CIFAR")
	weights_glob_str = "S1ToImageDecon*Error*_W.pvp";
      endif
      weights_glob_list = glob([checkpoints_list{1}, filesep, weights_glob_str]);
      num_weights_list = length(weights_glob_list);    
      weights_list = cell(num_weights_list,1);
      sparse_weights_ndx = zeros(num_weights_list,1);
      for i_weights_list = 1 : num_weights_list
	[weights_list_dir, weights_list_name, weights_list_ext, weights_list_ver] = fileparts(weights_glob_list{i_weights_list});
	weights_underscore_ndx = strfind(weights_list_name, "_W");
	weights_list{i_weights_list,1} = weights_list_name(1:weights_underscore_ndx(1)-1);
	weights_list{i_weights_list,2} = weights_list_name(weights_underscore_ndx(1):length(weights_list_name));
	if analyze_Sparse_flag
	  weights_layer_ndx = strfind(weights_list{i_weights_list,1},"To")-1;
	  weights_layer_str = weights_list{i_weights_list,1}(1:weights_layer_ndx);
	  for i_Sparse_list = 1:num_Sparse_list
	    layer_id_ndx = strfind(weights_layer_str, Sparse_list{i_Sparse_list, 2});
	    if ~isempty(layer_id_ndx)
	      sparse_weights_ndx(i_weights_list) = i_Sparse_list;
	      break;
	    endif
	  endfor
	endif
      endfor
      labelWeights_list = {}; %%...
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
   elseif strcmp(run_type, "SLP") || strcmp(run_type, "PCA") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SLP list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_weights_ndx = [1];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["S1ToImageReconS1Error"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["S1ToImageReconS1Error"], ["_W"]};
      labelWeights_list = {}; %%...
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
   elseif strcmp(run_type, "dSCANN") || strcmp(run_type, "DCNN") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% dSCANN or DCNN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_weights_ndx = [1];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["S1ToImageDeconError"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["S1ToImageDeconError"], ["_W"]};
      labelWeights_list = {}; %%...
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
   elseif strcmp(run_type, "DCNNX3") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNNX3 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_weights_ndx = [1 1 1];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["S1ToImageDeconError"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["S1ToImageDeconError"], ["_W"];
	   ["S1DeconS2ToImageDeconError"], ["_W"];
	   ["S1DeconS3ToImageDeconError"], ["_W"]};
      labelWeights_list = {}; %%...
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
   elseif strcmp(run_type, "DBN")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DBN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_weights_ndx = [1];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["S1ToImageReconError"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["S1ToImageReconError"], ["_W"]};
      labelWeights_list = {}; %%...
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
   elseif strcmp(run_type, "MRI") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MRI list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_weights_ndx = [1 1 1 1];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["S1ToImage0ReconS1Error"];
	   ["w1_"], ["S1ToImage1ReconS1Error"];
	   ["w1_"], ["S1ToImage2ReconS1Error"];
	   ["w1_"], ["S1ToImage3ReconS1Error"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["S1ToImage0ReconS1Error"], ["_W"];
	   ["S1ToImage1ReconS1Error"], ["_W"];
	   ["S1ToImage2ReconS1Error"], ["_W"];
	   ["S1ToImage3ReconS1Error"], ["_W"]};
      labelWeights_list = {}; %%...
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
   elseif strcmp(run_type, "M1") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% M1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_weights_ndx = [1];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["S1ToImageReconError"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["S1ToImageReconError"], ["_W"]};
      labelWeights_list = {}; %%...
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
   elseif strcmp(run_type, "MRI") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MRI list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_weights_ndx = [1 1 1 1];
    if ~checkpoint_weights_movie
      weights_list = ...
          {["w1_"], ["S1ToImage0ReconS1Error"];
	   ["w1_"], ["S1ToImage1ReconS1Error"];
	   ["w1_"], ["S1ToImage2ReconS1Error"];
	   ["w1_"], ["S1ToImage3ReconS1Error"]};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["S1ToImage0ReconS1Error"], ["_W"];
	   ["S1ToImage1ReconS1Error"], ["_W"];
	   ["S1ToImage2ReconS1Error"], ["_W"];
	   ["S1ToImage3ReconS1Error"], ["_W"]};
      labelWeights_list = {}; %%...
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
   elseif strcmp(run_type, "M1") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% M1 list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sparse_weights_ndx = [2];
    if ~checkpoint_weights_movie
      weights_list = ...
          {};
      checkpoints_list = {output_dir};
    else
      weights_list = ...
          {["M1ToGroundTruthReconM1Error"], ["_W"]};
      labelWeights_list = {}; %%...
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
    endif %% checkpoint_weights_movie
    num_checkpoints = size(checkpoints_list,1);
  endif %% run_type

  num_weights_list = size(weights_list,1);
  weight_patch_array_list = cell(num_weights_list, 1);
  if ~exist("weights_pad_size") || length(weights_pad_size(:)) < num_weights_list
    weights_pad_size = zeros(1, num_weights_list);
  endif
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
      if i_checkpoint ~= max_checkpoint
	%%continue;  %% comment this line to generate movie of all checkpoints
      endif
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
      tmp_ndx = sparse_weights_ndx(i_weights);
      if tmp_ndx > 0 && analyze_Sparse_flag
	tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
	tmp_rank = [];
      endif
      if analyze_Sparse_flag && ~isempty(tmp_rank)
	pre_hist_rank = tmp_rank;
      else
	pre_hist_rank = (1:weights_hdr{i_weights}.nf);
      endif

      if length(labelWeights_list) >= i_weights && ...
	    ~isempty(labelWeights_list{i_weights}) && ...
	    plot_weights && ...
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
      tot_patches = num_patches;
      num_patches = min(num_patches, max_patches);
      num_patches_rows = floor(sqrt(num_patches));
      num_patches_cols = ceil(num_patches / num_patches_rows);
      num_weights_colors = 1;
      num_weights_figs = ceil(tot_patches/num_patches);
      for i_weights_fig = 1 : num_weights_figs
      if num_weights_dims == 4
	num_weights_colors = size(weight_vals,3);
      endif
      if plot_weights && i_checkpoint == max_checkpoint
	weights_fig = figure;
	set(weights_fig, "name", [weights_name, "_", num2str(i_weights_fig)]);
      endif
      weight_patch_array = [];
      for k_patch = (i_weights_fig-1)*num_patches + 1  : i_weights_fig*num_patches
	i_patch = pre_hist_rank(k_patch);
	j_patch = k_patch - (i_weights_fig-1)*num_patches;
	if plot_weights && i_checkpoint == max_checkpoint
	  subplot(num_patches_rows, num_patches_cols, j_patch); 
	endif
	if num_weights_colors == 1
	  patch_tmp = squeeze(weight_vals(:,:,i_patch));
	else
	  patch_tmp = squeeze(weight_vals(:,:,:,i_patch));
	endif


	patch_tmp2 = permute(patch_tmp, [2,1,3]); %%patch_tmp; %% imresize(patch_tmp, 12);
	if num_weights_colors > 3

	  weight_colormap = prism(num_weights_colors+1);
	  patch_tmp2b = zeros([size(patch_tmp2,1),size(patch_tmp2,2),3]);
	  for weight_color_ndx = 1 : num_weights_colors
	    weight_color = weight_colormap(weight_color_ndx,:);
	    patch_tmp2b(:,:,1) = squeeze(patch_tmp2b(:,:,1)) + squeeze(patch_tmp2(:,:,weight_color_ndx) .* weight_color(1));
	    patch_tmp2b(:,:,2) = squeeze(patch_tmp2b(:,:,2)) + squeeze(patch_tmp2(:,:,weight_color_ndx) .* weight_color(2));
	    patch_tmp2b(:,:,3) = squeeze(patch_tmp2b(:,:,3)) + squeeze(patch_tmp2(:,:,weight_color_ndx) .* weight_color(3));
	  endfor
	  patch_tmp2 = patch_tmp2b;

	  % patch_tmp2b = zeros(size(patch_tmp2, 1),size(patch_tmp2, 2),3);
	  % for i_color = 1 : num_weights_colors
	  %   patch_tmp2b(:,:,1 + mod(i_color,3)) = patch_tmp2b(:,:,1 + mod(i_color,3)) + patch_tmp2(:,:,1 + mod(i_color,3));
	  % endfor
	  % patch_tmp2 = patch_tmp2b;

	endif
	%%min_patch = min(patch_tmp2(:));
	maxabs_patch = max(abs(patch_tmp2(:)));
	patch_tmp3 = uint8(127.5 + 127.5*(patch_tmp2) / (maxabs_patch + (maxabs_patch==0)));
	%%patch_tmp2 = uint8(permute(patch_tmp2, [2,1,3])); %% uint8(flipdim(permute(patch_tmp2, [2,1,3]),1));

	pad_size = weights_pad_size(i_weights);
	padded_patch_size = size(patch_tmp2);
	padded_patch_size(1) = padded_patch_size(1) + 2*pad_size;
	padded_patch_size(2) = padded_patch_size(2) + 2*pad_size;
	patch_tmp6 = repmat(uint8(128),padded_patch_size);
	if ndims(patch_tmp) == 3
	  patch_tmp6(pad_size+1:padded_patch_size(1)-pad_size,pad_size+1:padded_patch_size(2)-pad_size,:) = patch_tmp3; %%uint8(patch_tmp2);
	else
	  patch_tmp6(pad_size+1:padded_patch_size(1)-pad_size,pad_size+1:padded_patch_size(2)-pad_size) = patch_tmp3; %%uint8(patch_tmp2);
	endif
	

	if plot_weights && i_checkpoint == max_checkpoint
	  imagesc(patch_tmp6); 
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
	endif %% plot_weights
	if isempty(weight_patch_array)
	  weight_patch_array = ...
	      zeros(num_patches_rows*size(patch_tmp6,1), num_patches_cols*size(patch_tmp6,2), size(patch_tmp6,3));
	endif 
	col_ndx = 1 + mod(j_patch-1, num_patches_cols);
	row_ndx = 1 + floor((j_patch-1) / num_patches_cols);
	maxabs_patch = max(abs(patch_tmp6(:)));
	%normalized_patch = 127.5 + 127.5 * (patch_tmp6 / (maxabs_patch + (maxabs_patch==0)));
	weight_patch_array(((row_ndx-1)*size(patch_tmp6,1)+1):row_ndx*size(patch_tmp6,1), ...
			   ((col_ndx-1)*size(patch_tmp6,2)+1):col_ndx*size(patch_tmp6,2),:) = ...
	    uint8(patch_tmp6);
      endfor  %% j_patch
      %%keyboard;
      if plot_weights && i_checkpoint == max_checkpoint
	%%cwd = pwd;
	chdir(weights_dir)
	if ~isempty(labelWeights_vals)
	  saveas(weights_fig, [weights_dir, filesep, [weights_name,"_labeled"], "_", num2str(i_weights_fig), ".png"], "png");
	  %%saveas(weights_fig, [weights_name, "_labeled", ".png"], "png");
	else
	  saveas(weights_fig, [weights_dir, filesep, weights_name, "_", num2str(i_weights_fig), ".png"], "png");
	  %%saveas(weights_fig, [weights_name, ".png"], "png");
	endif
	%%chdir(cwd);
	if num_weights_list > 10
	  close(weights_fig);
	endif
      endif
      if i_checkpoint == max_checkpoint
	weight_patch_array_list{i_weights,i_weights_fig} = weight_patch_array;
      endif
      if i_checkpoint == max_checkpoint
 	save("-mat", ...
	     [weights_movie_dir, filesep, weights_name, ".mat"], ...
	     "weight_patch_array");
      endif
      
      maxabs_weight_patch_array = max(abs(weight_patch_array(:)));
      %%weight_patch_array = uint8(127.5 + 127.5 * (weight_patch_array / (maxabs_weight_patch_array + (maxabs_weight_patch_array==0))));
      imwrite(uint8(weight_patch_array), [weights_movie_dir, filesep, weights_name, "_", num2str(i_weights_fig), ".png"], "png");
      %% make histogram of all weights
      if plot_weights && i_checkpoint == max_checkpoint
	weights_hist_fig = figure;
	[weights_hist, weights_hist_bins] = hist(weight_vals(:), 100);
	bar(weights_hist_bins, log(weights_hist+1));
	set(weights_hist_fig, "name", ...
	    ["Hist_",  weights_list{i_weights,1}, weights_list{i_weights,2}, "_", num2str(weight_time, "%08d"), "_", num2str(i_weights_fig)]);
	saveas(weights_hist_fig, ...
	       [weights_dir, filesep, "weights_hist_",  weights_list{i_weights,1}, weights_list{i_weights,2}, "_", num2str(weight_time, "%08d"), "_", num2str(i_weights_fig)], "png");
	if num_weights_list > 10
	  close(weights_hist_fig);
	endif
      endif

      if ~isempty(labelWeights_vals) && ...
	    ~isempty(labelWeights_time) && ...
	    plot_weights && ...
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
	  endif
	  labeledWeights_subplot_str = ...
	      [num2str(label, "%d")];
	  title(labeledWeights_subplot_str);
	  axis off
	endfor %% label
	saveas(labeledWeights_fig,  [weights_dir, filesep, labeledWeights_str, ".png"], "_", num2str(i_weights_fig), "png");
      endif  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)
      endfor  %% i_weights_fig

    endfor %% i_checkpoint
  endfor %% i_weights
endif  %% plot_weights




plot_labelRecon = true;
labels_list = {};
labelRecon_list = {};
if plot_labelRecon
  if strcmp(run_type, "CIFAR_deep") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MNIST/CIFAR list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    labels_list = ...
	{["a8_"], ["Labels"]};
    labelRecon_list = ...
	{["a10_"], ["LabelRecon"]};
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
analyze_weights0_2 = true 
plot_weights0_2_flag = plot_flag;
plot_labelWeights_flag = true;
if analyze_weights0_2
  weights1_2_list = {};
  if strcmp(run_type, "default") || strcmp(run_type, "experts") || strcmp(run_type, "MaxPool") || strcmp(run_type, "DCA") || strcmp(run_type, "CIFAR")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MaxPool
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      break;
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_glob_list = ...
      glob([checkpoints_list{1}, filesep, "S2ToS1*econ*Error*_W.pvp"]);
      num_weights1_2_list = length(weights1_2_glob_list);    
      weights1_2_list = cell(num_weights1_2_list,2);
      sparse_weights0_2_ndx = zeros(num_weights1_2_list,1);
      post1_2_list = cell(num_weights1_2_list,2);
      image_list = cell(num_weights1_2_list,2);
      for i_weights1_2_list = 1 : num_weights1_2_list
	[weights1_2_list_dir, weights1_2_list_name, weights1_2_list_ext, weights1_2_list_ver] = fileparts(weights1_2_glob_list{i_weights1_2_list});
	weights1_2_underscore_ndx = strfind(weights1_2_list_name, "_W");
	weights1_2_list{i_weights1_2_list,1} = weights1_2_list_name(1:weights1_2_underscore_ndx(1)-1);
	weights1_2_list{i_weights1_2_list,2} = weights1_2_list_name(weights1_2_underscore_ndx(1):length(weights1_2_list_name));
	if analyze_Sparse_flag
	  weights1_2_layer_ndx = strfind(weights1_2_list{i_weights1_2_list,1},"To")-1;
	  weights1_2_layer_str = weights1_2_list{i_weights1_2_list,1}(1:weights1_2_layer_ndx);
	  for i_Sparse_list = 1:num_Sparse_list
	    layer_id_ndx = strfind(weights1_2_layer_str, Sparse_list{i_Sparse_list, 2});
	    if ~isempty(layer_id_ndx)
	      sparse_weights0_2_ndx(i_weights1_2_list) = i_Sparse_list;
	      break;
	    endif
	  endfor
	endif
	post1_2_list{i_weights1_2_list,1} = ["S1"];
	post1_2_list{i_weights1_2_list,2} = ["_A"];
	image_list{i_weights1_2_list,1} = ["Image"];
	image_list{i_weights1_2_list,2} = ["_A"];
      endfor
      %% list of weights from layer1 to image
      weights0_1_list = weights_list;
    endif %% checkpoint_weights_movie
    labelWeights_list = {}; %%...
    num_checkpoints = size(checkpoints_list,1);
    weights1_2_pad_size = [0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "SLP")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PASCAL_SLP list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      break;
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["S2ToS1ReconS2Error"], ["_W"]};
      post1_2_list = ...
          {["S1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["S1ToImageReconS1Error"], ["_W"]};
      image_list = ...
          {["Image"], ["_A"]};
%%      labelWeights_list = ...
%%	  {["V2ToLabelError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weights0_2_ndx = [2];
    num_checkpoints = size(checkpoints_list,1);
    weights1_2_pad_size = [0];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "dSCANN")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% dSCANN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      break;
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["S2ToS1DeconError"], ["_W"];
	   ["S2ToS1ReconS2Error"], ["_W"]};
      post1_2_list = ...
          {["S1"], ["_A"];
	   ["S1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["S1ToImageDeconError"], ["_W"];
	   ["S1ToImageDeconError"], ["_W"]};
      image_list = ...
          {["Image"], ["_A"];
	   ["Image"], ["_A"]};
%%      labelWeights_list = ...
%%	  {["V2ToLabelError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weights0_2_ndx = [2 2];
    num_checkpoints = size(checkpoints_list,1);
    weights1_2_pad_size = [0];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DCNN") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      break;
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["S2ToS1DeconError"], ["_W"];
	   ["S2DeconS3ToS1DeconErrorS3"], ["_W"]};
      post1_2_list = ...
          {["S1"], ["_A"];
	   ["S1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["S1ToImageDeconError"], ["_W"];
	   ["S1DeconS3ToImageDeconError"], ["_W"]};
      image_list = ...
          {["Image"], ["_A"];
	   ["Image"], ["_A"]};
%%      labelWeights_list = ...
%%	  {["V2ToLabelError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weights0_2_ndx = [2 2];
    num_checkpoints = size(checkpoints_list,1);
    weights1_2_pad_size = [0 0];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DCNNX3")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      break;
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["S2ToS1DeconErrorS2"], ["_W"];
	   ["S2DeconS3ToS1DeconErrorS3"], ["_W"]};
      post1_2_list = ...
          {["S1"], ["_A"];
	   ["S1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["S1DeconS2ToImageDeconError"], ["_W"];
	   ["S1DeconS3ToImageDeconError"], ["_W"]};
      image_list = ...
          {["Image"], ["_A"];
	   ["Image"], ["_A"]};
%%      labelWeights_list = ...
%%	  {["V2ToLabelError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weights0_2_ndx = [2 2];
    num_checkpoints = size(checkpoints_list,1);
    weights1_2_pad_size = [0 0];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DBN")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DBN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      break;
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["S2ToS1ReconError"], ["_W"]};
      post1_2_list = ...
          {["S1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["S1ToImageReconError"], ["_W"]};
      image_list = ...
          {["Image"], ["_A"]};
%%      labelWeights_list = ...
%%	  {["V2ToLabelError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weights0_2_ndx = [2];
    num_checkpoints = size(checkpoints_list,1);
    weights1_2_pad_size = [0];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "PCA")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PCA list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layer2 to layer1
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      break;
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weights1_2_list = ...
          {["S201ToS1ReconS2Error"], ["_W"];
	   ["S202ToS1ReconS2Error"], ["_W"];
	   ["S203ToS1ReconS2Error"], ["_W"];
	   ["S204ToS1ReconS2Error"], ["_W"];
	   ["S205ToS1ReconS2Error"], ["_W"];
	   ["S206ToS1ReconS2Error"], ["_W"];
	   ["S207ToS1ReconS2Error"], ["_W"];
	   ["S208ToS1ReconS2Error"], ["_W"]};
      post1_2_list = ...
          {["S1"], ["_A"];
	   ["S1"], ["_A"];
	   ["S1"], ["_A"];
	   ["S1"], ["_A"];
	   ["S1"], ["_A"];
	   ["S1"], ["_A"];
	   ["S1"], ["_A"];
	   ["S1"], ["_A"]};
      %% list of weights from layer1 to image
      weights0_1_list = ...
          {["S1ToImageReconS1Error"], ["_W"];
	   ["S1ToImageReconS1Error"], ["_W"];
	   ["S1ToImageReconS1Error"], ["_W"];
	   ["S1ToImageReconS1Error"], ["_W"];
	   ["S1ToImageReconS1Error"], ["_W"];
	   ["S1ToImageReconS1Error"], ["_W"];
	   ["S1ToImageReconS1Error"], ["_W"];
	   ["S1ToImageReconS1Error"], ["_W"]};
      image_list = ...
          {["Image"], ["_A"];
	   ["Image"], ["_A"];
	   ["Image"], ["_A"];
	   ["Image"], ["_A"];
	   ["Image"], ["_A"];
	   ["Image"], ["_A"];
	   ["Image"], ["_A"];
	   ["Image"], ["_A"]};
%%      labelWeights_list = ...
%%	  {["V2ToLabelError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weights0_2_ndx = [2 3 4 5 6 7 8 9];
    num_checkpoints = size(checkpoints_list,1);
    weights1_2_pad_size = [0 0 0 0 0 0 0 0 0];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  num_weights1_2_list = size(weights1_2_list,1);
  if num_weights1_2_list > 0

  if ~exist("weights1_2_pad_size") || length(weights1_2_pad_size(:)) < num_weights1_2_list
    weights1_2_pad_size = zeros(1, num_weights1_2_list);
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
      if i_checkpoint ~= max_checkpoint 
	continue;
      endif
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
      tmp_ndx = sparse_weights0_2_ndx(i_weights1_2);
      if analyze_Sparse_flag
	tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
	tmp_rank = [];
      endif
      if analyze_Sparse_flag && ~isempty(tmp_rank)
	pre_hist_rank = tmp_rank;
      else
	pre_hist_rank = (1:weights1_2_hdr{i_weights1_2}.nf);
      endif

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
      i_patch = 1;
      num_weights1_2_dims = ndims(weights1_2_vals);
      num_patches0_2 = size(weights1_2_vals, num_weights1_2_dims);
      num_patches0_2_per_fig = min(num_patches0_2, max_patches);
      num_figs0_2 = ceil(num_patches0_2/num_patches0_2_per_fig);
      %% algorithms assumes weights1_2 are one to many
      num_patches0_2_rows = floor(sqrt(num_patches0_2_per_fig));
      num_patches0_2_cols = ceil(num_patches0_2_per_fig / num_patches0_2_rows);
      %% for one to many connections: dimensions of weights1_2 are:
      %% weights1_2(nxp, nyp, nf_post, nf_pre)
      weights1_2_fig = zeros(num_figs0_2,1);
      if plot_weights0_2_flag && i_checkpoint == max_checkpoint
	for i_fig0_2 = 1 : num_figs0_2
	  weights1_2_fig(i_fig0_2) = figure;
	  set(weights1_2_fig(i_fig0_2), "name", [weights1_2_name, "_", num2str(i_fig0_2)]);
	endfor
      endif
      max_shrinkage = 8; %% 
      weight_patch0_2_array = cell(num_figs0_2,1);
      for kf_pre1_2_rank = 1  : num_patches0_2
	i_fig0_2 = ceil(kf_pre1_2_rank / num_patches0_2_per_fig);
	kf_pre1_2_rank_per_fig = kf_pre1_2_rank - (i_fig0_2 - 1) * num_patches0_2_per_fig;
	kf_pre1_2 = pre_hist_rank(kf_pre1_2_rank);
	if plot_weights0_2_flag && i_checkpoint == max_checkpoint
	  subplot(num_patches0_2_rows, num_patches0_2_cols, kf_pre1_2_rank_per_fig); 
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
	%%patch_tmp5 = ...
	%%    uint8((flipdim(patch_tmp3,1) - min_patch) * 255 / ...
	%%	  (max_patch - min_patch + ((max_patch - min_patch)==0)));
	patch_tmp5 = ...
	    (127.5 + 127.5*(flipdim(patch_tmp3,1) ./ (max(abs(patch_tmp3(:))) + (max(abs(patch_tmp3(:)))==0))));
		  
	pad_size = weights1_2_pad_size(i_weights1_2);
	padded_patch_size = size(patch_tmp5);
	padded_patch_size(1) = padded_patch_size(1) + 2*pad_size;
	padded_patch_size(2) = padded_patch_size(2) + 2*pad_size;
	patch_tmp6 = repmat(128,padded_patch_size);
	if ndims(patch_tmp5) == 3
	  patch_tmp6(pad_size+1:padded_patch_size(1)-pad_size,pad_size+1:padded_patch_size(2)-pad_size,:) = (patch_tmp5);
	else
	  patch_tmp6(pad_size+1:padded_patch_size(1)-pad_size,pad_size+1:padded_patch_size(2)-pad_size) = (patch_tmp5);
	endif
	
	if plot_weights0_2_flag && i_checkpoint == max_checkpoint
	  figure(weights1_2_fig(i_fig0_2));
	  subplot(num_patches0_2_rows, num_patches0_2_cols, kf_pre1_2_rank_per_fig); 
	  imagesc(patch_tmp6); 
	  if weights0_1_nfp == 1
	    colormap(gray);
	  endif
	  box off
	  axis off
	  axis image
	  %drawnow
	endif
	if plot_labelWeights_flag && i_checkpoint == max_checkpoint
	  if ~isempty(labelWeights_vals) %% && ~isempty(labelWeights_time) 
	    [~, max_label] = max(squeeze(labelWeights_vals(:,kf_pre1_2)));
	    text(weights0_2_nyp_shrunken/2, -weights0_2_nxp_shrunken/6, num2str(max_label-1), "color", [1 0 0]);
	  endif %% ~empty(labelWeights_vals)
	  %%drawnow;
	endif %% plot_weights0_2_flag && i_checkpoint == max_checkpoint

	if isempty(weight_patch0_2_array{i_fig0_2}) || kf_pre1_2_rank_per_fig == 1
	  weight_patch0_2_array{i_fig0_2} = ...
	      zeros(num_patches0_2_rows*(weights0_2_nyp_shrunken+2*pad_size), ...
		    num_patches0_2_cols*(weights0_2_nxp_shrunken+2*pad_size), weights0_1_nfp);
	endif
	col_ndx = 1 + mod(kf_pre1_2_rank_per_fig -1, num_patches0_2_cols);
	row_ndx = 1 + floor((kf_pre1_2_rank_per_fig -1) / num_patches0_2_cols);
	maxabs_patch0_2 = max(abs(patch_tmp6(:)));
	%%normalized_patch0_2 = 127.5 + 127.5 * (patch_tmp6 / (maxabs_patch0_2 + (maxabs_patch0_2==0)));
	weight_patch0_2_array{i_fig0_2}((1+(row_ndx-1)*(weights0_2_nyp_shrunken+2*pad_size)):...
					(row_ndx*(weights0_2_nyp_shrunken+2*pad_size)), ...
					(1+(col_ndx-1)*(weights0_2_nxp_shrunken+2*pad_size)):...
					(col_ndx*(weights0_2_nxp_shrunken+2*pad_size)),:) = ...
	    patch_tmp6; %%normalized_patch0_2;

	%% Plot the average movie weights for a label %%
	if plot_labelWeights_flag && i_checkpoint == max_checkpoint
	  if ~isempty(labelWeights_vals) 
	    if ~isempty(labeledWeights0_2{max_label})
	      labeledWeights0_2{max_label} = labeledWeights0_2{max_label} + double(patch_tmp6);
	    else
	      labeledWeights0_2{max_label} = double(patch_tmp6);
	    endif
	  endif %%  ~isempty(labelWeights_vals) 
	endif %% plot_weights0_2_flag && i_checkpoint == max_checkpoint

      endfor %% kf_pre1_2_ank

      if plot_weights0_2_flag && i_checkpoint == max_checkpoint
	for i_fig0_2 = 1 : num_figs0_2
	  figure(weights1_2_fig(i_fig0_2))
	  saveas(weights1_2_fig(i_fig0_2), [weights1_2_dir, filesep, weights1_2_name, "_", num2str(i_fig0_2), ".png"], "png");
	endfor
      endif
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
	endfor %% label = 1:size(labelWeights_vals,1)
	saveas(labeledWeights_fig,  [weights_dir, filesep, labeledWeights_str, ".png"], "png");
      endif %%  ~isempty(labelWeights_vals) 

      %%imwrite(uint8(weight_patch0_2_array), [weights1_2_movie_dir, filesep, weights1_2_name, ".png"], "png");
      for i_fig0_2 = 1 : num_figs0_2
	weight_patch0_2_array_list{i_weights1_2, i_fig0_2} = weight_patch0_2_array{i_fig0_2};
	num_combine0_2 = 1;
	for i_weights = 1 : num_weights_list
	  if sparse_weights0_2_ndx(i_weights1_2) == sparse_weights_ndx(i_weights)
	    size_weight_patch0_2_array = size(weight_patch0_2_array_list{i_weights1_2, i_fig0_2});
	    resized_weight_patch0_2_array = imresize(weight_patch_array_list{i_weights, i_fig0_2}, size_weight_patch0_2_array(1:2)); 
	    maxabs_resized_weight_patch0_2_array = max(abs(resized_weight_patch0_2_array(:)));
	    resized_weight_patch0_2_array_uint8 = ...
		127.5 + 127.5 * (resized_weight_patch0_2_array / (maxabs_resized_weight_patch0_2_array + (maxabs_resized_weight_patch0_2_array == 0)));
	    weight_patch0_2_array_list{i_weights1_2, i_fig0_2} = weight_patch0_2_array_list{i_weights1_2, i_fig0_2}  + ...
		resized_weight_patch0_2_array_uint8;
	    num_combine0_2 = num_combine0_2 + 1;
	  endif
	endfor
	weight_patch0_2_array_list{i_weights1_2, i_fig0_2} = weight_patch0_2_array_list{i_weights1_2, i_fig0_2} / num_combine0_2;
	maxabs_weight_patch0_2_array = max(abs(weight_patch0_2_array_list{i_weights1_2, i_fig0_2}(:)));
	weight_patch0_2_array_uint8 = ...
	    (127.5 + 127.5*(weight_patch0_2_array_list{i_weights1_2, i_fig0_2} / (maxabs_weight_patch0_2_array + (maxabs_weight_patch0_2_array == 0))));
	imwrite(uint8(weight_patch0_2_array_uint8), ...
		[weights1_2_movie_dir, filesep, weights1_2_name, "_", num2str(i_fig0_2), ".png"], "png");
	
      endfor
      if i_checkpoint == max_checkpoint
 	save("-mat", ...
	     [weights1_2_movie_dir, filesep, weights1_2_name, ".mat"], ...
	     "weight_patch0_2_array");
      endif

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
      endif

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

      endif  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

    endfor %% i_checkpoint

  endfor %% i_weights
  
  endif
endif  %% plot_weights





deRecon_flag = false && exist(labelWeights_vals) && ~isempty(labelWeights_vals);
if deRecon_flag
  num_deRecon = 3;
  deRecon_sparse_weights0_2_ndx = 2;
  deRecon_struct = Sparse_struct_array{deRecon_sparse_weights0_2_ndx};
  num_deRecon_frames = size(deRecon_struct,1);
  Recon_dir = [output_dir, filesep, "Recon"];
  for i_deRecon_frame = 1 : num_deRecon_frames
    deRecon_time = deRecon_struct{i_deRecon_frame}.time
    deRecon_indices = deRecon_struct{i_deRecon_frame}.values(:,1);
    deRecon_vals = deRecon_struct{i_deRecon_frame}.values(:,2);
    [deRecon_vals_sorted, deRecon_vals_rank] = sort(deRecon_vals, "descend");
    deRecon_indices_sorted = deRecon_indices(deRecon_vals_rank)+1;
    num_deRecon_indices = length(deRecon_indices(:));
    deRecon_hist_rank = Sparse_hist_rank_array{deRecon_sparse_weights0_2_ndx}(:);
    for i_deRecon_index = 1 : min(num_deRecon, num_deRecon_indices)
      deRecon_rank = find(deRecon_hist_rank == deRecon_indices_sorted(i_deRecon_index))
      if deRecon_rank > num_patches0_2
	continue;
      endif
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
    endfor %% i_deRecon_index
%%    disp(mat2str(labelWeights_vals(:,deRecon_rank(1: min(num_deRecon, num_deRecon_indices))));
    deRecon_labelWeights = labelWeights_vals(:,deRecon_indices_sorted);
    deRecon_label_activity = repmat(deRecon_vals_sorted(:)',[size(labelWeights_vals,1),1]);
    deRecon_label_prod = deRecon_labelWeights .* deRecon_label_activity;
    deRecon_vals_sorted
    sum_labelWeights = sum(deRecon_label_prod,2)
  endfor %% i_deRecon
endif %% deReconFlag







%%keyboard;
analyze_weightsN_Nplus1 = true;
plot_weightsN_Nplus1_flag = plot_flag;
plot_labelWeights_flag = false; %%true;
if analyze_weightsN_Nplus1
  
  weightsN_Nplus1_list = {};
  layersN_Nplus1_list = {};
  if strcmp(run_type, "default") || strcmp(run_type, "experts") || strcmp(run_type, "MaxPool") || strcmp(run_type, "DCA") || strcmp(run_type, "CIFAR")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% MaxPool
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~checkpoint_weights_movie
      checkpoints_list = {output_dir};
      break;
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weightsN_Nplus1_glob_list = ...
	  glob([checkpoints_list{1}, filesep, "S3ToS2*econ*Error*_W.pvp"]);
      num_weightsN_Nplus1_list = length(weightsN_Nplus1_glob_list);    
      weightsN_Nplus1_list = cell(num_weightsN_Nplus1_list,6);
      layersN_Nplus1_list = cell(num_weightsN_Nplus1_list,6);
      sparse_weightsN_Nplus1_ndx = zeros(num_weightsN_Nplus1_list,1);
      for i_weightsN_Nplus1_list = 1 : num_weightsN_Nplus1_list
	[weightsN_Nplus1_list_dir, weightsN_Nplus1_list_name, weightsN_Nplus1_list_ext, weightsN_Nplus1_list_ver] = fileparts(weightsN_Nplus1_glob_list{i_weightsN_Nplus1_list});
	weightsN_Nplus1_underscore_ndx = strfind(weightsN_Nplus1_list_name, "_W");
	weightsN_Nplus1_list{i_weightsN_Nplus1_list,1} = weightsN_Nplus1_list_name(1:weightsN_Nplus1_underscore_ndx(1)-1);
	weightsN_Nplus1_list{i_weightsN_Nplus1_list,2} = weightsN_Nplus1_list_name(weightsN_Nplus1_underscore_ndx(1):length(weightsN_Nplus1_list_name));
	if analyze_Sparse_flag
	  weightsN_Nplus1_layer_ndx = strfind(weightsN_Nplus1_list{i_weightsN_Nplus1_list,1},"To")-1;
	  weightsN_Nplus1_layer_str = weightsN_Nplus1_list{i_weightsN_Nplus1_list,1}(1:weightsN_Nplus1_layer_ndx);
	  for i_Sparse_list = 1:num_Sparse_list
	    layer_id_ndx = strfind(weightsN_Nplus1_layer_str, Sparse_list{i_Sparse_list, 2});
	    if ~isempty(layer_id_ndx)
	      sparse_weightsN_Nplus1_ndx(i_weightsN_Nplus1_list) = i_Sparse_list;
	      break;
	    endif
	  endfor
	endif
	weightsN_Nplus1_list{i_weightsN_Nplus1_list,3} = weights1_2_list{1,1};
	weightsN_Nplus1_list{i_weightsN_Nplus1_list,4} = weights1_2_list{1,2};
	weightsN_Nplus1_list{i_weightsN_Nplus1_list,5} = weights_list{1,1};
	weightsN_Nplus1_list{i_weightsN_Nplus1_list,6} = weights_list{1,2};
      endfor
      %% list of weights from layer1 to image
    endif %% checkpoint_weights_movie
    layersN_Nplus1_list = ...
        {["S3"], ["_A"], ["S2"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"]};
    labelWeights_list = {[], []}; %%...
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DCNN")
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layerN to layer0 (Image)
    if ~checkpoint_weights_movie
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weightsN_Nplus1_list = ...
          {["S3ToS2DeconError"], ["_W"], ["S2ToS1DeconError"], ["_W"], ["S1ToImageDeconError"], ["_W"]};
      layersN_Nplus1_list = ...
          {["S3"], ["_A"], ["S2"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"]};
      labelWeights_list = ...
	  {[], []}; %%{["S3ToGroundTruthError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weightsN_Nplus1_ndx = [3];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "PCA") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PCA list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layerN to layer0 (Image)
    if ~checkpoint_weights_movie
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weightsN_Nplus1_list = ...
          {["PCAToS201ReconPCAError"], ["_W"], ["S201ToS1ReconS2Error"], ["_W"], ["S1ToImageReconS1Error"], ["_W"];
	   ["PCAToS202ReconPCAError"], ["_W"], ["S202ToS1ReconS2Error"], ["_W"], ["S1ToImageReconS1Error"], ["_W"];
	   ["PCAToS203ReconPCAError"], ["_W"], ["S203ToS1ReconS2Error"], ["_W"], ["S1ToImageReconS1Error"], ["_W"];
	   ["PCAToS204ReconPCAError"], ["_W"], ["S204ToS1ReconS2Error"], ["_W"], ["S1ToImageReconS1Error"], ["_W"];
	   ["PCAToS205ReconPCAError"], ["_W"], ["S205ToS1ReconS2Error"], ["_W"], ["S1ToImageReconS1Error"], ["_W"];
	   ["PCAToS206ReconPCAError"], ["_W"], ["S206ToS1ReconS2Error"], ["_W"], ["S1ToImageReconS1Error"], ["_W"];
	   ["PCAToS207ReconPCAError"], ["_W"], ["S207ToS1ReconS2Error"], ["_W"], ["S1ToImageReconS1Error"], ["_W"];
	   ["PCAToS208ReconPCAError"], ["_W"], ["S208ToS1ReconS2Error"], ["_W"], ["S1ToImageReconS1Error"], ["_W"]};
      layersN_Nplus1_list = ...
          {["PCA"], ["_A"], ["S201"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"];
	   ["PCA"], ["_A"], ["S202"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"];
	   ["PCA"], ["_A"], ["S203"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"];
	   ["PCA"], ["_A"], ["S204"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"];
	   ["PCA"], ["_A"], ["S205"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"];
	   ["PCA"], ["_A"], ["S206"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"];
	   ["PCA"], ["_A"], ["S207"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"];
	   ["PCA"], ["_A"], ["S208"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"]};
      labelWeights_list = ...
	  {[], []}; %%{["S3ToGroundTruthError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weightsN_Nplus1_ndx = [10 10 10 10 10 10 10 10];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "dSCANN") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% dSCANN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layerN to layer0 (Image)
    if ~checkpoint_weights_movie
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weightsN_Nplus1_list = ...
          {["S3ToS2DeconError"], ["_W"], ["S2ToS1DeconError"], ["_W"], ["S1ToImageDeconError"], ["_W"];
	   ["S3ToS2ReconS3Error"], ["_W"], ["S2ToS1ReconS2Error"], ["_W"], ["S1ToImageDeconError"], ["_W"]};
      layersN_Nplus1_list = ...
          {["S3"], ["_A"], ["S2"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"];
	   ["S3"], ["_A"], ["S2"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"]};
      labelWeights_list = ...
	  {[], []}; %%{["S3ToGroundTruthError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weightsN_Nplus1_ndx = [3 3];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DCNNX3") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DCNN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layerN to layer0 (Image)
    if ~checkpoint_weights_movie
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weightsN_Nplus1_list = ...
          {["S3ToS2DeconErrorS3"], ["_W"], ["S2DeconS3ToS1DeconErrorS3"], ["_W"], ["S1DeconS3ToImageDeconError"], ["_W"]};
      layersN_Nplus1_list = ...
          {["S3"], ["_A"], ["S2"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"]};
      labelWeights_list = ...
	  {[], []}; %%{["S3ToGroundTruthError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weightsN_Nplus1_ndx = [3];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif strcmp(run_type, "DBN") 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% DBN list
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% list of weights from layerN to layer0 (Image)
    if ~checkpoint_weights_movie
    else
      checkpoints_list = getCheckpointList(checkpoint_parent, checkpoint_children);
      weightsN_Nplus1_list = ...
          {["S3ToS2ReconS3Error"], ["_W"], ["S2ToS1ReconS2Error"], ["_W"], ["S1ToImageDeconError"], ["_W"]};
      layersN_Nplus1_list = ...
          {["S3"], ["_A"], ["S2"], ["_A"], ["S1"], ["_A"], ["Image"], ["_A"]};
      labelWeights_list = ...
	  {[], []}; %%{["S3ToGroundTruthError"], ["_W"]};
    endif %% checkpoint_weights_movie
    %% list of indices for reading rank order of presynaptic neuron as function of activation frequency
    sparse_weightsN_Nplus1_ndx = [3];
    num_checkpoints = size(checkpoints_list,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  endif %% run_type
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  num_weightsN_Nplus1_list = size(weightsN_Nplus1_list,1);
  num_layersN_Nplus1_list = size(layersN_Nplus1_list,2)/2;
  weightsNminus1_Nplus1_array_list = cell(num_weightsN_Nplus1_list, num_checkpoints);
  if num_weightsN_Nplus1_list == 0
    analyze_weightsN_Nplus1 = false;
    warning(["num_weightsN_Nplus1_list == 0"]);  
  elseif size(weightsN_Nplus1_list,2)/2 ~= num_layersN_Nplus1_list-1;
    analyze_weightsN_Nplus1 = false;
    warning(["num_weightsN_Nplus1_list ~= num_layersN_Nplus1_list-1", ...
	     ", num_weightsN_Nplus1_list = ", num2str(num_weightsN_Nplus1_list), ...
	     ", num_layersN_Nplus1_list = ", num2str(num_layersN_Nplus1_list)]);
  endif
  if num_weightsN_Nplus1_list == 0
    analyze_weightsN_Nplus1 = false;
  endif

  if analyze_weightsN_Nplus1
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
	if ~exist(weightsN_Nplus1_file, "file")
	  warning(["file does not exist: ", weightsN_Nplus1_file]);
	  continue;
	endif
	weightsN_Nplus1_fid = fopen(weightsN_Nplus1_file);
	weightsN_Nplus1_hdr{i_weightN_Nplus1} = readpvpheader(weightsN_Nplus1_fid);    
	fclose(weightsN_Nplus1_fid);

	weightN_Nplus1_time = weightsN_Nplus1_hdr{i_weightN_Nplus1}.time;
	if weightN_Nplus1_time > max_weightN_Nplus1_time
	  max_weightN_Nplus1_time = weightN_Nplus1_time;
	  max_checkpoint = i_checkpoint;
	endif
      endfor %% i_checkpoint

      %% get weights headers
      checkpoint_dir = checkpoints_list{max_checkpoint,:};
      for i_layerN_Nplus1 = 1 : num_layersN_Nplus1_list-1
	weightsN_Nplus1_file = ...
	    [checkpoint_dir, filesep, ...
	     weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2-1}, ...
	     weightsN_Nplus1_list{i_weightN_Nplus1, i_layerN_Nplus1*2}, ".pvp"]
	if ~exist(weightsN_Nplus1_file, "file")
	  error(["file does not exist: ", weightsN_Nplus1_file]);
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
      

      %% get rank order of presynaptic elements
      i_layerN_Nplus1 = 1;
      tmp_ndx = sparse_weightsN_Nplus1_ndx(i_weightN_Nplus1);
      if analyze_Sparse_flag
	tmp_rank = Sparse_hist_rank_array{tmp_ndx};
      else
	tmp_rank = [];
      endif
      if analyze_Sparse_flag && ~isempty(tmp_rank)
	pre_hist_rank = tmp_rank;
      else
	pre_hist_rank = (1:weightsN_Nplus1_hdr{i_weightN_Nplus1, i_layerN_Nplus1}.nf);
      endif


      %% loop over checkpoints
      for i_checkpoint = 1 : num_checkpoints
	if(i_checkpoint ~= max_checkpoint)
	  continue;
	endif

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
	num_patchesN_Nplus1_per_fig = min(num_patchesN_Nplus1, max_patches);
	num_figsN_Nplus1 = ceil(num_patchesN_Nplus1/num_patchesN_Nplus1_per_fig);
	weightsN_Nplus1_fig = zeros(num_figsN_Nplus1,1);      
	if plot_weightsN_Nplus1_flag && i_checkpoint == max_checkpoint
	  for i_figN_Nplus1 = 1 : num_figsN_Nplus1
	    weightsN_Nplus1_fig(i_figN_Nplus1) = figure;
	    set(weightsN_Nplus1_fig(i_figN_Nplus1), "name", [weightsN_Nplus1_name, "_", num2str(i_figN_Nplus1)]);
	  endfor
	endif
	num_patchesN_Nplus1_rows = floor(sqrt(min(num_patchesN_Nplus1_per_fig, max_patches)));
	num_patchesN_Nplus1_cols = ceil(min(num_patchesN_Nplus1_per_fig, max_patches) / num_patchesN_Nplus1_rows);

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
	    for i_figN_Nplus1 = 1 : num_figsN_Nplus1
	      weightsNminus1_Nplus1_fig(i_figN_Nplus1) = figure;
	      set(weightsNminus1_Nplus1_fig, "name", [weightsN_Nplus1_name, "_", num2str(i_figN_Nplus1)]);
	    endfor  
	  endif %% plot_flag

	  max_shrinkage = 8; %% 
	  %% storage for next iteration of deconvolved weights
	  weightsNminus1_Nplus1_array = cell(num_figsN_Nplus1,1);
	  %% plot weights in rank order
	  for kf_preN_Nplus1_rank = 1  : num_patchesN_Nplus1
	    i_figN_Nplus1 = ceil(kf_preN_Nplus1_rank / num_patchesN_Nplus1_per_fig);
	    kf_preN_Nplus1_rank_per_fig = kf_preN_Nplus1_rank - (i_figN_Nplus1 - 1) * num_patchesN_Nplus1_per_fig;	  
	    kf_preN_Nplus1 = pre_hist_rank(kf_preN_Nplus1_rank);

	    plotNminus1_Nplus1_flag = ...
		plot_flag && ...
		i_layerN_Nplus1 == num_layersN_Nplus1_list-2 && ...
		i_checkpoint == max_checkpoint && ...
		kf_preN_Nplus1_rank <= max_patches;
	    if plotNminus1_Nplus1_flag
	      subplot(num_patchesN_Nplus1_rows, num_patchesN_Nplus1_cols, kf_preN_Nplus1_rank_per_fig); 
	    endif

	    num_weightsN_Nplus1_dims = ...
		ndims(weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1});
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
		  if ndims(weightsNminus1_N_vals) == 3 %%weightsNminus1_N_nfp == 1
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
	    %%min_patch = min(patch_tmp3(:));
	    maxabs_patch_tmp3 = max(abs(patch_tmp3(:)));
	    patch_tmp5 = ...
		(127.5 + 127.5*( (patch_tmp3) / (maxabs_patch_tmp3 + (maxabs_patch_tmp3==0))));
	    
	    if plotNminus1_Nplus1_flag
	      figure(weightsN_Nplus1_fig(i_figN_Nplus1));
	      subplot(num_patchesN_Nplus1_rows, num_patchesN_Nplus1_cols, kf_preN_Nplus1_rank_per_fig); 
	      imagesc(patch_tmp5); 
	      if ndims(patch_tmp5) == 2 
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

	    if isempty(weightsNminus1_Nplus1_array{i_figN_Nplus1})
	      weightsNminus1_Nplus1_array{i_figN_Nplus1} = ...
		  zeros(num_patchesN_Nplus1_rows*weightsNminus1_Nplus1_nyp_shrunken, ...
			num_patchesN_Nplus1_cols*weightsNminus1_Nplus1_nxp_shrunken, ...
			weightsNminus1_N_nfp);
	      weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1+1} = ...
		  zeros(weightsNminus1_Nplus1_nyp_shrunken, ...
			weightsNminus1_Nplus1_nxp_shrunken, ...
			weightsNminus1_N_nfp, ...
			num_patchesN_Nplus1);
	    endif
	    col_ndx = 1 + mod(kf_preN_Nplus1_rank_per_fig-1, num_patchesN_Nplus1_cols);
	    row_ndx = 1 + floor((kf_preN_Nplus1_rank_per_fig-1) / num_patchesN_Nplus1_cols);
	    maxabs_patchNminus1_Nplus1 = max(abs(patch_tmp5(:)));
				%normalized_patchNminus1_Nplus1 = 127.5 + 127.5 * (patch_tmp5 / (maxabs_patchNminus1_Nplus1 + (maxabs_patchNminus1_Nplus1==0)));
	    if ndims(patch_tmp5) == 3
	      weightsNminus1_Nplus1_array{i_figN_Nplus1}((1+(row_ndx-1)*weightsNminus1_Nplus1_nyp_shrunken):...
							 (row_ndx*weightsNminus1_Nplus1_nyp_shrunken), ...
							 (1+(col_ndx-1)*weightsNminus1_Nplus1_nxp_shrunken): ...
							 (col_ndx*weightsNminus1_Nplus1_nxp_shrunken),:) = ...
		  patch_tmp5;
	    else
	      weightsNminus1_Nplus1_array{i_figN_Nplus1}((1+(row_ndx-1)*weightsNminus1_Nplus1_nyp_shrunken):...
							 (row_ndx*weightsNminus1_Nplus1_nyp_shrunken), ...
							 (1+(col_ndx-1)*weightsNminus1_Nplus1_nxp_shrunken): ...
							 (col_ndx*weightsNminus1_Nplus1_nxp_shrunken)) = ...
		  patch_tmp5;
	    endif
	    %%normalized_patchNminus1_Nplus1; %%
	    
	    %% set weightsN_Nplus1_vals to patch_tmp3
	    %%	  weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1+1}(:, :, :, kf_preN_Nplus1_rank) = ...
	    %%	      patch_tmp3;
	    if ndims(patch_tmp3) == 3
	      weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1+1}(:, :, :, kf_preN_Nplus1) = ...
		  patch_tmp3;
 	    else
	      weightsN_Nplus1_vals{i_weightN_Nplus1, i_layerN_Nplus1+1}(:, :, kf_preN_Nplus1) = ...
		  patch_tmp3;
	    endif
	    
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
	  
	  for i_figN_Nplus1 = 1 : num_figsN_Nplus1
	    weightsN_Nplus1_str{i_figN_Nplus1} = ...
		[weightsN_Nplus1_list{i_weightN_Nplus1,1}, weightsN_Nplus1_list{i_weightN_Nplus1,2}, ...
		 "_", ...
		 weightsN_Nplus1_list{i_weightN_Nplus1,3}, weightsN_Nplus1_list{i_weightN_Nplus1,4}, ...
		 "_", num2str(weightsN_Nplus1_time, "%08d"), ...
		 "_", num2str(i_figN_Nplus1)];
	  endfor
	  if plot_flag && ...
		i_checkpoint == max_checkpoint && ...
		i_layerN_Nplus1 == num_layersN_Nplus1_list-2 
	    for i_figN_Nplus1 = 1 : num_figsN_Nplus1
	      figure(weightsN_Nplus1_fig(i_figN_Nplus1));
	      saveas(weightsN_Nplus1_fig(i_figN_Nplus1), [weightsN_Nplus1_dir, filesep, weightsN_Nplus1_str{i_figN_Nplus1}, ".png"], "png");
	      if ~isempty(labelWeights_vals) 
		labeledWeightsN_Nplus1_str = ...
		    ["labeledWeights_", ...
		     weightsN_Nplus1_str{i_figN_Nplus1}];
		labeledWeights_fig = figure("name", labeledWeightsN_Nplus1_str{i_figN_Nplus1});
		title(labeledWeightsN_Nplus1_str{i_figN_Nplus1});
		rows_labeledWeights = ceil(sqrt(size(labelWeights_vals,1)));
		cols_labeledWeights = ceil(size(labelWeights_vals,1) / rows_labeledWeights);
		for label = 1:size(labelWeights_vals,1)
		  subplot(rows_labeledWeights, cols_labeledWeights, label);
		  labeledWeights_subplot_str = ...
		      [num2str(label, "%d")];
		  imagesc(squeeze(labeledWeightsNminus1_Nplus1{label}));
		  title(labeledWeights_subplot_str);
		  axis off
		endfor %% label = 1:size(labelWeights_vals,1)
		saveas(labeledWeights_fig,  [weightsN_Nplus1_dir, filesep, labeledWeightsN_Nplus1_str{i_figN_Nplus1}, ".png"], "png");
	      endif %%  ~isempty(labelWeights_vals) 
	    endfor
	  endif
	  
	  %%	  imwrite(uint8(weightsNminus1_Nplus1_array), ...
	  %%		  [weightsN_Nplus1_movie_dir, filesep, weightsN_Nplus1_str, ".png"], "png");
	  if i_checkpoint == max_checkpoint
 	    save("-mat", ...
		 [weightsN_Nplus1_movie_dir, filesep, weightsN_Nplus1_str{1}, ".mat"], ...
		 "weightsNminus1_Nplus1_array");
	  endif
	  if i_layerN_Nplus1 == num_layersN_Nplus1_list-2 
	    for i_figN_Nplus1 = 1 : num_figsN_Nplus1
	      weightsNminus1_Nplus1_array_list{i_weightN_Nplus1, i_figN_Nplus1} = weightsNminus1_Nplus1_array{i_figN_Nplus1};
	      num_combineN_Nplus1 = 1;
	      for i_weights = 1 : num_weights_list
		if sparse_weightsN_Nplus1_ndx(i_weightN_Nplus1) == sparse_weights_ndx(i_weights)
		  size_weightsNminus1_Nplus1_array = ...
		      size(weightsNminus1_Nplus1_array_list{i_weightN_Nplus1, i_figN_Nplus1});
		  resized_weight_patch_array = ...
		      imresize(weight_patch_array_list{i_weights}, size_weightsNminus1_Nplus1_array(1:2));
		  maxabs_resized_weight_patch_array = max(abs(resized_weight_patch_array(:)));
		  resized_weight_patchNminus1_Nplus1_array_uint8 = ...
		      127.5 + 127.5 * (resized_weight_patch_array / (maxabs_resized_weight_patch_array + (maxabs_resized_weight_patch_array==0)));
		  weightsNminus1_Nplus1_array_list{i_weightN_Nplus1, i_figN_Nplus1} = ...
		      weightsNminus1_Nplus1_array_list{i_weightN_Nplus1, i_figN_Nplus1}  + ...
		      resized_weight_patchNminus1_Nplus1_array_uint8;
		  num_combineN_Nplus1 = num_combineN_Nplus1 + 1;
		endif
	      endfor
	      weightsNminus1_Nplus1_array_list{i_weightN_Nplus1, i_figN_Nplus1} = ...
		  weightsNminus1_Nplus1_array_list{i_weightN_Nplus1, i_figN_Nplus1} / num_combineN_Nplus1;
	      maxabs_weightsNminus1_Nplus1_array = max(abs(weightsNminus1_Nplus1_array_list{i_weightN_Nplus1, i_figN_Nplus1}(:)));
	      weightsNminus1_Nplus1_array_uint8 = ...
		  (127.5 + 127.5*(weightsNminus1_Nplus1_array_list{i_weightN_Nplus1, i_figN_Nplus1} / ...
				  (maxabs_weightsNminus1_Nplus1_array + (maxabs_weightsNminus1_Nplus1_array == 0))));
	      imwrite(uint8(weightsNminus1_Nplus1_array_uint8), ...
		      [weightsN_Nplus1_movie_dir, filesep, weightsN_Nplus1_str{i_figN_Nplus1}, ".png"], "png");
				%keyboard;
	    endfor
	  endif
	  

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
		    weightsN_Nplus1_str{i_figN_Nplus1}], "png");
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
	  endif  %% ~isempty(labelWeights_vals) && ~isempty(labelWeights_time)

	endfor %% i_checkpoint

      endfor %% i_layerN_Nplus1

    endfor %% i_weightN_Nplus1
    
  endif  %% plot_weights



  deRecon_flag = true && ~isempty(labelWeights_vals);
  if deRecon_flag
    num_deRecon = 3;
    deRecon_sparse_weightsN_Nplus1_ndx = 3;
    deRecon_struct = Sparse_struct_array{deRecon_sparse_weightsN_Nplus1_ndx};
    num_deRecon_frames = size(deRecon_struct,1);
    Recon_dir = [output_dir, filesep, "Recon"];
    for i_deRecon_frame = 1 : num_deRecon_frames
      deRecon_time = deRecon_struct{i_deRecon_frame}.time
      deRecon_indices = deRecon_struct{i_deRecon_frame}.values(:,1);
      deRecon_vals = deRecon_struct{i_deRecon_frame}.values(:,2);
      [deRecon_vals_sorted, deRecon_vals_rank] = sort(deRecon_vals, "descend");
      deRecon_indices_sorted = deRecon_indices(deRecon_vals_rank)+1;
      num_deRecon_indices = length(deRecon_indices(:));
      deRecon_hist_rank = Sparse_hist_rank_array{deRecon_sparse_weightsN_Nplus1_ndx}(:);
      for i_deRecon_index = 1 : min(num_deRecon, num_deRecon_indices)
	deRecon_rank = find(deRecon_hist_rank == deRecon_indices_sorted(i_deRecon_index))
	if deRecon_rank > num_patchesN_Nplus1
	  continue;
	endif
	col_ndx = 1 + mod(deRecon_rank-1, num_patchesN_Nplus1_cols);
	row_ndx = 1 + floor((deRecon_rank-1) / num_patchesN_Nplus1_cols);
	row_indices = (1+(row_ndx-1)*weightsNminus1_Nplus1_nyp_shrunken):(row_ndx*weightsNminus1_Nplus1_nyp_shrunken);
	col_indices = (1+(col_ndx-1)*weightsNminus1_Nplus1_nxp_shrunken):(col_ndx*weightsNminus1_Nplus1_nxp_shrunken);
	deRecon_patch = weightsNminus1_Nplus1_array(row_indices, col_indices, :);
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
      endfor %% i_deRecon_index
      %%    disp(mat2str(labelWeights_vals(:,deRecon_rank(1: min(num_deRecon, num_deRecon_indices))));
      deRecon_labelWeights = labelWeights_vals(:,deRecon_indices_sorted);
      deRecon_label_activity = repmat(deRecon_vals_sorted(:)',[size(labelWeights_vals,1),1]);
      deRecon_label_prod = deRecon_labelWeights .* deRecon_label_activity;
      deRecon_vals_sorted
      sum_labelWeights = sum(deRecon_label_prod,2)
    endfor %% i_deRecon
  endif %% deReconFlag

endif %% plot_flag





