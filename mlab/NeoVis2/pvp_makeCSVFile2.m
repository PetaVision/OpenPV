
function [num_frames, ...
	  tot_frames, ...
	  nnz_frames, ...
	  tot_time, ...
	  CSV_struct] = ...
      pvp_makeCSVFile2(NEOVISION_DATASET_ID, ...
		       NEOVISION_DISTRIBUTION_ID, ...
		       repo_path, ...
		       ObjectType, ...
		       pvp_edge_filter, ...
		       pvp_version_str, ...
		       clip_name, ...
		       pvp_frame_skip, ...
		       pvp_frame_offset, ...
		       clip_path, ...
		       num_ODD_kernels, ...
		       pvp_bootstrap_str, ...
		       pvp_bootstrap_level_str, ...
		       patch_size, ...
		       std_patch_size, ...
		       max_patch_size, ...
		       min_patch_size, ...
		       pvp_layer, ...
		       pvp_path, ...
		       training_flag, ...
		       num_procs)
  %% takes PetaVision non-spiking activity files generated in response to
  %% a video clip and produces a CSV file indicating locations of
  %% specified object
  %%keyboard;
  more off
  begin_time = time();
  
  home_path = ...
      [filesep, "home", filesep, "garkenyon", filesep];
  %% [filesep, "Users", filesep, "gkenyon", filesep, "NeoVision", filesep]; %%
  
  num_input_args = 0;
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("NEOVISION_DATASET_ID") || isempty(NEOVISION_DATASET_ID)
    NEOVISION_DATASET_ID = "Heli"; %% "Tower"; %%  "Tail"; %% 
  endif
  neovision_dataset_id = tolower(NEOVISION_DATASET_ID); %% 
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("NEOVISION_DISTRIBUTION_ID") || isempty(NEOVISION_DISTRIBUTION_ID)
    NEOVISION_DISTRIBUTION_ID = "Challenge"; %% "Formative"; %% "Training"; %%  
  endif
  neovision_distribution_id = tolower(NEOVISION_DISTRIBUTION_ID); %% 
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("repo_path") || isempty(repo_path)
    repo_path = [filesep, "mnt", filesep, "data1", filesep, "repo", filesep];
  endif
  program_path = [repo_path, ...
		  "neovision-programs-petavision", filesep, ...
		  NEOVISION_DATASET_ID, filesep, ...
		  NEOVISION_DISTRIBUTION_ID, filesep]; %% 		  
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("ObjectType") || isempty(ObjectType)
    ObjectType = "Car"; %% "Cyclist"; %%  
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist(pvp_edge_filter) || isempty(pvp_edge_filter)
    pvp_edge_filter = "canny";
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist(pvp_version_str) %% || isempty(pvp_version_str)
    pvp_version_str = "2";
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("clip_name") || isempty(clip_name)
    clip_name = "027";
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_frame_skip") || isempty(pvp_frame_skip)
    pvp_frame_skip = 1;
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_frame_offset") || isempty(pvp_frame_offset)
    pvp_frame_offset = 1;
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("clip_path") || isempty(clip_path)
    clip_path = [program_path, ...
		 pvp_edge_filter, filesep, ...
		 clip_name, filesep]; %% 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("num_ODD_kernels") || isempty(num_ODD_kernels)
    num_ODD_kernels = 3;  %% 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_bootstrap_str") 
    pvp_bootstrap_str = ""; %% "_bootstrap";  %% 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_bootstrap_level_str") 
    pvp_bootstrap_level_str = ""; %% "1";  %% 
  endif
  clip_log_dir = [repo_path, "neovision-programs-petavision", filesep, ...
		  NEOVISION_DATASET_ID, filesep, NEOVISION_DISTRIBUTION_ID, filesep, ...
		  "log", filesep, ObjectType, filesep];
  clip_log_pathname = [clip_log_dir, "log.txt"];
  clip_log_struct = struct;
  if exist(clip_log_pathname, "file")
    clip_log_fid = fopen(clip_log_pathname, "r");
    clip_log_struct.tot_unread = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_rejected = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_clips = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_DoG = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_canny = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_cropped = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_mean = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_std = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_border_artifact_top = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_border_artifact_bottom = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_border_artifact_left = str2num(fgets(clip_log_fid));
    clip_log_struct.tot_border_artifact_right = str2num(fgets(clip_log_fid));
    clip_log_struct.ave_original_size = str2num(fgets(clip_log_fid));
    clip_log_struct.ave_cropped_size = str2num(fgets(clip_log_fid));
    clip_log_struct.std_original_size = str2num(fgets(clip_log_fid));
    clip_log_struct.std_cropped_size = str2num(fgets(clip_log_fid));
    clip_log_struct.max_cropped_size = str2num(fgets(clip_log_fid));
    clip_log_struct.min_cropped_size = str2num(fgets(clip_log_fid));
    fclose(clip_log_fid);
    default_patch_size = clip_log_struct.ave_cropped_size;
    default_std_size = clip_log_struct.std_cropped_size;
    default_max_size = clip_log_struct.max_cropped_size;
    default_min_size = clip_log_struct.min_cropped_size;
  else
    default_patch_size = [64, 64];
    default_std_size = [0, 0];
    default_max_size = default_patch_size;
    default_min_size = default_patch_size;
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("patch_size") || isempty(patch_size)
    patch_size = default_patch_size;
    disp(["patch_size = ", num2str(patch_size)]);
  endif
  if nargin < num_input_args || ~exist("patch_size") || isempty(patch_size)
    std_patch_size = default_std_size;
    disp(["std_patch_size = ", num2str(std_patch_size)]);
  endif
  if nargin < num_input_args || ~exist("patch_size") || isempty(patch_size)
    max_patch_size = default_max_size;
    disp(["max_patch_size = ", num2str(max_patch_size)]);
  endif
  if nargin < num_input_args || ~exist("patch_size") || isempty(patch_size)
    min_patch_size = default_min_size;
    disp(["min_patch_size = ", num2str(min_patch_size)]);
  endif
  if nargin < num_input_args || ~exist("pvp_layer") || isempty(pvp_layer)
    pvp_layer = 7;  %% 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_path") || isempty(pvp_path)
    pvp_path = [program_path, "activity", filesep, ...
		clip_name, filesep, ObjectType, num2str(num_ODD_kernels), pvp_bootstrap_str, pvp_bootstrap_level_str, filesep, pvp_edge_filter, pvp_version_str, filesep];
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("training_flag") || isempty(training_flag)
    training_flag = 1;
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("num_procs") || isempty(num_procs)
    num_procs = 24;  %% 
  endif
  
  global VERBOSE_FLAG
  VERBOSE_FLAG = 1;

  global pvp_patch_size
  pvp_patch_size = patch_size;

  global pvp_std_patch_size
  pvp_std_patch_size = std_patch_size;
  
  global pvp_max_patch_size
  pvp_max_patch_size = max_patch_size;
  
  global pvp_min_patch_size
  pvp_min_patch_size = min_patch_size;
  
  global pvp_training_flag
  pvp_training_flag = training_flag;

  global pvp_use_PANN_boundingBoxes
  if isempty(pvp_use_PANN_boundingBoxes)
    pvp_use_PANN_boundingBoxes = 0;
  endif
  
  %%setenv('GNUTERM', 'x11');
  image_type = ".png";
  
  %% path to generic image processing routines
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);
  
  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);
  
  global ODD_subdir
  ODD_path = [program_path, "ODD", filesep]; 
  mkdir(ODD_path);
  ODD_clip_dir = [ODD_path, clip_name, filesep];
  mkdir(ODD_clip_dir);
  ODD_dir = [ODD_clip_dir, ObjectType, num2str(num_ODD_kernels), pvp_bootstrap_str, pvp_bootstrap_level_str, filesep];
  mkdir(ODD_dir);
  ODD_subdir = [ODD_dir, pvp_edge_filter, pvp_version_str, filesep];
  mkdir(ODD_subdir);
  
  ROC_path = [program_path, "ROC", filesep]; 
  mkdir(ROC_path);
  ROC_clip_dir = [ROC_path, clip_name, filesep];
  mkdir(ROC_clip_dir);
  ROC_dir = [ROC_clip_dir, ObjectType, num2str(num_ODD_kernels), pvp_bootstrap_str, pvp_bootstrap_level_str, filesep];
  mkdir(ROC_dir);
  ROC_subdir = [ROC_dir, pvp_edge_filter, pvp_version_str, filesep];
  mkdir(ROC_subdir);

  pvp_results_path = [program_path, "results", filesep];
  mkdir(pvp_results_path);
  pvp_results_dir = [pvp_results_path, clip_name, filesep];
  mkdir(pvp_results_dir);
  pvp_results_subdir0 = ...
      [pvp_results_dir, ObjectType, num2str(num_ODD_kernels), pvp_bootstrap_str, pvp_bootstrap_level_str, filesep];
  mkdir(pvp_results_subdir0);
  pvp_results_subdir = ...
      [pvp_results_subdir0, pvp_edge_filter, pvp_version_str, filesep];
  mkdir(pvp_results_subdir);

  global target_bootstrap_dir
  global distractor_bootstrap_dir
  if isempty(pvp_bootstrap_str)
    target_bootstrap_dir = ...
	[repo_path,  "neovision-chips-", neovision_dataset_id, filesep, NEOVISION_DATASET_ID, "-PNG-", NEOVISION_DISTRIBUTION_ID, filesep];
    mkdir(target_bootstrap_dir);
    target_bootstrap_dir = ...
	[target_bootstrap_dir, ObjectType, "_bootstrap0", filesep];
    mkdir(target_bootstrap_dir);
    distractor_bootstrap_dir = ...
	[repo_path,  "neovision-chips-", neovision_dataset_id, filesep, NEOVISION_DATASET_ID, "-PNG-", NEOVISION_DISTRIBUTION_ID, filesep];
    mkdir(distractor_bootstrap_dir);
    distractor_bootstrap_dir = ...
	[distractor_bootstrap_dir, "distractor", "_bootstrap0", filesep];
    mkdir(distractor_bootstrap_dir);
  else
    target_bootstrap_dir = ...
	[repo_path,  "neovision-chips-", neovision_dataset_id, filesep, NEOVISION_DATASET_ID, "-PNG-", NEOVISION_DISTRIBUTION_ID, filesep, ObjectType, pvp_bootstrap_str, pvp_bootstrap_level_str, filesep];
    distractor_bootstrap_dir = ...
	[repo_path,  "neovision-chips-", neovision_dataset_id, filesep, NEOVISION_DATASET_ID, "-PNG-", NEOVISION_DISTRIBUTION_ID, filesep, "distractor", pvp_bootstrap_str, pvp_bootstrap_level_str, filesep];
  endif
  mkdir(target_bootstrap_dir);
  mkdir(distractor_bootstrap_dir);
  

  global pvp_density_thresh
  hit_and_miss_stats_pathname = [ROC_subdir, "hit_and_miss_stats.txt"];
  BB_stats_pathname = [ROC_subdir, "BB_stats.txt"];
  if 0 %%~pvp_training_flag && exist(hit_and_miss_stats_pathname, "file")
    hit_and_miss_stats_struct = struct;
    hit_and_miss_stats_fid = fopen(hit_and_miss_stats_pathname, "r");
    hit_and_miss_stats_struct.pvp_tot_hits = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_tot_miss = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_min_hit_density = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_max_hit_density = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_ave_hit_density = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_std_hit_density = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_median_hit_density = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_min_miss_density = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_max_miss_density = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_ave_miss_density = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_std_miss_density = str2num(fgets(hit_and_miss_stats_fid));
    hit_and_miss_stats_struct.pvp_median_miss_density = str2num(fgets(hit_and_miss_stats_fid));
    fclose(hit_and_miss_stats_fid);
    pvp_density_thresh = ...
	(hit_and_miss_stats_struct.pvp_ave_hit_density); 
  elseif 0 %% ~pvp_training_flag && exist(BB_stats_pathname, "file")
    BB_stats_struct = struct;
    BB_stats_fid = fopen(BB_stats_pathname, "r");
    BB_stats_struct.pvp_min_BB_density = str2num(fgets(BB_stats_fid));
    BB_stats_struct.pvp_max_BB_density = str2num(fgets(BB_stats_fid));
    BB_stats_struct.pvp_ave_BB_density = str2num(fgets(BB_stats_fid));
    BB_stats_struct.pvp_std_BB_density = str2num(fgets(BB_stats_fid));
    if ~feof(BB_stats_fid)
      BB_stats_struct.pvp_median_BB_density = str2num(fgets(BB_stats_fid));
    endif
    fclose(BB_stats_fid);
    pvp_density_thresh = ...
	(BB_stats_struct.pvp_ave_BB_density(1)); %% + BB_stats_struct.pvp_std_BB_density(1)) / 2;
  else
    pvp_density_thresh = -1.0;  %% flag to use ave density across image
  endif
  disp(["pvp_density_thresh = ", num2str(pvp_density_thresh)]);
  
  %%keyboard;
  i_CSV = 0;
  if 1 %% ~strcmp(NEOVISION_DISTRIBUTION_ID,"Challenge")  
    true_CSV_path = ...
	[repo_path, ...
	 "neovision-data-", neovision_distribution_id, "-", neovision_dataset_id, ...
	 filesep, "CSV", filesep];
    true_CSV_filename = [clip_name, ".csv"];
    true_CSV_pathname = [true_CSV_path, true_CSV_filename];
    if ~exist(true_CSV_pathname, "file")
      error(["~exist: true_CSV_pathname = ", true_CSV_pathname]);
    endif
    true_CSV_fid = fopen(true_CSV_pathname, "r");
    true_CSV_header = fgets(true_CSV_fid);
    true_CSV_list = cell(1);
    while ~feof(true_CSV_fid)
      i_CSV = i_CSV + 1;
      true_CSV_list{i_CSV} = fgets(true_CSV_fid);
    endwhile
    fclose(true_CSV_fid);
  elseif 0 %% strcmp(NEOVISION_DISTRIBUTION_ID,"Challenge")
    true_CSV_path = ...
	[repo_path, ...
	 "neovision-results-", neovision_distribution_id, "-", neovision_dataset_id, filesep, clip_name, filesep];
    true_CSV_filename = [NEOVISION_DATASET_ID, "_", clip_name, "_PANN_998_Objects.csv"];
    true_CSV_pathname = [true_CSV_path, true_CSV_filename];
    if ~exist(true_CSV_pathname, "file")
      error(["~exist: true_CSV_pathname = ", true_CSV_pathname]);
    endif
    true_CSV_fid = fopen(true_CSV_pathname, "r");
    true_CSV_header = fgets(true_CSV_fid);
    true_CSV_list = cell(1);
    while ~feof(true_CSV_fid)
      i_CSV = i_CSV + 1;
      true_CSV_list{i_CSV} = fgets(true_CSV_fid);
    endwhile
    fclose(true_CSV_fid);    
  else
    true_CSV_header = "Frame,BoundingBox_X1,BoundingBox_Y1,BoundingBox_X2,BoundingBox_Y2,BoundingBox_X3,BoundingBox_Y3,BoundingBox_X4,BoundingBox_Y4,ObjectType,Occlusion,Ambiguous,Confidence,SiteInfo,Version\n";
  endif
  num_true_CSV = i_CSV;
  
  %% get frame IDs 
  clip_dir = clip_path; %% 
  if ~exist(clip_dir, "dir")
    error(["~exist(clip_dir):", clip_dir]);
  endif
  frameIDs_path = ...
      [clip_dir, '*', image_type];
  frame_pathnames_all = glob(frameIDs_path);
  num_frames = size(frame_pathnames_all,1);
  disp(["num_frames = ", num2str(num_frames)]);
  
  tot_frames = length(pvp_frame_offset : pvp_frame_skip : num_frames);
  disp(["tot_frames = ", num2str(tot_frames)]);
  
  %% read pvp activity into cell array
  [pvp_fid, ...
   pvp_header, ...
   pvp_index ] = ...
      pvp_openActivityFile(pvp_path, pvp_layer);
  [layerID] = neoVisLayerID(pvp_layer);
  
  global NFEATURES NCOLS NROWS N
  NCOLS = pvp_header(pvp_index.NX_GLOBAL);
  NROWS = pvp_header(pvp_index.NY_GLOBAL);
  NFEATURES = pvp_header(pvp_index.NF);
  N = NFEATURES * NCOLS * NROWS;
  
  pvp_time = zeros(tot_frames, 1);
  pvp_offset = zeros(tot_frames, 1);
  pvp_time = cell(tot_frames, 1);
  pvp_activity = cell(tot_frames, 1);
  frame_pathnames = cell(tot_frames, 1);
  
  %%keyboard;
  pvp_offset_tmp = 0;
  i_frame = 0;
  for j_frame = pvp_frame_offset : pvp_frame_skip : num_frames
    i_frame = i_frame + 1;
    pvp_frame = j_frame + pvp_layer - 1;
    [pvp_time{i_frame},...
     pvp_activity{i_frame}, ...
     pvp_offset(i_frame)] = ...
	pvp_readSparseLayerActivity(pvp_fid, pvp_frame, pvp_header, pvp_index, pvp_offset_tmp);
    if pvp_offset(i_frame) == -1
      break;
      i_frame = i_frame - 1;
    endif
    pvp_offset_tmp = pvp_offset(i_frame);
    frame_pathnames{i_frame} = frame_pathnames_all{j_frame};
    disp(["i_frame = ", num2str(i_frame)]);
    disp(["pvp_time = ", num2str(pvp_time{i_frame})]);
    disp(["frame_ID = ", frame_pathnames{i_frame}]);
    disp(["mean(pvp_activty) = ", num2str(mean(pvp_activity{i_frame}(:)))]);    
  endfor
  fclose(pvp_fid);
  nnz_frames = i_frame;
  disp(["nnz_frames = ", num2str(nnz_frames)]);
  if nnz_frames <= 0
    return;
  endif

  if num_procs > nnz_frames
    num_procs = nnz_frames;
  endif
  
  %% struct for storing rank order of comma separators between fields
  truth_CSV_struct = cell(tot_frames, 1);
  DCR_CSV_struct = cell(tot_frames, 1);
  other_CSV_struct = cell(tot_frames, 1);
  if pvp_training_flag
    true_CSV_comma_rank = struct;
    true_CSV_comma_rank.Frame = [1, 2];
    true_CSV_comma_rank.BoundingBox_X1 = [2, 3];
    true_CSV_comma_rank.BoundingBox_Y1 = [3, 4];
    true_CSV_comma_rank.BoundingBox_X2 = [4, 5];
    true_CSV_comma_rank.BoundingBox_Y2 = [5, 6];
    true_CSV_comma_rank.BoundingBox_X3 = [6, 7];
    true_CSV_comma_rank.BoundingBox_Y3 = [7, 8];
    true_CSV_comma_rank.BoundingBox_X4 = [8, 9];
    true_CSV_comma_rank.BoundingBox_Y4 = [9, 10];
    true_CSV_comma_rank.ObjectType = [10, 11];
    num_truth_BBs = 0;
    num_other_BBs = 0;
    num_DCR_BBs = 0;
    for i_CSV = 1 : num_true_CSV
      true_CSV_comma_ndx = [1, strfind(true_CSV_list{i_CSV}, ",")];
      ObjectType_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.ObjectType(1))+1;
      ObjectType_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.ObjectType(2))-1;
      CSV_ObjectType = true_CSV_list{i_CSV}(ObjectType_ndx(1):ObjectType_ndx(2));
      truth_CSV_struct_tmp.ObjectType = CSV_ObjectType;
      Frame_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.Frame(1));
      Frame_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.Frame(2))-1;
      true_Frame = true_CSV_list{i_CSV}(Frame_ndx(1):Frame_ndx(2));
      i_frame = str2num(true_Frame) + 1;
      if i_frame > nnz_frames
	break;
      endif
      truth_CSV_struct_tmp = struct;
      truth_CSV_struct_tmp.Frame = true_Frame;
      BoundingBox_X1_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X1(1))+1;
      BoundingBox_X1_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X1(2))-1;
      BoundingBox_X1 = true_CSV_list{i_CSV}(BoundingBox_X1_ndx(1):BoundingBox_X1_ndx(2));
      truth_CSV_struct_tmp.BoundingBox_X1 = str2num(BoundingBox_X1);
      BoundingBox_Y1_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y1(1))+1;
      BoundingBox_Y1_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y1(2))-1;
      BoundingBox_Y1 = true_CSV_list{i_CSV}(BoundingBox_Y1_ndx(1):BoundingBox_Y1_ndx(2));
      truth_CSV_struct_tmp.BoundingBox_Y1 = str2num(BoundingBox_Y1);
      BoundingBox_X2_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X2(1))+1;
      BoundingBox_X2_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X2(2))-1;
      BoundingBox_X2 = true_CSV_list{i_CSV}(BoundingBox_X2_ndx(1):BoundingBox_X2_ndx(2));
      truth_CSV_struct_tmp.BoundingBox_X2 = str2num(BoundingBox_X2);
      BoundingBox_Y2_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y2(1))+1;
      BoundingBox_Y2_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y2(2))-1;
      BoundingBox_Y2 = true_CSV_list{i_CSV}(BoundingBox_Y2_ndx(1):BoundingBox_Y2_ndx(2));
      truth_CSV_struct_tmp.BoundingBox_Y2 = str2num(BoundingBox_Y2);
      BoundingBox_X3_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X3(1))+1;
      BoundingBox_X3_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X3(2))-1;
      BoundingBox_X3 = true_CSV_list{i_CSV}(BoundingBox_X3_ndx(1):BoundingBox_X3_ndx(2));
      truth_CSV_struct_tmp.BoundingBox_X3 = str2num(BoundingBox_X3);
      BoundingBox_Y3_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y3(1))+1;
      BoundingBox_Y3_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y3(2))-1;
      BoundingBox_Y3 = true_CSV_list{i_CSV}(BoundingBox_Y3_ndx(1):BoundingBox_Y3_ndx(2));
      truth_CSV_struct_tmp.BoundingBox_Y3 = str2num(BoundingBox_Y3);
      BoundingBox_X4_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X4(1))+1;
      BoundingBox_X4_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X4(2))-1;
      BoundingBox_X4 = true_CSV_list{i_CSV}(BoundingBox_X4_ndx(1):BoundingBox_X4_ndx(2));
      truth_CSV_struct_tmp.BoundingBox_X4 = str2num(BoundingBox_X4);
      BoundingBox_Y4_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y4(1))+1;
      BoundingBox_Y4_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y4(2))-1;
      BoundingBox_Y4 = true_CSV_list{i_CSV}(BoundingBox_Y4_ndx(1):BoundingBox_Y4_ndx(2));
      truth_CSV_struct_tmp.BoundingBox_Y4 = str2num(BoundingBox_Y4);
      if strcmp(CSV_ObjectType, ObjectType)
	truth_CSV_struct{i_frame}{num_truth_BBs + 1} = truth_CSV_struct_tmp;
	num_truth_BBs = length(truth_CSV_struct{i_frame});
      elseif strcmp(CSV_ObjectType, "DCR")
	DCR_CSV_struct{i_frame}{num_DCR_BBs + 1} = truth_CSV_struct_tmp;
	num_DCR_BBs = length(DCR_CSV_struct{i_frame});
      else
	other_CSV_struct{i_frame}{num_other_BBs + 1} = truth_CSV_struct_tmp;
	num_other_BBs = length(other_CSV_struct{i_frame});
      endif
    endfor
  endif %% pvp_training_flag
  
  disp("");
  %%keyboard;
  if num_procs > 1
    CSV_struct = parcellfun(num_procs, @pvp_makeCSVFileKernel2, ...
			    frame_pathnames, pvp_time, pvp_activity, truth_CSV_struct, other_CSV_struct, DCR_CSV_struct, ...
			    "UniformOutput", false);
  else
    CSV_struct = cellfun(@pvp_makeCSVFileKernel2, ...
			 frame_pathnames, pvp_time, pvp_activity, truth_CSV_struct, other_CSV_struct, DCR_CSV_struct, ...
			 "UniformOutput", false);
  endif
  
  disp("");

  frames_per_CSV_file = 150000;
  num_CSV_files = length(CSV_struct);
  disp(["num_CSV_files = ", num2str(num_CSV_files)]);
  num_CSV_files_tmp = ceil(num_CSV_files / frames_per_CSV_file);
  CSV_ObjectType = ObjectType;
  CSV_Occlusion = 0; %% false
  CSV_Ambiguous = 0; %% false
  CSV_SiteInfo = 0;
  CSV_Version = 1.4;
  pvp_tot_hits = 0;
  pvp_tot_miss = 0;
  pvp_miss_density = [];
  pvp_hit_density = [];
  if ~isempty(CSV_struct) && ~isempty(CSV_struct{1})
    disp(fieldnames(CSV_struct{1}));
  endif
  for i_CSV_file = 1 : num_CSV_files_tmp
    if isempty(pvp_bootstrap_level_str)
      csv_id = i_CSV_file + 1;
    else
      csv_id = i_CSV_file + 1 + str2num(pvp_bootstrap_level_str);
    endif
    pvp_results_filename = ...
	[NEOVISION_DATASET_ID, "_", NEOVISION_DISTRIBUTION_ID, "_", clip_name,...
	 "_PetaVision_", ObjectType, "_", num2str(csv_id, "%3.3i"), ".csv"];
    pvp_results_pathname = [pvp_results_subdir, pvp_results_filename];
    disp(["pvp_results_pathname = ", pvp_results_pathname]);
    pvp_results_fid = fopen(pvp_results_pathname, "w");
    fputs(pvp_results_fid, true_CSV_header);
    start_frame = 1 + frames_per_CSV_file * (i_CSV_file - 1);
    stop_frame = frames_per_CSV_file * i_CSV_file;
    stop_frame = min(stop_frame, nnz_frames);
    for i_frame = start_frame : stop_frame
      %%CSV_struct{i_frame}.Frame = pvp_frame_offset + pvp_frame_skip*(i_frame-1) - 1;
      %%CSV_struct{i_frame}.Frame = i_frame - 1;
      if isempty(CSV_struct{i_frame}) continue; endif
      if CSV_struct{i_frame}.num_active == 0 continue; endif;
      disp(["i_frame = ", num2str(i_frame)]);
      disp(["frame_ID = ", CSV_struct{i_frame}.frame_filename]);
      disp(["pvp_time = ", num2str(CSV_struct{i_frame}.pvp_time)]);
      disp(["mean(pvp_activty) = ", num2str(CSV_struct{i_frame}.mean_activity)]);    
      disp(["num_active = ", num2str(CSV_struct{i_frame}.num_active)]);
      if pvp_training_flag
	disp(["num_active_BB_mask = ", num2str(CSV_struct{i_frame}.num_active_BB_mask)]);
	disp(["num_active_BB_notmask = ", num2str(CSV_struct{i_frame}.num_active_BB_notmask)]);
	disp(["num_BB_mask = ", num2str(CSV_struct{i_frame}.num_BB_mask)]);
	disp(["num_BB_notmask = ", num2str(CSV_struct{i_frame}.num_BB_notmask)]);
      endif
      pvp_num_hits = length(CSV_struct{i_frame}.hit_list);
      if isempty(CSV_struct{i_frame}.hit_list) continue; endif
      pvp_tot_hits = pvp_tot_hits + pvp_num_hits;
      for i_hit = 1 : pvp_num_hits
        if isempty(CSV_struct{i_frame}.hit_list{i_hit}) continue; endif
	pvp_hit_density = [pvp_hit_density; CSV_struct{i_frame}.hit_list{i_hit}.hit_density];
	csv_str = [];
        csv_str = num2str(i_frame - 1); %%CSV_struct{i_frame}.Frame;
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.hit_list{i_hit}.BoundingBox_X1)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.hit_list{i_hit}.BoundingBox_Y1)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.hit_list{i_hit}.BoundingBox_X2)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.hit_list{i_hit}.BoundingBox_Y2)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.hit_list{i_hit}.BoundingBox_X3)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.hit_list{i_hit}.BoundingBox_Y3)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.hit_list{i_hit}.BoundingBox_X4)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.hit_list{i_hit}.BoundingBox_Y4)];
	csv_str = [csv_str, ",", CSV_ObjectType];
	csv_str = [csv_str, ",", num2str(CSV_Occlusion)];
	csv_str = [csv_str, ",", num2str(CSV_Ambiguous)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.hit_list{i_hit}.Confidence)];
	csv_str = [csv_str, ",", num2str(CSV_SiteInfo)];
	csv_str = [csv_str, ",", num2str(CSV_Version)];
	csv_str = [csv_str, "\n"];
	fputs(pvp_results_fid, csv_str);
      endfor %% i_hit
      %% repeat above for miss_list
      pvp_num_miss = length(CSV_struct{i_frame}.miss_list);
      if isempty(CSV_struct{i_frame}.miss_list) continue; endif
      pvp_tot_miss = pvp_tot_miss + pvp_num_miss;
      for i_miss = 1 : pvp_num_miss
        if isempty(CSV_struct{i_frame}.miss_list{i_miss}) continue; endif
	pvp_miss_density = [pvp_miss_density; CSV_struct{i_frame}.miss_list{i_miss}.hit_density];
	csv_str = [];
        csv_str = num2str(i_frame - 1); %%CSV_struct{i_frame}.Frame;
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.miss_list{i_miss}.BoundingBox_X1)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.miss_list{i_miss}.BoundingBox_Y1)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.miss_list{i_miss}.BoundingBox_X2)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.miss_list{i_miss}.BoundingBox_Y2)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.miss_list{i_miss}.BoundingBox_X3)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.miss_list{i_miss}.BoundingBox_Y3)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.miss_list{i_miss}.BoundingBox_X4)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.miss_list{i_miss}.BoundingBox_Y4)];
	csv_str = [csv_str, ",", CSV_ObjectType];
	csv_str = [csv_str, ",", num2str(CSV_Occlusion)];
	csv_str = [csv_str, ",", num2str(CSV_Ambiguous)];
	csv_str = [csv_str, ",", num2str(CSV_struct{i_frame}.miss_list{i_miss}.Confidence)];
	csv_str = [csv_str, ",", num2str(CSV_SiteInfo)];
	csv_str = [csv_str, ",", num2str(CSV_Version)];
	csv_str = [csv_str, "\n"];
	fputs(pvp_results_fid, csv_str);
      endfor %% i_miss
      disp("");
    endfor %% i_frame
  endfor %% i_CSV_file
  fclose(pvp_results_fid);
  
  if pvp_tot_hits == 0
    return;
  endif

  %%keyboard;
  disp(["pvp_tot_hits = ", num2str(pvp_tot_hits)]);
  disp(["pvp_tot_miss = ", num2str(pvp_tot_miss)]);
  pvp_num_hit_and_miss_bins = 100;
  if isempty(pvp_hit_density)
    return;
  endif
  pvp_min_hit_density = min(pvp_hit_density);
  pvp_max_hit_density = max(pvp_hit_density);
  disp(["pvp_min_hit_density = ", num2str(pvp_min_hit_density)]);
  disp(["pvp_max_hit_density = ", num2str(pvp_max_hit_density)]);
  pvp_ave_hit_density = sum(pvp_hit_density) / pvp_tot_hits;
  pvp_std_hit_density = sqrt(sum(pvp_hit_density.^2) / pvp_tot_hits);
  pvp_median_hit_density = median(pvp_hit_density);
  disp(["pvp_ave_hit_density = ", num2str(pvp_ave_hit_density)]);
  disp(["pvp_std_hit_density = ", num2str(pvp_std_hit_density)]);
  disp(["pvp_median_hit_density = ", num2str(pvp_median_hit_density)]);
  if isempty(pvp_miss_density)
    return;
  endif
  pvp_min_miss_density = min(pvp_miss_density);
  pvp_max_miss_density = max(pvp_miss_density);
  disp(["pvp_min_miss_density = ", num2str(pvp_min_miss_density)]);
  disp(["pvp_max_miss_density = ", num2str(pvp_max_miss_density)]);
  pvp_ave_miss_density = sum(pvp_miss_density) / pvp_tot_miss;
  pvp_std_miss_density = sqrt(sum(pvp_miss_density.^2) / pvp_tot_miss);
  pvp_median_miss_density = median(pvp_miss_density);
  disp(["pvp_ave_miss_density = ", num2str(pvp_ave_miss_density)]);
  disp(["pvp_std_miss_density = ", num2str(pvp_std_miss_density)]);
  disp(["pvp_median_miss_density = ", num2str(pvp_median_miss_density)]);
  pvp_hit_and_miss_stats_pathname = [ROC_subdir, "hit_and_miss_stats.txt"];
  hist_plot = 0;
  if hist_plot
    pvp_hit_and_miss_hist_pathname = [ROC_subdir, "hit_and_miss_hist.png"];
    [pvp_hit_and_miss_hist, pvp_hit_and_miss_bins] = ...
	hist([pvp_miss_density; pvp_hit_density], pvp_num_hit_and_miss_bins);
    pvp_hit_hist = hist(pvp_hit_density, pvp_hit_and_miss_bins);
    pvp_miss_hist = hist(pvp_miss_density, pvp_hit_and_miss_bins);
    pvp_hit_and_miss_hist_fig = figure;
    pvp_hit_and_miss_bh = ...
	bar(pvp_hit_and_miss_bins(2:pvp_num_hit_and_miss_bins), ...
	    pvp_hit_hist(2:pvp_num_hit_and_miss_bins), 0.8);
    set( pvp_hit_and_miss_bh, 'EdgeColor', [1 0 0] );
    set( pvp_hit_and_miss_bh, 'FaceColor', [1 0 0] );
    hold on;
    pvp_hit_and_miss_bh = ...
	bar(pvp_hit_and_miss_bins(2:pvp_num_hit_and_miss_bins), ...
	    pvp_miss_hist(2:pvp_num_hit_and_miss_bins), 0.6);
    set( pvp_hit_and_miss_bh, 'EdgeColor', [0 0 1] );
    set( pvp_hit_and_miss_bh, 'FaceColor', [0 0 1] );
    print(pvp_hit_and_miss_hist_fig, pvp_hit_and_miss_hist_pathname);
  endif
  if pvp_training_flag
    save("-ascii", ...
	 pvp_hit_and_miss_stats_pathname, ...
	 "pvp_tot_hits", ...
	 "pvp_tot_miss", ...
	 "pvp_min_hit_density", ...
	 "pvp_max_hit_density", ...
	 "pvp_ave_hit_density", ...
	 "pvp_std_hit_density", ...
	 "pvp_median_hit_density", ...
	 "pvp_min_miss_density", ...
	 "pvp_max_miss_density", ...
	 "pvp_ave_miss_density", ...
	 "pvp_std_miss_density", ...
	 "pvp_median_miss_density");
  endif
  disp("");
  %%close all;
  
  
  
  if pvp_training_flag == 1
    pvp_BB_density = zeros(nnz_frames, 2);
    pvp_num_BB_hist_bins = 100;
    for i_frame = 1 : nnz_frames
      if isempty(CSV_struct{i_frame}) continue; endif
      if CSV_struct{i_frame}.num_active == 0 continue; endif
      pvp_BB_density(i_frame, 1) = ...
	  CSV_struct{i_frame}.num_active_BB_mask / ...
	  (CSV_struct{i_frame}.num_BB_mask + (CSV_struct{i_frame}.num_BB_mask == 0));
      pvp_BB_density(i_frame, 2) = ...
	  CSV_struct{i_frame}.num_active_BB_notmask / ...
	  (CSV_struct{i_frame}.num_BB_notmask + (CSV_struct{i_frame}.num_BB_notmask == 0));
    endfor
    pvp_min_BB_density = min(pvp_BB_density);
    pvp_max_BB_density = max(pvp_BB_density);
    pvp_ave_BB_density = mean(pvp_BB_density);
    pvp_std_BB_density = std(pvp_BB_density);
    pvp_median_BB_density = median(pvp_BB_density);
    pvp_z_score = ...
	(pvp_ave_BB_density(1) - pvp_ave_BB_density(2)) / ...
	(pvp_std_BB_density(1) + pvp_std_BB_density(2));
    disp("");
    disp(["min_BB_density = ", num2str(pvp_min_BB_density)]);
    disp(["max_BB_density = ", num2str(pvp_max_BB_density)]);
    disp(["ave_BB_density = ", num2str(pvp_ave_BB_density)]);
    disp(["std_BB_density = ", num2str(pvp_std_BB_density)]);
    disp(["median_BB_density = ", num2str(pvp_median_BB_density)]);
    disp(["z_score = ", num2str(pvp_z_score)]);
    pvp_BB_hist_delta = ...
	(max(pvp_max_BB_density(:)) - min(pvp_min_BB_density(:))) / ...
	pvp_num_BB_hist_bins;
    pvp_BB_hist_edges = min(pvp_min_BB_density(:)) : pvp_BB_hist_delta : max(pvp_max_BB_density(:));
    pvp_BB_hist_centers = ...
	(pvp_BB_hist_edges(2:pvp_num_BB_hist_bins+1) + pvp_BB_hist_edges(1:pvp_num_BB_hist_bins)) / 2;
    pvp_BB_hist_centers = [pvp_BB_hist_centers, pvp_BB_hist_centers(end)+pvp_BB_hist_delta];
    pvp_BB_hist = ...
	histc(pvp_BB_density, pvp_BB_hist_edges);
    pvp_BB_hist_fig = figure;
    pvp_bh = bar(pvp_BB_hist_centers, pvp_BB_hist(:,1), 0.8);
    hold on;
    set( pvp_bh, 'EdgeColor', [1 0 0] );
    set( pvp_bh, 'FaceColor', [1 0 0] );
    pvp_bh = bar(pvp_BB_hist_centers, pvp_BB_hist(:,2), 0.6);
    set( pvp_bh, 'EdgeColor', [0 0 1] );
    set( pvp_bh, 'FaceColor', [0 0 1] );
    pvp_BB_hist_pathname = [ROC_subdir, "BB_hist.png"];
    print(pvp_BB_hist_fig, pvp_BB_hist_pathname, "-dpng");
    pvp_BB_stats_pathname = [ROC_subdir, "BB_stats.txt"];
    save("-ascii", ...
	 pvp_BB_stats_pathname, ...
	 "pvp_min_BB_density", ...
	 "pvp_max_BB_density", ...
	 "pvp_ave_BB_density", ...
	 "pvp_std_BB_density", ...
	 "pvp_median_BB_density");
    disp("");
  endif
  

  end_time = time();
  tot_time = end_time - begin_time;
  disp(["tot_time = ", num2str(tot_time)]);

  

endfunction %% pvp_makeCSVFile



