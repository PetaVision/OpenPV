%% begin definition of most variable input params
<<<<<<< HEAD
clip_ids = [1:50]; %% [7:17,21:22,30:31]; %%
clip_name = cell(length(clip_ids),1);
for i_clip = 1 : length(clip_name)                                                                                         
  clip_name{i_clip} = num2str(clip_ids(i_clip), "%3.3i");
endfor
num_ODD_kernels = 0; %% 5; %% 
pvp_layer = 3;  %% 8; %%  
pvp_path_flag = true; %% false; %% 
NEOVISION_DISTRIBUTION_ID = "Training"; %%"Challenge"; %% "Formative"; %%   
ObjectType = "Car"; %% "Cyclist"; %%  
global make_bootstrap_chips_flag 
make_bootstrap_chips_flag = false; %% true; %% 
global make_target_mask_flag 
make_target_mask_flag = true; %% false; %%  
global miss_list_flag;
miss_list_flag = false;
num_procs = 8; %% 24;  %% 
%% end most variable portion of input params

home_path = ...
    [filesep, "home", filesep, "gkenyon", filesep];
NEOVISION_DATASET_ID = "Heli"; %% "Tower"; %%  "Tail"; %% 
neovision_dataset_id = tolower(NEOVISION_DATASET_ID); %% 
neovision_distribution_id = tolower(NEOVISION_DISTRIBUTION_ID); %% 
repo_path = [filesep, "nh", filesep, "compneuro", filesep, "Data", filesep, "repo", filesep];
=======
clip_ids = [26:26]; %% [7:17,21:22,30:31]; %%
clip_name = cell(length(clip_ids),1);
for i_clip = 1 : length(clip_name)                                                                                         
  clip_name{i_clip} = num2str(clip_ids(i_clip), "%3.3i");
endfor
num_ODD_kernels = 0; %% 5; %% 
pvp_layer = 3;  %% 8; %%  
pvp_path_flag = true; %% false; %% 
NEOVISION_DISTRIBUTION_ID = "Challenge"; %% "Formative"; %% "Training"; %%  
ObjectType = "Car"; %% "Cyclist"; %%  
global make_bootstrap_chips_flag 
make_bootstrap_chips_flag = false; %% true; %% 
global make_target_mask_flag 
make_target_mask_flag = false; %% true; %% 
global miss_list_flag;
miss_list_flag = false;
num_procs = 8; %% 24;  %% 
%% end most variable portion of input params

home_path = ...
    [filesep, "home", filesep, "gkenyon", filesep];
NEOVISION_DATASET_ID = "Heli"; %% "Tower"; %%  "Tail"; %% 
neovision_dataset_id = tolower(NEOVISION_DATASET_ID); %% 
neovision_distribution_id = tolower(NEOVISION_DISTRIBUTION_ID); %% 
repo_path = [filesep, "mnt", filesep, "data", filesep, "repo", filesep];
>>>>>>> refs/remotes/eclipse_auto/master
program_path = [repo_path, ...
		"neovision-programs-petavision", filesep, ...
		NEOVISION_DATASET_ID, filesep, ...
		NEOVISION_DISTRIBUTION_ID, filesep]; %% 		  
pvp_edge_filter = "canny";
pvp_frame_skip = 1; %% 1000;
pvp_frame_offset = 1; %% 160;
num_ODD_kernels_str = "";
if num_ODD_kernels > 1
  num_ODD_kernels_str = num2str(num_ODD_kernels);
elseif num_ODD_kernels == 0
  num_ODD_kernels_str = "0";
endif
pvp_bootstrap_str = ""; %% "_bootstrap"; %%  
pvp_bootstrap_level_str = ""; %% "1";
pvp_version_str = ""; %% "0"; %% 
clip_log_dir = [program_path, ...
		"log", filesep, ObjectType, filesep];
clip_log_pathname = [clip_log_dir, "log.txt"];
if exist(clip_log_pathname, "file")
  clip_log_struct = struct;
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
  patch_size = ...
      fix(clip_log_struct.ave_cropped_size); %% + clip_log_struct.std_original_size);
  std_patch_size = clip_log_struct.std_cropped_size;
  max_patch_size = clip_log_struct.max_cropped_size;
  min_patch_size = clip_log_struct.min_cropped_size;
else
  patch_size = [256, 256]; %%[128, 128];
  std_patch_size = [0, 0];
  max_patch_size = patch_size;
  min_patch_size = patch_size;
endif %% exist(clip_log_pathname)
disp(["patch_size = ", num2str(patch_size)]);
disp(["std_patch_size = ", num2str(std_patch_size)]);
training_flag = 1;

global target_mask_dir
global distractor_mask_dir
global frame_mask_dir
target_mask_dir = "";
distractor_mask_dir = "";
frame_mask_dir = "";
if make_target_mask_flag
  mask_dir = ...
      [program_path, "mask", filesep]; 
  mkdir(mask_dir);
  target_mask_dir = ...
      [mask_dir, ObjectType, filesep];
  mkdir(target_mask_dir);
  distractor_mask_dir = ...
      [mask_dir, ObjectType, "_", "distractor", filesep];
  mkdir(distractor_mask_dir);
  frame_mask_dir = ...
      [program_path, pvp_edge_filter, filesep, "mask", filesep];
  mkdir(frame_mask_dir);
endif

canny_flag = false;
for i_clip = 1 : length(clip_name)
  disp(clip_name{i_clip});
  
  if pvp_path_flag == false
    pvp_path = [];
  else 
    pvp_path = ...
	[program_path, "activity", filesep, ObjectType, num_ODD_kernels_str, ...
	 pvp_bootstrap_str, filesep, pvp_edge_filter, filesep, ...
	 clip_name{i_clip}, pvp_version_str, filesep];
  endif

  %% check if chips should be drawn from original images or from canny filtered clip
  if canny_flag == false
    clip_path = [repo_path, ...
		 "neovision-data-", ...
		 neovision_distribution_id, "-", neovision_dataset_id, filesep, ...
		 clip_name{i_clip}, filesep]; %% 		  
  else
    clip_path = [program_path, ...
		 pvp_edge_filter, filesep, ...
		 clip_name{i_clip}, filesep]; %% 
  endif

  [num_frames, ...
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
		       clip_name{i_clip}, ...
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
		       num_procs);
  if ~isempty(pvp_path)
    if ~isempty(pvp_bootstrap_level_str)
      csv_id = (2 + str2num(pvp_bootstrap_level_str));
    else
      csv_id = 2;
    endif
    csv_file = ...
	[program_path, "results", filesep, clip_name{i_clip}, filesep, ...
	 ObjectType, num2str(num_ODD_kernels), pvp_bootstrap_str, filesep, pvp_edge_filter, pvp_version_str, filesep, ...
	 NEOVISION_DATASET_ID, "_", NEOVISION_DISTRIBUTION_ID, "_",  clip_name{i_clip}, "_PetaVision_", ObjectType, "_", num2str(csv_id, "%3.3i"), ".csv"]  
    csv_repo = ...
	[repo_path, "neovision-results-", neovision_distribution_id, "-", neovision_dataset_id, filesep, clip_name{i_clip}, filesep]
    mkdir(csv_repo);
    copyfile(csv_file, csv_repo)
endif
endfor


