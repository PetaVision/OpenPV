
function [pvp_DoG_dir, ...
	  canny_dir, ...
	  num_frames, ...
	  tot_frames, ...
	  tot_DoG, ...
	  tot_canny, ...
	  tot_time] = ...
  pvp_edgeFilterFrames(NEOVISION_DATASET_ID, ...
		       NEOVISION_DISTRIBUTION_ID, ...
		       pvp_repo_path, ...
		       pvp_clip_path, ...
		       pvp_clip_name, ...
		       pvp_program_path, ...
		       pvp_DoG_flag, ...
		       pvp_DoG_struct, ...
		       canny_flag, ...
		       canny_struct, ...
		       pvp_num_procs)
  
  %% perform edge filtering on DARPA NeoVis2 video clips, 
  %% mirror BCs used to pad individual frames before edge extraction.
  %% resize image frames if pad_size ~= image_size

  global pvp_DoG_flag
  global canny_flag
  global pvp_pvp_DoG_dir
  global pvp_DoG_struct
  global canny_dir
  global canny_struct
  global pvp_image_margin

  global pvp_home_path
  global pvp_mlab_path
  if isempty(pvp_home_path)
    pvp_home_path = ...
	[filesep, "home", filesep, "garkenyon", filesep];
  endif
  if isempty(pvp_mlab_path)
    pvp_mlab_path = ...
	[pvp_home_path, "workspace-indigo", filesep, "PetaVision", filesep, "mlab", filesep];
  endif

  global PVP_VERBOSE_FLAG
  if ~exist("PVP_VERBOSE_FLAG") || isempty(PVP_VERBOSE_FLAG)
    PVP_VERBOSE_FLAG = 0;
  endif
 
  more off;
  begin_time = time();

  num_input_args = 0
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("NEOVISION_DATASET_ID") || isempty(NEOVISION_DATASET_ID)
    NEOVISION_DATASET_ID = "Heli"; %% "Tower"; %% "Tail"; %% 
  endif
  neovision_dataset_id = tolower(NEOVISION_DATASET_ID); %% 
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("NEOVISION_DISTRIBUTION_ID") || isempty(NEOVISION_DISTRIBUTION_ID)
    NEOVISION_DISTRIBUTION_ID = "Formative"; %% "Training"; %%  "Challenge"; %%
  endif
  neovision_distribution_id = tolower(NEOVISION_DISTRIBUTION_ID); %% 
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_repo_path") || isempty(pvp_repo_path)
    pvp_repo_path = [filesep, "mnt", filesep, "datasets", filesep, "NeoVision2", filesep, "repo", filesep];
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_clip_path") || isempty(pvp_clip_path)
    pvp_clip_path = ...
	[pvp_repo_path, "neovision-data", neovision_distribution_id, "-", neovision_dataset_id, filesep]; 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_clip_name") || isempty(pvp_clip_name)
    pvp_clip_name =  "050"; %%
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_program_path") || isempty(pvp_program_path)
    pvp_program_path = ...
	[pvp_repo_path, "neovision-programs-", neovision_distribution_id, "-", neovision_dataset_id, filesep]; 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_DoG_flag") || isempty(pvp_DoG_flag)
    pvp_DoG_flag = 1;  %% 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_DoG_struct") || isempty(pvp_DoG_struct)
    pvp_DoG_struct = struct;  %% 
    pvp_DoG_struct.amp_center_DoG = 1;
    pvp_DoG_struct.sigma_center_DoG = 1;
    pvp_DoG_struct.amp_surround_DoG = 1;
    pvp_DoG_struct.sigma_surround_DoG = 2 * pvp_DoG_struct.sigma_center_DoG;
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("canny_flag") || isempty(canny_flag)
    canny_flag = 0;  %% 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("canny_struct") || isempty(canny_struct)
    canny_struct = struct;  %% 
    canny_struct.sigma_canny = 1;
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_num_procs") || isempty(pvp_num_procs)
    pvp_num_procs = 16;  %% 
  endif
  
  setenv('GNUTERM', 'x11');

  local_dir = pwd;

  num_frames = -1;
  tot_frames = -1;
  tot_DoG = -1;
  tot_canny = -1;
  
  %% path to generic image processing routins
  img_proc_dir = [pvp_mlab_path, filesep, "imgProc", filesep];
  addpath(img_proc_dir);
  
  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = [pvp_mlab_path, filesep, "stringKernels", filesep];
  addpath(str_kernel_dir);

  if ~exist(clip_dir, "dir")
    error(["~exist(clip_dir): ", clip_dir]);
  endif
  frame_dir = ...
      [clip_dir, pvp_clip_name, filesep];  %%
  if ~exist(frame_dir, "dir")
    error(["~exist(frame_dir): ", frame_dir]);
  endif

  log_path = [pvp_program_path, "log", filesep];
  mkdir(log_path);
  log_dir = [log_path, pvp_clip_name, filesep];
  mkdir(log_dir);

  list_path = [pvp_program_path, "list", filesep];
  mkdir(list_path);
  list_dir = [list_path, pvp_clip_name, filesep];
  mkdir(list_dir);

  if pvp_DoG_flag
    DoG_folder = [pvp_program_path, "DoG", filesep];
    mkdir(DoG_folder);
    pvp_DoG_dir = [DoG_folder, pvp_clip_name, filesep];
    mkdir(pvp_DoG_dir);
  else
    pvp_DoG_dir = [];
  endif %% pvp_DoG_flag
  if canny_flag
    canny_folder = [pvp_program_path, "canny", filesep];
    mkdir(canny_folder);
    canny_dir = [canny_folder, pvp_clip_name, filesep];
    mkdir(canny_dir);
  else
    canny_dir = [];
  endif %% canny_flag

  image_type = ".png";
  pvp_image_margin = 8;

  frame_path = ...
      [frame_dir, '*', image_type];
  frame_pathnames = glob(frame_path);
  num_frames = size(frame_pathnames,1);
  disp(['num_frames = ', num2str(num_frames)]);
    
  %%keyboard;
  if pvp_num_procs > 1
    [status_info] = ...
	parcellfun(pvp_num_procs, @pvp_edgeFilterFramesKernel, frame_pathnames, "UniformOutput", false);
  else
    [status_info] = ...
	cellfun(@pvp_edgeFilterFramesKernel, frame_pathnames, "UniformOutput", false);
  endif

  tot_rejected = 0;
  tot_mean = 0;
  tot_std = 0;
  for i_frame = 1 : num_frames
    tot_rejected = tot_rejected + status_info{i_frame}.rejected_flag;
    if status_info{i_frame}.rejected_flag
      continue;
    endif
    tot_DoG = tot_DoG + status_info{i_frame}.pvp_DoG_flag;
    tot_canny = tot_canny + status_info{i_frame}.canny_flag;
    tot_mean = tot_mean + status_info{i_frame}.mean;
    tot_std = tot_std + status_info{i_frame}.std;
  endfor %% i_frame

  tot_frames = num_frames - tot_rejected;

   
  ave_mean = tot_mean / tot_frames;
  disp(["ave_mean = ", ...
	num2str(ave_mean)]);
  ave_std = tot_std / tot_frames;
  disp(["ave_std = ", ...
	num2str(ave_std)]);
  
  
  disp(["tot_frames = ", ...
	num2str(tot_frames)]);
  disp(["tot_rejected = ", ...
	num2str(tot_rejected)]);
  disp(["tot_DoG = ", ...
	num2str(tot_DoG)]);
  disp(["tot_canny = ", ...
	num2str(tot_canny)]);
  
  end_time = time();
  tot_time = end_time - begin_time;
  disp(["tot_time = ", num2str(tot_time)]);
  
  log_filename = [log_dir, "edgeFilterFrameslog.txt"];
  save("-ascii", ...
       log_filename, ...
       "tot_rejected", ...
       "tot_frames", ...
       "tot_DoG", ...
       "tot_canny", ...
       "ave_mean", ...
       "ave_std", ...
       "tot_time"); 
  
  
endfunction %% pvp_edgeFilterFrames
