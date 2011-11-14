
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
			   pvp_edge_type, ...
			   pvp_clip_path, ...
			   pvp_clip_name, ...
			   pvp_program_path, ...
			   pvp_num_procs)
  
  %% perform edge filtering on DARPA NeoVis2 video clips, 
  %% mirror BCs used to pad individual frames before edge extraction.
  %% resize image frames if pad_size ~= image_size

  global pvp_DoG_flag
  global pvp_canny_flag
  global pvp_DoG_dir
  global pvp_DoG_struct
  global pvp_canny_dir
  global pvp_canny_struct
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

  num_argin = 0
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("NEOVISION_DATASET_ID") || isempty(NEOVISION_DATASET_ID)
    NEOVISION_DATASET_ID = "Heli"; %% "Tower"; %% "Tail"; %% 
  endif
  neovision_dataset_id = tolower(NEOVISION_DATASET_ID); %% 
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("NEOVISION_DISTRIBUTION_ID") || isempty(NEOVISION_DISTRIBUTION_ID)
    NEOVISION_DISTRIBUTION_ID = "Formative"; %% "Training"; %%  "Challenge"; %%
  endif
  neovision_distribution_id = tolower(NEOVISION_DISTRIBUTION_ID); %% 
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_repo_path") || isempty(pvp_repo_path)
    pvp_repo_path = [filesep, "mnt", filesep, "datasets", filesep, "NeoVision2", filesep, "repo", filesep];
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist(pvp_edge_type) || isempty(pvp_edge_type)
    pvp_edge_type = "DoG";  %%  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_clip_path") || isempty(pvp_clip_path)
    pvp_clip_path = ...
	[pvp_repo_path, "neovision-data", neovision_distribution_id, "-", neovision_dataset_id, filesep]; 
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_clip_name") || isempty(pvp_clip_name)
    pvp_clip_name =  "050"; %%
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_program_path") || isempty(pvp_program_path)
    pvp_program_path = ...
	[pvp_repo_path, "neovision-programs-petavision", filesep]; 
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_num_procs") || isempty(pvp_num_procs)
    pvp_num_procs = 16;  %% 
  endif
  
  %%setenv('GNUTERM', 'x11');

  local_dir = pwd;

  num_frames = -1;
  tot_frames = -1;
  tot_DoG = -1;
  tot_canny = -1;
  
  %% path to generic image processing routins
  imgProc_path = [pvp_mlab_path, "imgProc", filesep];
  if isempty(strfind(path, imgProc_path))
    addpath(imgProc_path);
  endif
  
  %% path to string manipulation kernels for use with parcellfun
  strKernels_path = [pvp_mlab_path, "stringKernels", filesep];
  if isempty(strfind(path, strKernels_path))
    addpath(strKernels_path);
  endif

  if ~exist(pvp_clip_path, "dir")
    error(["~exist(pvp_clip_path): ", pvp_clip_path]);
  endif
  frame_dir = ...
      [pvp_clip_path, pvp_clip_name, filesep];  %%
  if ~exist(frame_dir, "dir")
    error(["~exist(frame_dir): ", frame_dir]);
  endif

  log_dir = ...
      [pvp_program_path, ...
       NEOVISION_DATASET_ID, filesep, NEOVISION_DISTRIBUTION_ID, filesep, ...
       pvp_clip_name, filesep, pvp_edge_type, filesep, ...
       "log", filesep];
  mkdir(log_dir);

  if pvp_DoG_flag
    pvp_DoG_dir = ...
	[pvp_program_path, ...
	 NEOVISION_DATASET_ID, filesep, NEOVISION_DISTRIBUTION_ID, filesep, ...
	 pvp_clip_name, filesep, ...
	 "DoG", filesep];
    mkdir(pvp_DoG_dir);
  else
    pvp_DoG_dir = [];
  endif %% pvp_DoG_flag

  if pvp_canny_flag
    canny_dir = ...
	[pvp_program_path, ...
	 NEOVISION_DATASET_ID, filesep, NEOVISION_DISTRIBUTION_ID, filesep, ...
	 pvp_clip_name, filesep, ...
	 "canny", filesep];
    mkdir(pvp_canny_dir);
  else
    pvp_canny_dir = [];
  endif %% pvp_canny_flag

  frame_path = ...
      [frame_dir, '*', pvp_image_type];
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
