
function [DoG_dir, ...
	  canny_dir, ...
	  num_frames, ...
	  tot_frames, ...
	  tot_DoG, ...
	  tot_canny, ...
	  tot_time] = ...
      pvp_edgeFilterFrames(clip_path, ...
	       clip_name, ...
	       program_path, ...
	       DoG_flag, ...
	       DoG_struct, ...
	       canny_flag, ...
	       canny_struct, ...
	       num_procs)
  
  %% perform edge filtering on DARPA NeoVis2 video clips, 
  %% mirror BCs used to pad individual frames before edge extraction.
  %% resize image frames if pad_size ~= image_size

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
  if nargin < num_input_args || ~exist("clip_path") || isempty(clip_path)
    clip_path = ...
	["/mnt/data1/repo/neovision-training-tailwind/TAILWIND_FOUO-PNG-Training/"]; 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("clip_name") || isempty(clip_name)
    clip_name =  "050"; %%
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("program_path") || isempty(program_path)
    program_path = "/mnt/data1/repo/neovision-programs-petavision/Tail/";  %% 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("DoG_flag") || isempty(DoG_flag)
    DoG_flag = 1;  %% 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("DoG_struct") || isempty(DoG_struct)
    DoG_struct = struct;  %% 
    DoG_struct.amp_center_DoG = 1;
    DoG_struct.sigma_center_DoG = 1;
    DoG_struct.amp_surround_DoG = 1;
    DoG_struct.sigma_surround_DoG = 2 * DoG_struct.sigma_center_DoG;
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
  if nargin < num_input_args || ~exist("num_procs") || isempty(num_procs)
    num_procs = 16;  %% 
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
      [clip_dir, clip_name, filesep];  %%
  if ~exist(frame_dir, "dir")
    error(["~exist(frame_dir): ", frame_dir]);
  endif

  log_path = [program_path, "log", filesep];
  mkdir(log_path);
  log_dir = [log_path, clip_name, filesep];
  mkdir(log_dir);

  list_path = [program_path, "list", filesep];
  mkdir(list_path);
  list_dir = [list_path, clip_name, filesep];
  mkdir(list_dir);

  if DoG_flag
    DoG_folder = [program_path, "DoG", filesep];
    mkdir(DoG_folder);
    DoG_dir = [DoG_folder, clip_name, filesep];
    mkdir(DoG_dir);
  else
    DoG_dir = [];
  endif %% DoG_flag
  if canny_flag
    canny_folder = [program_path, "canny", filesep];
    mkdir(canny_folder);
    canny_dir = [canny_folder, clip_name, filesep];
    mkdir(canny_dir);
  else
    canny_dir = [];
  endif %% canny_flag

  image_type = ".png";
  image_margin = 8;

  frame_path = ...
      [frame_dir, '*', image_type];
  frame_pathnames = glob(frame_path);
  num_frames = size(frame_pathnames,1);
  disp(['num_frames = ', num2str(num_frames)]);
    
  %%keyboard;
  if num_procs > 1
    [status_info] = ...
	parcellfun(num_procs, @pvp_edgeFilterFramesKernel, frame_pathnames, "UniformOutput", false);
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
    tot_DoG = tot_DoG + status_info{i_frame}.DoG_flag;
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