
function [fileOfFramenames, ...
	  list_dir, ...
	  num_frames, ...
	  tot_frames, ...
	  tot_time, ...
	  rand_state] = ...
      pvp_clipFileOfFilenames(NEOVISION_DATASET_ID, ...
			  NEOVISION_DISTRIBUTION_ID, ...
			  clip_dir, ...
			  clip_name, ...
			  repo_path, ...
			  num_frames, ...
			  skip_frames, ...
			  offset_frames, ...
			  shuffle_flag, ...
			  noclobber_flag, ...
			  rand_state)

  %% makes list of paths to  DARPA video clip frames for training and/or testing
  %% image frames are drawn from folder clip_dir,

  begin_time = time();

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

  num_argin = 0;
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("NEOVISION_DATASET_ID") || isempty(NEOVISION_DATASET_ID)
    NEOVISION_DATASET_ID = "Heli"; %% "Tower"; %% "Tail"; %% 
  endif
  neovision_dataset_id = tolower(NEOVISION_DATASET_ID); %% 
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("NEOVISION_DISTRIBUTION_ID") || isempty(NEOVISION_DISTRIBUTION_ID)
    NEOVISION_DISTRIBUTION_ID = "Formative"; %% "Training"; %%  
  endif
  neovision_distribution_id = tolower(NEOVISION_DISTRIBUTION_ID); %% 
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist(clip_dir) || isempty(clip_dir)
    clip_dir = "DoG";  %%  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist(object_name) || isempty(object_name)
    clip_name =  "050"; 
  endif
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("repo_path") || isempty(repo_path)
    repo_path = [filesep, "mnt", filesep, "data1", filesep, "repo", filesep];
  endif
  program_path = [repo_path, ...
		 "neovision-programs-petavision", filesep, ...
		 NEOVISION_DATASET_ID, filesep]; %% 		  
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("clip_name") || isempty(clip_name)
    clip_name = "050";
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("num_frames") || isempty(num_frames)
    num_frames = -1;  %% -1 use all images in clip_dir
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("skip_frames") || isempty(skip_frames)
    skip_frames = 1; %% 4; %% 1;  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("offset_frames") || isempty(offset_frames)
    offset_frames = 1; %% 1;  
  endif
  %% 0 -> FIFO ordering, %%
  %% 1 -> random sampling, 
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("shuffle_flag") || isempty(shuffle_flag)
    shuffle_flag = 0; %% 1;  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("noclobber_flag") || isempty(noclobber_flag)
    noclobber_flag = 0; %% 1;  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("rand_state") || isempty(rand_state)
    rand_state = rand("state");
  endif
  rand("state", rand_state);

 
  %%setenv('GNUTERM', 'x11');

  local_dir = pwd;
  image_type = "png";

  list_path = [program_path, "list", filesep];
  mkdir(list_path);
  filenames_path = [list_path, clip_name, filesep];
  mkdir(filenames_path);

  %% path to generic image processing routins
  img_proc_dir = [pvp_mlab_path, filesep, "imgProc", filesep];
  addpath(img_proc_dir);
  
  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = [pvp_mlab_path, filesep, "stringKernels", filesep];
  addpath(str_kernel_dir);

  frames_folder = [program_path, clip_dir, filesep];
  frames_path = [frames_folder, clip_name, filesep];

  tot_frames = 0;

  frames_search_str = ...
      [frames_path, '*.', image_type];
  frame_filenames = glob(frames_search_str);
  %% add sort here?
  frame_names = ...
      cellfun(@strFolderFromPath, frame_filenames, "UniformOutput", false);

  num_frames = size(frame_names,1);   
  disp(['num_frames = ', num2str(num_frames)]);
  
  tot_frames = length(offset_frames:skip_frames:num_frames);
  disp(['tot_frames = ', num2str(tot_frames)]);

  if num_frames < 0
    num_frames = tot_frames;
  endif

  if shuffle_flag
    [rank_ndx, write_train_ndx] = sort(rand(num_frames,1));
    write_train_ndx = write_train_ndx(offset_frames:skip_frames:num_frames);
  else
    write_train_ndx = offset_frames:skip_frames:num_frames;
  endif

  if num_frames < tot_frames
    write_train_ndx = write_train_ndx(1:num_frames);
  elseif num_frames > tot_frames
    num_frames = tot_frames;
  endif

  if noclobber_flag
    num_fileOfFramenames = ...
	length(glob([filenames_path, "clip_", clip_name, "_fileOfFramenames", "[0-9+].txt"]));
    fileOfFramenames = ...
	[filenames_path, "clip_", clip_name, "_fileOfFramenames", num2str(num_fileOfFramenames+1), ".txt"];
  else
    fileOfFramenames = ...
	[filenames_path, "clip_", clip_name, "_fileOfFramenames", ".txt"];
  endif
  disp(["fileOfFramenames = ", fileOfFramenames]);
  fid_train = fopen(fileOfFramenames, "w", "native");
  for i_file = 1 : num_frames
    fprintf(fid_train, "%s\n", train_filenames{write_train_ndx(i_file)});
  endfor %%
  fclose(fid_train);
  
  if shuffle_flag
    num_rand_state = length(glob([filenames_path, "rand_state", "[0-9+].mat"]));
    rand_state_filename = [filenames_path, "rand_state", num2str(num_rand_state+1), ".mat"];
    disp(["rand_state_filename = ", rand_state_filename]);
    save("-binary", rand_state_filename, "rand_state");
  endif

  end_time = time;
  tot_time = end_time - begin_time;

 endfunction %% clipFileOfFilenames