
function [pvp_fileOfFrames, ...
	  pvp_list_path, ...
	  pvp_fileOfFrames_path, ...
	  pvp_num_frames, ...
	  pvp_tot_frames, ...
	  pvp_tot_time, ...
	  rand_state] = ...
      pvp_clipFileOfFilenames(NEOVISION_DATASET_ID, ...
			      NEOVISION_DISTRIBUTION_ID, ...
			      pvp_repo_path, ...
			      pvp_program_path, ...
			      pvp_edge_type, ...
			      pvp_clip_name, ...
			      pvp_num_frames, ...
			      pvp_skip_frames, ...
			      pvp_offset_frames, ...
			      shuffle_flag, ...
			      noclobber_flag, ...
			      rand_state)

  %% makes list of paths to  DARPA video clip frames for training and/or testing
  %% image frames are drawn from folder pvp_clip_dir,

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
  num_input_args = num_input_args + 1;
  if nargin < num_input_args || ~exist("pvp_repo_path") || isempty(pvp_repo_path)
    pvp_repo_path = [filesep, "mnt", filesep, "datasets", filesep, "repo", filesep];
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist(pvp_program_path) || isempty(pvp_program_path)
    program_path = [pvp_repo_path, ...
		    "neovision-programs-petavision", filesep, ...
		    NEOVISION_DATASET_ID, filesep]; %% 	
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist(pvp_edge_type) || isempty(pvp_edge_type)
    pvp_edge_type = "DoG";  %%  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist(pvp_clip_name) || isempty(pvp_clip_name)
    pvp_clip_name =  "050"; 
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_num_frames") || isempty(pvp_num_frames)
    pvp_num_frames = -1;  %% -1 use all images in pvp_clip_dir
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_skip_frames") || isempty(pvp_skip_frames)
    pvp_skip_frames = 1; %% 4; %% 1;  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_offset_frames") || isempty(pvp_offset_frames)
    pvp_offset_frames = 1; %% 1;  
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
  pvp_fileOfFrames_path = ...
      [list_path, NEOVISION_DATASET_ID, filesep, NEOVISION_DISTRIBUTION_ID, filesep, ...
       pvp_clip_name, filesep, pvp_edge_type, filesep];
  mkdir(pvp_fileOfFrames_path);

  %% path to generic image processing routins
  img_proc_dir = [pvp_mlab_path, filesep, "imgProc", filesep];
  addpath(img_proc_dir);
  
  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = [pvp_mlab_path, filesep, "stringKernels", filesep];
  addpath(str_kernel_dir);

  frames_folder = [program_path, pvp_clip_dir, filesep];
  frames_path = [frames_folder, pvp_clip_name, filesep];

  pvp_tot_frames = 0;

  frames_search_str = ...
      [frames_path, '*.', image_type];
  frame_filenames = glob(frames_search_str);
  %% add sort here?
  frame_names = ...
      cellfun(@strFolderFromPath, frame_filenames, "UniformOutput", false);

  pvp_num_frames = size(frame_names,1);   
  disp(['pvp_num_frames = ', num2str(pvp_num_frames)]);
  
  pvp_tot_frames = length(pvp_offset_frames:pvp_skip_frames:pvp_num_frames);
  disp(['pvp_tot_frames = ', num2str(pvp_tot_frames)]);

  if pvp_num_frames < 0
    pvp_num_frames = pvp_tot_frames;
  endif

  if shuffle_flag
    [rank_ndx, write_train_ndx] = sort(rand(pvp_num_frames,1));
    write_train_ndx = write_train_ndx(pvp_offset_frames:pvp_skip_frames:pvp_num_frames);
  else
    write_train_ndx = pvp_offset_frames:pvp_skip_frames:pvp_num_frames;
  endif

  if pvp_num_frames < pvp_tot_frames
    write_train_ndx = write_train_ndx(1:pvp_num_frames);
  elseif pvp_num_frames > pvp_tot_frames
    pvp_num_frames = pvp_tot_frames;
  endif

  pvp_fileOfFrames_prefix = [NEOVISION_DATASET_ID, "_", NEOVISION_DISTRIBUTION_ID, "_", pvp_clip_name, "_", pvp_edge_type, "_", "frames"];
  if noclobber_flag
    num_fileOfFrames = ...
	length(glob([pvp_fileOfFrames_path, pvp_fileOfFrames_prefix, "[0-9+].txt"]));
    pvp_fileOfFrames = ...
	[pvp_fileOfFrames_path, pvp_fileOfFrames_prefix, num2str(num_fileOfFrames+1), ".txt"];
  else
    pvp_fileOfFrames = ...
	[pvp_fileOfFrames_path, pvp_fileOfFrames_prefix, ".txt"];
  endif
  disp(["pvp_fileOfFrames = ", pvp_fileOfFrames]);
  fid_train = fopen(pvp_fileOfFrames, "w", "native");
  for i_file = 1 : pvp_num_frames
    fprintf(fid_train, "%s\n", train_filenames{write_train_ndx(i_file)});
  endfor %%
  fclose(fid_train);
  
  if shuffle_flag
    num_rand_state = length(glob([pvp_fileOfFrames_path, "rand_state", "[0-9+].mat"]));
    rand_state_filename = [pvp_fileOfFrames_path, "rand_state", num2str(num_rand_state+1), ".mat"];
    disp(["rand_state_filename = ", rand_state_filename]);
    save("-binary", rand_state_filename, "rand_state");
  endif

  end_time = time;
  pvp_tot_time = end_time - begin_time;

 endfunction %% clipFileOfFilenames