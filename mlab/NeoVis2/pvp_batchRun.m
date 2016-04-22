
global pvp_home_path
global pvp_mlab_path
pvp_local_path = pwd;
pvp_home_path = ...
    ["~", filesep];
    %%[filesep, "Users", filesep, "gkenyon", filesep];
    %%[filesep, "home", filesep, "garkenyon", filesep];
pvp_workspace_path = ...
    [pvp_home_path, "workspace-indigo", filesep];
pvp_mlab_path = ...
    [pvp_workspace_path, "PetaVision", filesep, "mlab", filesep];
pvp_NeoVis2_path = ...
    [pvp_mlab_path, "NeoVis2", filesep];

%% path to generic image processing routins
pvp_imgProc_path = [pvp_mlab_path, "imgProc", filesep];

%% path to string manipulation kernels for use with parcellfun
pvp_strKernels_path = [pvp_mlab_path, "stringKernels", filesep];

%% add all paths at once due to bug in octave 3.2.4
pvp_path_list = [pvp_strKernels_path pathsep() pvp_imgProc_path pathsep() pvp_NeoVis2_path pathsep()];
pvp_addpath(pvp_path_list);

global PVP_VERBOSE_FLAG
PVP_VERBOSE_FLAG = 1;

pvp_num_procs = 1;
pvp_image_type = ".png";
global pvp_image_type

global pvp_image_margin
pvp_image_margin = 8;

pvp_object_list = cell(12,1);
pvp_object_list{1} = "Boat"; 
pvp_object_list{2} = "Bus"; 
pvp_object_list{3} = "Car"; 
pvp_object_list{4} = "Container"; 
pvp_object_list{5} = "Cyclist"; 
pvp_object_list{6} = "Helicopter"; 
pvp_object_list{7} = "Person"; 
pvp_object_list{8} = "Plane"; 
pvp_object_list{9} = "Tractor-Trailer"; 
pvp_object_list{10} = "Truck"; 
pvp_object_list{11} = "distractor";  %% non-DARPA object
pvp_object_list{12} = "target"; %% any DARPA object  

NEOVISION_DATASET_ID = "Heli"; %% Tower; %% Tailwind; %%
neovision_dataset_id = tolower(NEOVISION_DATASET_ID);

NEOVISION_DISTRIBUTION_ID = "Training"; %%"Formative"; %%  "Challenge"; %%
neovision_distribution_id = tolower(NEOVISION_DISTRIBUTION_ID);

pvp_repo_path = [pvp_home_path, "NeoVision2", filesep]; %% [filesep, "mnt", filesep, "data1", filesep, "repo", filesep];
pvp_program_path = [pvp_repo_path, "neovision-programs-petavision", filesep];

neovision_dataset_path = [pvp_program_path, NEOVISION_DATASET_ID, filesep];
mkdir(neovision_dataset_path);

neovision_distribution_path = [neovision_dataset_path, NEOVISION_DISTRIBUTION_ID, filesep];
mkdir(neovision_distribution_path);


%% pvp_edge_type = 0, DoG
%% pvp_edge_type = 1, canny
pvp_edge_type = "DoG";
global pvp_DoG_flag
pvp_DoG_flag = strcmp(pvp_edge_type,"DoG");
global pvp_canny_flag
pvp_canny_flag = strcmp(pvp_edge_type,"canny");
global pvp_DoG_struct
pvp_DoG_struct = struct;  %% 
global pvp_canny_struct
pvp_canny_struct = struct;  %% 
if pvp_DoG_flag == 1
  pvp_DoG_struct.amp_center_DoG = 1;
  pvp_DoG_struct.sigma_center_DoG = 1;
  pvp_DoG_struct.amp_surround_DoG = 1;
  pvp_DoG_struct.sigma_surround_DoG = 2 * pvp_DoG_struct.sigma_center_DoG;
elseif pvp_canny_flag == 1
  pvp_canny_struct.sigma_canny = 1.0;
endif

pvp_num_ODD_kernels = 3;

pvp_skip_frames = 200;
pvp_offset_frames = 1;
pvp_shuffle_flag = 0;
pvp_noclobber_flag = 0;
pvp_rand_state = [];

pvp_params_template = ...
    [pvp_program_path, ...
     NEOVISION_DATASET_ID, filesep, ...     
     "params", filesep, "pvp_template.params"];

pvp_clip_folder = ...
    [pvp_repo_path, "neovision-data-", neovision_distribution_id, "-", neovision_dataset_id, filesep];
pvp_clip_search_str = ...
    [pvp_clip_folder, "[0-9+][0-9+][0-9+]*"];
pvp_clip_pathnames = glob(pvp_clip_search_str);
pvp_num_clips = length(pvp_clip_pathnames);
for pvp_clip_ndx = 1 : pvp_num_clips

  pvp_clip_pathname = pvp_clip_pathnames{pvp_clip_ndx};
  pvp_clip_name = strFolderFromPath(pvp_clip_pathname);
  pvp_clip_path = strExtractPath(pvp_clip_pathname);

  %% edge filter clips
  keyboard;
  [pvp_DoG_dir, ...
   pvp_canny_dir, ...
   num_frames_edge, ...
   tot_frames_edge, ...
   tot_DoG, ...
   tot_canny, ...
   tot_time_edge] = ...
      pvp_edgeFilterFrames(NEOVISION_DATASET_ID, ...
			   NEOVISION_DISTRIBUTION_ID, ...
			   pvp_repo_path, ...
			   pvp_edge_type, ...
			   pvp_clip_path, ...
			   pvp_clip_name, ...
			   pvp_program_path, ...
			   pvp_num_procs);

  %% make file of clip names
  num_frames_tmp = -1;
  [pvp_fileOfFrames, ...
   pvp_list_path, ...
   pvp_fileOfFrames_path, ...
   pvp_num_frames, ...
   pvp_tot_frames, ...
   pvp_tot_time, ...
   pvp_rand_state] = ...
      pvp_clipFileOfFilenames(NEOVISION_DATASET_ID, ...
			      NEOVISION_DISTRIBUTION_ID, ...
			      pvp_repo_path, ...
			      pvp_program_path, ...
			      pvp_edge_type, ...
			      pvp_clip_name, ...
			      pvp_num_frames, ...
			      pvp_skip_frames, ...
			      pvp_offset_frames, ...
			      pvp_shuffle_flag, ...
			      pvp_noclobber_flag, ...
			      pvp_rand_state);

  for pvp_object_ndx = 3

    %% make PetaVision params file
    [pvp_params_file, ...
     pvp_params_dir] = ...
	pvp_makeParams(NEOVISION_DATASET_ID, ...
		       NEOVISION_DISTRIBUTION_ID, ...
		       pvp_repo_path, ...
		       pvp_program_path, ...
		       pvp_params_template, ...
		       pvp_clip_name, ...
		       pvp_object_type, ...
		       pvp_num_ODD_kernels, ...
		       pvp_edge_type, ...
		       pvp_frame_size, ...
		       pvp_fileOfFrames);
    
    %% run PetaVision
    pvp_exe = ...
	[pvp_workspace_path, "Clique2", filesep, "Debug", filesep, "Clique2"];
    [pvp_status, pvp_output] = ...
	system();

	       %% make CSV files
	       
	       %% push onto repository
	       
  endfor %% pvp_object_ndx
  
endfor %% pvp_clip_ndx




