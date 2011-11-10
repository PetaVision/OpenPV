
global pvp_home_path
global pvp_mlab_path
pvp_home_path = ...
    [filesep, "home", filesep, "garkenyon", filesep];
pvp_mlab_path = ...
    [pvp_home_path, "workspace-indigo", filesep, "PetaVision", filesep, "mlab", filesep];
pvp_NeoVis2_path = [pvp_mlab_path, "NeoVis2", filesep];
addpath(pvp_NeoVis2_path);

global PVP_VERBOSE_FLAG
PVP_VERBOSE_FLAG = 1;

pvp_num_procs = 1;

pvp_image_type = ".png";
global pvp_image_type

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

NEOVISION_DATASET_ID = "Heli"; %% Tower; %% Tail; %%
neovision_dataset_id = tolower(NEOVISION_DATASET_ID);

NEOVISION_DISTRIBUTION_ID = "Formative"; %% "Training"; %% "Challenge"; %%
neovision_distribution_id = tolower(NEOVISION_DISTRIBUTION_ID);

pvp_repo_path = [filesep, "mnt", filesep, "data1", filesep, "repo", filesep];
pvp_program_path = [pvp_repo_path, "neovision-programs-petavision", filesep];

%% pvp_edge_type = 0, DoG
%% pvp_edge_type = 1, canny
pvp_edge_type = 0;
pvp_DoG_flag = pvp_edge_type == 0;
if pvp_DoG_flag == 1
  pvp_DoG_struct = struct;  %% 
  pvp_DoG_struct.amp_center_DoG = 1;
  pvp_DoG_struct.sigma_center_DoG = 1;
  pvp_DoG_struct.amp_surround_DoG = 1;
  pvp_DoG_struct.sigma_surround_DoG = 2 * pvp_DoG_struct.sigma_center_DoG;
endif
canny_flag = 0;

pvp_num_ODD_kernels = 3;

pvp_skip_frames = 1;
pvp_offset_frames = 1;
pvp_shuffle_flag = 0;
pvp_noclobber_flag = 0;
pvp_rand_state = [];

pvp_params_template = ...
    [pvp_program_path, ...
     NEOVISION_DATASET_ID, filesep, ...
     "params", filesep, "pvp_template.params"];

pvp_clip_folder = ...
    [pvp_repo_path, "neovision-data-", neovision-distribution-id, "-", neovision_dataset_id];
pvp_clip_search_str = ...
      [pvp_clip_folder, "[0-9][0-9][0-9]*"];
pvp_clip_pathnames = glob(pvp_clip_search_str);
pvp_num_clips = length(pvp_clip_pathnames);
for pvp_clip_ndx = 1 : pvp_num_clips

  clip_pathname = pvp_clip_pathnames{pvp_clip_ndx};
  clip_name = strFolderFromPath(clip_pathname);
  clip_path = strExtractPath(clip_pathname);

  %% edge filter clips
  [DoG_dir, ...
   canny_dir, ...
   num_frames_edge, ...
   tot_frames_edge, ...
   tot_DoG, ...
   tot_canny, ...
   tot_time_edge] = ...
      pvp_edgeFilterFrames(clip_path, ...
			   clip_name, ...
			   pvp_program_path, ...
			   pvp_DoG_flag, ...
			   pvp_DoG_struct, ...
			   canny_flag, ...
			   canny_struct, ...
			   pvp_num_procs);

  %% make file of clip names
  num_frames_tmp = -1;
  [list_fileOfFramenames, ...
   list_dir, ...
   num_frames_fileOfFilenames, ...
   tot_frames_fileOfFilenames, ...
   tot_time_fileOfFilenames, ...
   pvp_rand_state] = ...
      pvp_clipFileOfFilenames(NEOVISION_DATASET_ID, ...
			      NEOVISION_DISTRIBUTION_ID, ...
			      DoG_dir, ...
			      clip_name, ...
			      pvp_repo_path, ...
			      num_frames_edge, ...
			      pvp_skip_frames, ...
			      pvp_offset_frames, ...
			      pvp_shuffle_flag, ...
			      pvp_noclobber_flag, ...
			      pvp_rand_state);

  for pvp_object_ndx = 3

  %% make PetaVision params file
    [pvp_params_file] = ...
	pvp_makePetaVisionParamsFile(pvp_params_template, ...
				     pvp_program_path);
    
    
    %% run PetaVision

    %% make CSV files
    
    %% push onto repository
    
  endfor %% pvp_object_ndx

endfor %% pvp_clip_ndx




