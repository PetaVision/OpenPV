
global home_path
global mlab_path
home_path = ...
    [filesep, "home", filesep, "garkenyon", filesep];
mlab_path = ...
    [home_path, "workspace-indigo", filesep, "PetaVision", filesep, "mlab", filesep];
NeoVis2_path = [mlab_path, "NeoVis2", filesep];
addpath(NeoVis2_path);

image_type = ".png";

object_list = cell(12,1);
object_list{1} = "Boat"; 
object_list{2} = "Bus"; 
object_list{3} = "Car"; 
object_list{4} = "Container"; 
object_list{5} = "Cyclist"; 
object_list{6} = "Helicopter"; 
object_list{7} = "Person"; 
object_list{8} = "Plane"; 
object_list{9} = "Tractor-Trailer"; 
object_list{10} = "Truck"; 
object_list{11} = "distractor";  %% non-DARPA object
object_list{12} = "target"; %% any DARPA object  

NEOVISION_DATASET_ID = "Heli";
NEOVISION_DISTRIBUTION_ID = "Formative";


pvp_params_template = ...
    ["/mnt/data1/repo/neovision-programs-petavision/", ...
     NEOVISION_DATASET_ID, ...
     "/params/pvp_template.params"];

clip_parent_dir = ["/mnt/data4/NeoVision2/", NEOVISION_DATASET_ID];
clip_search_str = ...
      [clip_parent_dir, "[0-9][0-9][0-9]"];
clip_pathnames = glob(clip_search_str);
num_clips = clip_pathnames;
for i_clip = 1 : num_clips

  %% make file of clip names
  clip_path = clip_pathnames{i_clip};

  [num_frames, ...
   tot_frames, ...
   tot_time, ...
   rand_state] = ...
      clipFileOfFilenames(NEOVISION_DATASET_ID, ...
			  NEOVISION_DISTRIBUTION_ID, ...
			  clip_path, ...
			  repo_path, ...
			  num_frames, ...
			  skip_frames, ...
			  offset_frames, ...
			  clip_dir, ...
			  list_dir, ...
			  shuffle_flag, ...
			  noclobber_flag, ...
			  rand_state);
 
  %% make PetaVision params file

  %% run PetaVision

  %% make CSV files

  %% push onto repository


endfor




