clear all
more off

%% clip_name gives the folder storing input images or chips
clips_flag = true; %% false; %% 
if clips_flag 
  clip_ids = [1:25]; %% [26:50]; %% 
  clip_name = cell(length(clip_ids),1);
  for i_clip = 1 : length(clip_name)
    clip_name{i_clip} = num2str(clip_ids(i_clip), "%3.3i");   
  endfor
else
  clip_name = cell(1);
  clip_name{1} = "Car"; %% "Car_bootstrap0"; %% "Plane"; %% "distractor"; %% 
endif
%% chip_path is the path to the folders referenced by clip_name 
chip_path = "/mnt/data/repo/neovision-data-formative-heli/";  %%chip_path = "/mnt/data/repo/neovision-programs-petavison/Heli/Training/canny/"; 
chip_path_append = ""; %% "8FC";
petavision_dir = ...
    ["/mnt/data/repo/neovision-programs-petavision", filesep];
dataset_dir = [petavision_dir, "Heli", filesep]; %% "noamoeba3", filesep];
flavor_dir = [dataset_dir, "Formative", filesep]; %% "3way", filesep];
mask_dir = []; %% ["/mnt/data/repo/neovision-chips-heli/Heli-PNG-Formative", clip_name{i_clip}, "_mask", filesep]; %%
pad_size = [1080 1920]; %% [256 256]; %%   
num_procs = 8;

global DoG_flag
global canny_flag
global DoG_dir
global DoG_struct
global canny_dir
global canny_struct
global image_margin
global pad_size
global cropped_dir
global rejected_dir
global border_artifact_thresh
global image_size_thresh
global VERBOSE_FLAG


%% petavision_path is the path to the results folders
mkdir(dataset_dir);
mkdir(flavor_dir);
petavision_path = flavor_dir;
mkdir(petavision_path);

DoG_flag = 0;
DoG_struct = struct;  %% 
DoG_struct.amp_center_DoG = 1;
DoG_struct.sigma_center_DoG = 1;
DoG_struct.amp_surround_DoG = 1;
DoG_struct.sigma_surround_DoG = 2 * DoG_struct.sigma_center_DoG;

canny_flag = 1;
canny_struct = struct;  %% 
canny_struct.sigma_canny = 1;

%%keyboard;
for i_clip = 1 : length(clip_name)
  disp(["clip_name = ", clip_name{i_clip}]);

  [tot_chips, ...
   tot_DoG, ...
   tot_canny, ...
   tot_cropped, ...
   tot_mean, ...
   tot_std, ...
   tot_border_artifact_top, ...
   tot_border_artifact_bottom, ...
   tot_border_artifact_left, ...
   tot_border_artifact_right, ...
   ave_original_size, ...
   ave_cropped_size, ...
   std_original_size, ...
   std_cropped_size, ...
   max_cropped_size, ...
   min_cropped_size, ...
   tot_time] = ...
      padChips(chip_path, ...
	       clip_name{i_clip}, ...
	       petavision_path, ...
	       DoG_flag, ...
	       DoG_struct, ...
	       canny_flag, ...
	       canny_struct, ...
	       pad_size, ...
	       num_procs);

endfor

num_versions = 16; %% 128; %%
num_train = repmat(-1,num_versions,1);
skip_train_images = num_versions;
begin_train_images = 1;
shuffle_flag = 0;
for i_clip = 1 : length(clip_name)
  train_path = [petavision_path, "canny", filesep, clip_name{i_clip}, filesep];
%%  train_path = [chip_path, chip_path_append, filesep, clip_name{i_clip}, filesep]; %% [petavision_path, "canny", filesep, clip_name{i_clip}, filesep]; %% 
  list_dir2 = [petavision_path, "list_canny", filesep];
  mkdir(list_dir2);
  if ~isempty(chip_path_append)
    list_dir3 = [list_dir2, chip_path_append, filesep]; 
    mkdir(list_dir3);
  else
    list_dir3 = list_dir2;
  endif
  list_dir = [list_dir3, clip_name{i_clip}, filesep]; 
%%  mkdir(list_dir);
%%  mask_dir = [chip_path, chip_path_append, filesep, "a", filesep]; %% [chip_path, clip_name{i_clip}, "_mask", filesep];
  chipFileOfFilenames2(train_path, ...
		       chip_path_append, ...
		       num_train, ...
		       skip_train_images, ...
		       begin_train_images, ...
		       clip_name{i_clip}, ...
		       list_dir, ...
		       mask_dir, ...
		       shuffle_flag, ...
		       []);

endfor



