
function [num_frames] = pvp_makeCSVFile(CSV_path, ...
					clip_name, ...
					target_name, ...
					distractor_name, ...
					patch_size, ...
					pvp_path, ...
					pvp_layer, ...
					plot_trials_skip)
  %% takes PetaVision non-spiking activity files generated in response to
  %% a video clip and produces a CSV file indicating locations of
  %% specified object
  
  begin_time = time();

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

  machine_path = ...
      [filesep, "Users", filesep, "gkenyon", filesep];

  

  if nargin < 1 || ~exist("CSV_path") || isempty(CSV_path)
    CSV_path = [macine_path, "Pictures", filesep, "Tower", filesep, ...
		"neovision-chips-tower", filesep]; %% 
  endif
  if nargin < 2 || ~exist("clip_name") || isempty(clip_name)
    clip_name = "Tower_050";
  endif
  if nargin < 3 || ~exist("target_name") || isempty(target_name)
    target_name = "Cyclist";
  endif
  if nargin < 4 || ~exist("distractor_name") || isempty(distractor_name)
    distractor_name = "distractor";
  endif
  if nargin < 5 || ~exist("patch_size") || isempty(patch_size)
    patch_size = [256, 256];
  endif
  if nargin < 6 || ~exist("pvp_path") || isempty(pvp_path)
    pvp_path = [machine_path, "workspace_indigo", filesep, "Clique2", ...
	       filesep, "Tower", filesep, "DoG", filesep, ...
	       target_name, filesep, "test", filesep, clip_name, filesep];
  endif
  if nargin < 7 || ~exist("pvp_layer") || isempty(pvp_layer)
    pvp_layer = 7;  %% 
  endif
  if nargin < 8 || ~exist("num_procs") || isempty(num_procs)
    num_procs = 6;  %% 
  endif

  setenv('GNUTERM', 'x11');
  image_type = ".png";

  %% get frame IDs 
  clip_dir = [CSV_path, clip_name, filesep];
  if ~exist("clip_dir", "dir")
    error(["~exist(clip_dir):", clip_dir]);
  endif
  frameIDs_path = ...
      [clip_dir, '*', image_type];
  frame_IDs = glob(frameIDs_path);
  num_frames = size(frame_IDs,1);
  disp(['num_frames = ', num2str(num_frames)]);

  %% read pvp activity into cell array
  pvp_activity = cell(num_frames, 1);
  [pvp_fid, ...
   pvp_header, ...
   pvp_index ] = ...
      pvp_openActivityFile(pvp_path, pvp_layer);
  [layerID] = neoVisLayerID(max_layer) 
    
  for pvp_frame = 1 : num_frames
    [pvp_time(pvp_frame),...
     pvp_activity{pvp_frame}] = ...
	pvp_readLayerActivity(pvp_fid, pvp_frame, pvp_layer, pvp_header);
    disp(["pvp_frame = ", num2str(pvp_frame)]);
    disp(["pvp_time = ", num2str(pvp_time(pvp_frame))]);
    disp(["frame_ID = ", frame_IDs{pvp_frame}]);
    disp(["mean(pvp_activty) = ", num2str(mean(pvp_activity{pvp_frame}(:)))]);    
  endfor
  fclose(pvp_fid);

  if num_procs > 1
    CSV_struct = parcellfun(num_procs, @pvp_makeCSVFileKernel, ...
			    frame_IDs, pvp_activity, "UniformOutput", false);
  endif
  

endfunction %% pvp_makeCSVFile



