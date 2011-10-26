
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
    CSV_path = [machine_path, "Pictures", filesep, "Tower", filesep, ...
		"neovision-data-formative-tower", filesep]; %% 
  endif
  if nargin < 2 || ~exist("clip_name") || isempty(clip_name)
    clip_name = "050";
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
    pvp_path = [machine_path, "workspace-indigo", filesep, "Clique2", ...
		filesep, "input", filesep, "Tower", filesep, clip_name, filesep, ...
	       target_name, filesep, "DoG", filesep];
  endif
  if nargin < 7 || ~exist("pvp_layer") || isempty(pvp_layer)
    pvp_layer = 5;  %% 
  endif
  if nargin < 8 || ~exist("num_procs") || isempty(num_procs)
    num_procs = 1;  %% 
  endif

  setenv('GNUTERM', 'x11');
  image_type = ".png";

  annotated_dir = [CSV_path, clip_name, filesep];

  %% get frame IDs 
  clip_dir = [CSV_path, clip_name, filesep];
  if ~exist(clip_dir, "dir")
    error(["~exist(clip_dir):", clip_dir]);
  endif
  frameIDs_path = ...
      [clip_dir, '*', image_type];
  frame_IDs = glob(frameIDs_path);
  num_frames = size(frame_IDs,1);
  disp(['num_frames = ', num2str(num_frames)]);

  %% read pvp activity into cell array
  [pvp_fid, ...
   pvp_header, ...
   pvp_index ] = ...
      pvp_openActivityFile(pvp_path, pvp_layer);
  [layerID] = neoVisLayerID(pvp_layer);

  global NFEATURES NCOLS NROWS N
  NCOLS = pvp_header(pvp_index.NX_GLOBAL);
  NROWS = pvp_header(pvp_index.NY_GLOBAL);
  NFEATURES = pvp_header(pvp_index.NF);
  N = NFEATURES * NCOLS * NROWS;
  
  pvp_time = zeros(num_frames, 1);
  pvp_offset = zeros(num_frames, 1);
  
  pvp_offset_tmp = 0;
  for i_frame = 1 : num_frames
    pvp_frame = i_frame + pvp_layer - 2;
    [pvp_time(i_frame),...
     pvp_activity_tmp, ...
     pvp_offset(i_frame)] = ...
	pvp_readSparseLayerActivity(pvp_fid, pvp_frame, pvp_header, pvp_index, pvp_offset_tmp);
    if pvp_offset(i_frame) == -1
      break;
    endif
    pvp_activity{i_frame,1} = pvp_activity_tmp;
    pvp_offset_tmp = pvp_offset(i_frame);
    disp(["i_frame = ", num2str(i_frame)]);
    disp(["pvp_time = ", num2str(pvp_time(i_frame))]);
    disp(["frame_ID = ", frame_IDs{i_frame}]);
    disp(["mean(pvp_activty) = ", num2str(mean(pvp_activity{i_frame}(:)))]);    
  endfor
  fclose(pvp_fid);

  tot_frames = length(pvp_activity);
  pvp_time_cell = cell(tot_frames, 1);
  frame_IDs_cell = cell(tot_frames, 1);
  %%pvp_activity_tmp = cell(tot_frames, 1);
  for i_frame = 1 : tot_frames
    frame_IDs_cell{i_frame} = frame_IDs{i_frame};
    pvp_time_cell{i_frame} = pvp_time(i_frame);
    %%pvp_activity_tmp{i_frame} = pvp_activity{i_frame};
  endfor

  if num_procs > 1
    CSV_struct = parcellfun(num_procs, @pvp_makeCSVFileKernel, ...
			    frame_IDs_cell, pvp_time_cell, pvp_activity, "UniformOutput", false);
  else
    CSV_struct = cellfun(@pvp_makeCSVFileKernel, ...
			 frame_IDs_cell, pvp_time_cell, pvp_activity, "UniformOutput", false);
  endif

  for i_frame = 1 : tot_frames
    disp(["frame_ID = ", CSV_struct{i_frame}.frame_ID]);
    disp(["pvp_time = ", num2str(CSV_struct{i_frame}.pvp_time)]);
    disp(["mean(pvp_activty) = ", num2str(CSV_struct{i_frame}.mean_activity), "\n"]);    
  endfor

  skip_plot = 21;
  for i_frame = 1 : skip_plot : tot_frames
    imagesc(CSV_struct{i_frame}.pvp_image); 
  endfor

  
  

endfunction %% pvp_makeCSVFile



