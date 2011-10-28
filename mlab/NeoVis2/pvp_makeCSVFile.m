
function [num_frames] = pvp_makeCSVFile(CSV_path, ...
					clip_name, ...
					ObjectType, ...
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
    CSV_path = [machine_path, "Pictures", filesep, "NeoVision", filesep, "Tower", filesep, ...
		"neovision-data-formative-tower", filesep]; %% 
  endif
  if nargin < 2 || ~exist("clip_name") || isempty(clip_name)
    clip_name = "050";
  endif
  if nargin < 3 || ~exist("ObjectType") || isempty(ObjectType)
    ObjectType = "Cyclist";
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
	       ObjectType, filesep, "DoG", filesep];
  endif
  if nargin < 7 || ~exist("pvp_layer") || isempty(pvp_layer)
    pvp_layer = 5;  %% 
  endif
  if nargin < 8 || ~exist("num_procs") || isempty(num_procs)
    num_procs = 4;  %% 
  endif

  setenv('GNUTERM', 'x11');
  image_type = ".png";

  %% path to generic image processing routines
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);

  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);

  annotated_path = [CSV_path, "annotated", filesep]; 
  mkdir(annotated_path);
  annotated_clip_dir = [annotated_path, clip_name, filesep];
  mkdir(annotated_clip_dir);
  annotated_dir = [annotated_clip_dir, ObjectType, filesep];
  mkdir(annotated_dir);

  true_path = [CSV_path, "truth", filesep, clip_name, filesep];
  true_CSV_filename = [clip_name, ".csv"];
  true_CSV_pathname = [true_path, true_CSV_filename];
  true_CSV_fid = fopen(true_CSV_pathname, "r");
  true_CSV_header = fgets(true_CSV_fid);
  true_CSV_list = cell(1);
  i_CSV = 0;
  while ~feof(true_CSV_fid)
    i_CSV = i_CSV + 1;
    true_CSV_list{i_CSV} = fgets(true_CSV_fid);
  endwhile
  fclose(true_CSV_fid);
  num_true_CSV = i_CSV;

  %% get frame IDs 
  clip_dir = [CSV_path, "clips", filesep, clip_name, filesep];
  if ~exist(clip_dir, "dir")
    error(["~exist(clip_dir):", clip_dir]);
  endif
  frameIDs_path = ...
      [clip_dir, '*', image_type];
  frame_pathnames_all = glob(frameIDs_path);
  num_frames = size(frame_pathnames_all,1);
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
    disp(["frame_ID = ", frame_pathnames_all{i_frame}]);
    disp(["mean(pvp_activty) = ", num2str(mean(pvp_activity{i_frame}(:)))]);    
  endfor
  fclose(pvp_fid);

  tot_frames = length(pvp_activity);
  pvp_time_cell = cell(tot_frames, 1);
  frame_pathnames = cell(tot_frames, 1);
  for i_frame = 1 : tot_frames
    frame_pathnames{i_frame} = frame_pathnames_all{i_frame};
    pvp_time_cell{i_frame} = pvp_time(i_frame);
  endfor

  %% struct for storing rank order of comma separators between fields
  true_CSV_comma_rank = struct;
  true_CSV_comma_rank.Frame = [1, 2];
  true_CSV_comma_rank.BoundingBox_X1 = [2, 3];
  true_CSV_comma_rank.BoundingBox_Y1 = [3, 4];
  true_CSV_comma_rank.BoundingBox_X2 = [4, 5];
  true_CSV_comma_rank.BoundingBox_Y2 = [5, 6];
  true_CSV_comma_rank.BoundingBox_X3 = [6, 7];
  true_CSV_comma_rank.BoundingBox_Y3 = [7, 8];
  true_CSV_comma_rank.BoundingBox_X4 = [8, 9];
  true_CSV_comma_rank.BoundingBox_Y4 = [9, 10];
  true_CSV_comma_rank.ObjectType = [10, 11];

  true_CSV_struct = cell(tot_frames, 1);
  for i_CSV = 1 : num_true_CSV
    true_CSV_comma_ndx = [1, strfind(true_CSV_list{i_CSV}, ",")];
    ObjectType_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.ObjectType(1))+1;
    ObjectType_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.ObjectType(2))-1;
    CSV_ObjectType = true_CSV_list{i_CSV}(ObjectType_ndx(1):ObjectType_ndx(2));
    if ~strcmp(CSV_ObjectType, ObjectType)
      continue;
    endif
    true_CSV_struct_tmp.ObjectType = CSV_ObjectType;
    Frame_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.Frame(1));
    Frame_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.Frame(2))-1;
    Frame = true_CSV_list{i_CSV}(Frame_ndx(1):Frame_ndx(2));
    i_frame = str2num(Frame) + 1;
    if i_frame > tot_frames
      break;
    endif
    true_CSV_struct_tmp = struct;
    true_CSV_struct_tmp.Frame = Frame;
    BoundingBox_X1_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X1(1))+1;
    BoundingBox_X1_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X1(2))-1;
    BoundingBox_X1 = true_CSV_list{i_CSV}(BoundingBox_X1_ndx(1):BoundingBox_X1_ndx(2));
    true_CSV_struct_tmp.BoundingBox_X1 = str2num(BoundingBox_X1);
    BoundingBox_Y1_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y1(1))+1;
    BoundingBox_Y1_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y1(2))-1;
    BoundingBox_Y1 = true_CSV_list{i_CSV}(BoundingBox_Y1_ndx(1):BoundingBox_Y1_ndx(2));
    true_CSV_struct_tmp.BoundingBox_Y1 = str2num(BoundingBox_Y1);
    BoundingBox_X2_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X2(1))+1;
    BoundingBox_X2_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X2(2))-1;
    BoundingBox_X2 = true_CSV_list{i_CSV}(BoundingBox_X2_ndx(1):BoundingBox_X2_ndx(2));
    true_CSV_struct_tmp.BoundingBox_X2 = str2num(BoundingBox_X2);
    BoundingBox_Y2_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y2(1))+1;
    BoundingBox_Y2_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y2(2))-1;
    BoundingBox_Y2 = true_CSV_list{i_CSV}(BoundingBox_Y2_ndx(1):BoundingBox_Y2_ndx(2));
    true_CSV_struct_tmp.BoundingBox_Y2 = str2num(BoundingBox_Y2);
    BoundingBox_X3_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X3(1))+1;
    BoundingBox_X3_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X3(2))-1;
    BoundingBox_X3 = true_CSV_list{i_CSV}(BoundingBox_X3_ndx(1):BoundingBox_X3_ndx(2));
    true_CSV_struct_tmp.BoundingBox_X3 = str2num(BoundingBox_X3);
    BoundingBox_Y3_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y3(1))+1;
    BoundingBox_Y3_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y3(2))-1;
    BoundingBox_Y3 = true_CSV_list{i_CSV}(BoundingBox_Y3_ndx(1):BoundingBox_Y3_ndx(2));
    true_CSV_struct_tmp.BoundingBox_Y3 = str2num(BoundingBox_Y3);
    BoundingBox_X4_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X4(1))+1;
    BoundingBox_X4_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_X4(2))-1;
    BoundingBox_X4 = true_CSV_list{i_CSV}(BoundingBox_X4_ndx(1):BoundingBox_X4_ndx(2));
    true_CSV_struct_tmp.BoundingBox_X4 = str2num(BoundingBox_X4);
    BoundingBox_Y4_ndx(1) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y4(1))+1;
    BoundingBox_Y4_ndx(2) = true_CSV_comma_ndx(true_CSV_comma_rank.BoundingBox_Y4(2))-1;
    BoundingBox_Y4 = true_CSV_list{i_CSV}(BoundingBox_Y4_ndx(1):BoundingBox_Y4_ndx(2));
    true_CSV_struct_tmp.BoundingBox_Y4 = str2num(BoundingBox_Y4);
    num_BBs = length(true_CSV_struct{i_frame});
    true_CSV_struct{i_frame}{num_BBs + 1} = true_CSV_struct_tmp;
 endfor

  if num_procs > 1
    CSV_struct = parcellfun(num_procs, @pvp_makeCSVFileKernel, ...
			    frame_pathnames, pvp_time_cell, pvp_activity, true_CSV_struct, ...
			    "UniformOutput", false);
  else
    CSV_struct = cellfun(@pvp_makeCSVFileKernel, ...
			 frame_pathnames, pvp_time_cell, pvp_activity, true_CSV_struct, ...
			 "UniformOutput", false);
  endif

  for i_frame = 1 : tot_frames
    CSV_struct{i_frame}.Frame = i_frame-1;
    CSV_struct{i_frame}.ObjectType = ObjectType;
    CSV_struct{i_frame}.Occlusion = 0; %% false
    CSV_struct{i_frame}.Ambiguous = 0; %% false
    CSV_struct{i_frame}.Version = 1.4;
    disp(["frame_ID = ", CSV_struct{i_frame}.frame_filename]);
    disp(["pvp_time = ", num2str(CSV_struct{i_frame}.pvp_time)]);
    disp(["mean(pvp_activty) = ", num2str(CSV_struct{i_frame}.mean_activity), "\n"]);    
  endfor

  for i_frame = 1 : tot_frames
    pvp_image_pathname = [annotated_dir, CSV_struct{i_frame}.frame_filename]
    imwrite(CSV_struct{i_frame}.pvp_image, pvp_image_pathname);
    %% imagesc(CSV_struct{i_frame}.pvp_image); 
  endfor

  
  

endfunction %% pvp_makeCSVFile



