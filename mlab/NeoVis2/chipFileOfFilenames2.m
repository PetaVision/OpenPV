
function [train_filenames, ...
	  tot_train_images, ...
	  tot_time, ...
	  rand_state] = ...
      chipFileOfFilenames2(chip_path, ...
			  object_name, ...
			  num_train, ...
			  skip_train_images, ...
			  begin_train_images, ...
			  clip_name, ...
			  list_dir, ...
			  shuffle_flag, ...
			  rand_state)

  %% makes list of paths to  DARPA chip images for training,
  %% training files are drawn from images folder clip_name,

  begin_time = time();

  program_dir = ...
      ["/mnt/data/PetaVision/noamoeba/33x33/"];
%%      ["/mnt/data1/PetaVision/amoeba/3way/"];

  num_argin = 0;
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("chip_path") || isempty(chip_path)
    chip_path = ["~/Pictures/amoeba/256", filesep]; 
%%    chip_path = ["/mnt/data1/repo/neovision-programs-petavision/Heli/Challenge", filesep]; 
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("object_name") || isempty(object_name)
    object_name = "8"; %% "Car"; %% "Plane"; %% "Car_bootstrap1"; %%  "distractor"; "030"; %%    
  endif
  object_name_suffix = "FC";
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("num_train") || isempty(num_train)
    num_train = repmat(-1, 16, 1);  %% -1 use all images in clip_name
    %% if num_train is a vector of length > 1, make length(num_train) separate training files with the specified number of images in each
  endif
  num_output_files = length(num_train);
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("skip_train_images") || isempty(skip_train_images)
    skip_train_images = num_output_files; %% 4; %% 1;  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("begin_train_images") || isempty(begin_train_images)
    begin_train_images = 1; %% 1;  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("clip_name") || isempty(clip_name)
    clip_name = "d"; %% "canny";  %%  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("list_dir") || isempty(list_dir)
    list_head = [program_dir, "list", filesep];
    mkdir(list_head);
    list_object_dir = [list_head, object_name, object_name_suffix, filesep];
    mkdir(list_object_dir);
    list_clip_name = [list_object_dir, clip_name, filesep];
    mkdir(list_clip_name);
    list_dir = list_clip_name; %%, num2str(2)];  %% 
  endif
  %% 0 -> FIFO ordering, %%
  %% 1 -> random sampling, 
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("shuffle_flag") || isempty(shuffle_flag)
    shuffle_flag = 0; %% 1;  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("rand_state") || isempty(rand_state)
    rand_state = rand("state");
  endif
  rand("state", rand_state);

 
  setenv('GNUTERM', 'x11');

  local_dir = pwd;
  image_type = "png";

  train_filenames = {};
  list_path = list_dir; %% [chip_path, list_dir, filesep];
  mkdir(list_path);
  filenames_path = list_path;
  mkdir(filenames_path);

  %% path to generic image processing routines
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);


  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);

  object_folder = [chip_path, object_name, object_name_suffix, filesep];
  train_path = [object_folder, clip_name, filesep];

  tot_train_images = 0;

  train_search_str = ...
      [train_path, '*.', image_type];
  train_filenames = glob(train_search_str);
  train_names = ...
      cellfun(@strFolderFromPath, train_filenames, "UniformOutput", false);

  num_train_images = size(train_names,1);   
  disp(['num_train_images = ', num2str(num_train_images)]);
  
  output_filename_root = object_name;

  num_output_files = length(num_train);
  for i_output = 1 : num_output_files

    begin_train_images2 = begin_train_images + i_output - 1;
    
    write_train_ndx = begin_train_images2:skip_train_images:(num_train_images-num_output_files+i_output);
    tot_train_images = length(write_train_ndx);
    disp(['tot_train_images = ', num2str(tot_train_images)]);
   
    if num_train(i_output) < 0 
      num_train(i_output) = floor(tot_train_images);
    endif

    if shuffle_flag
      [rank_ndx, rand_val] = sort(rand(tot_train_images,1));
      write_train_ndx = write_train_ndx(rank_ndx);
    endif

    if num_train(i_output) < tot_train_images
      write_train_ndx = write_train_ndx(1:num_train(i_output));
    elseif num_train(i_output) > tot_train_images
      num_train(i_output) = tot_train_images;
    endif

    output_filename_prefix = [object_name, object_name_suffix, "_", clip_name];
    if num_output_files > 1
      output_filename = [output_filename_prefix, "_",  num2str(i_output, "%3.3i")];
    endif
    noclobber_flag = 0; %% 
    if noclobber_flag
      num_fileOfFilenames_train = ...
	  length(glob([filenames_path, output_filename, "_fileOfFilenames", "[0-9+].txt"]));
      fileOfFilenames_train = ...
	  [filenames_path, output_filename, "_fileOfFilenames", num2str(num_fileOfFilenames_train+1), ".txt"];
    else
      fileOfFilenames_train = [filenames_path, output_filename, "_fileOfFilenames", ".txt"];
    endif
    disp(["fileOfFilenames_train = ", fileOfFilenames_train]);
    fid_train = fopen(fileOfFilenames_train, "w", "native");
    for i_file = 1 : num_train(i_output)
      fprintf(fid_train, "%s\n", train_filenames{write_train_ndx(i_file)});
    endfor %%
    fclose(fid_train);

  endfor %% i_output
  
  num_rand_state = length(glob([filenames_path, "rand_state", "[0-9+].mat"]));
  rand_state_filename = [filenames_path, "rand_state", num2str(num_rand_state+1), ".mat"];
  disp(["rand_state_filename = ", rand_state_filename]);
  save("-binary", rand_state_filename, "rand_state");

  end_time = time;
  tot_time = end_time - begin_time;

 endfunction %% chipFileOfFilenames
