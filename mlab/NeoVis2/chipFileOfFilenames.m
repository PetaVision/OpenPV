
function [train_filenames, ...
	  tot_train_images, ...
	  tot_time, ...
	  rand_state] = ...
      chipFileOfFilenames(imageNet_path, ...
			  object_name, ...
			  num_train, ...
			  train_dir, ...
			  list_dir, ...
			  shuffle_flag, ...
			  rand_state)

  %% makes list of paths to  DARPA chip images for training,
  %% training files are drawn from images folder train_dir,

  begin_time = time();

  if nargin < 1 || ~exist(imageNet_path) || isempty(imageNet_path)
    chip_path = "~/Pictures/HellChips/";
  endif
  if nargin < 2 || ~exist(object_name) || isempty(object_name)
    object_name = "Person"; %% "Cyclist"; %% "Plane"; %% "Boat"; %% "Container"; %% "Helicopter"; %% "Car";  %%  
  endif
  if nargin < 3 || ~exist("num_train") || isempty(num_train)
    num_train = -1;  %% -1 use all images in train_dir
  endif
  if nargin < 4 || ~exist(train_dir) || isempty(train_dir)
    train_dir = "DoG";  %%  
  endif
  if nargin < 5 || ~exist("list_dir") || isempty(list_dir)
    list_dir = "list";  %% 
  endif
  if nargin < 6 || ~exist("shuffle_flag") || isempty(shuffle_flag)
    shuffle_flag = 1;  %% 
  endif
  if nargin < 7 || ~exist("rand_state") || isempty(rand_state)
    rand_state = rand("state");
  endif
  rand("state", rand_state);

 
  setenv('GNUTERM', 'x11');

  local_dir = pwd;
  image_type = "png";

  train_filenames = {};
  list_path = [chip_path, list_dir, filesep];
  mkdir(list_path);
  filenames_path = [list_path, object_name, filesep];
  mkdir(filenames_path);

  %% path to generic image processing routins
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);


  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);

  train_folder = [chip_path, train_dir, filesep];
  train_path = [train_folder, object_name, filesep];

  tot_train_images = 0;

  train_search_str = ...
      [train_path, '*.', image_type];
  train_filenames = glob(train_search_str);
  train_names = ...
      cellfun(@strFolderFromPath, train_filenames, "UniformOutput", false);
  num_train_images = size(train_names,1);   
  tot_train_images = tot_train_images + num_train_images;
  disp(['num_train_images = ', num2str(num_train_images)]);
   

  tot_train_files = length(train_filenames);
  disp(["tot_train_files = ", num2str(tot_train_files)]);
  if tot_train_files ~= tot_train_images
    error("tot_train_files ~= tot_train_images");
  endif

  if num_train < 0
    num_train = tot_train_files;
  endif

  if num_train < tot_train_files || shuffle_flag
    [rank_ndx, write_train_ndx] = sort(rand(tot_train_images,1));
  else
    write_train_ndx = 1:num_train;
  endif

  num_fileOfFilenames_train = length(glob([filenames_path, "train_fileOfFilenames", "[0-9+].txt"]));
  fileOfFilenames_train = [filenames_path, "train_fileOfFilenames", num2str(num_fileOfFilenames_train+1), ".txt"];
  disp(["fileOfFilenames_train = ", fileOfFilenames_train]);
  fid_train = fopen(fileOfFilenames_train, "w", "native");
  for i_file = 1 : num_train
    fprintf(fid_train, "%s\n", train_filenames{write_train_ndx(i_file)});
  endfor %%
  fclose(fid_train);
  
  num_rand_state = length(glob([filenames_path, "rand_state", "[0-9+].mat"]));
  rand_state_filename = [filenames_path, "rand_state", num2str(num_rand_state+1), ".mat"];
  disp(["rand_state_filename = ", rand_state_filename]);
  save("-binary", rand_state_filename, "rand_state");

  end_time = time;
  tot_time = end_time - begin_time;

 endfunction %% chipFileOfFilenames