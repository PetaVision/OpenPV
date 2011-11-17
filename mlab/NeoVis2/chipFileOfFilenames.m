
function [train_filenames, ...
	  tot_train_images, ...
	  tot_time, ...
	  rand_state] = ...
      chipFileOfFilenames(chip_path, ...
			  object_name, ...
			  num_train, ...
			  skip_train_images, ...
			  begin_train_images, ...
			  train_dir, ...
			  list_dir, ...
			  shuffle_flag, ...
			  rand_state)

  %% makes list of paths to  DARPA chip images for training,
  %% training files are drawn from images folder train_dir,

  begin_time = time();

  num_argin = 0;
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("chip_path") || isempty(chip_path)
    chip_path = ["/mnt/data1/repo/neovision-programs-petavision/Heli/Training", filesep]; 
%%    chip_path = ["/mnt/data1/repo/neovision-programs-petavision/Heli/Challenge", filesep]; 
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("object_name") || isempty(object_name)
  object_name =  "051"; %% "Car"; %%"distractor"; %%   "030"; %%  "Plane"; %%  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("num_train") || isempty(num_train)
    num_train = -1;  %% -1 use all images in train_dir
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("skip_train_images") || isempty(skip_train_images)
    skip_train_images = 1; %% 4; %% 1;  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("begin_train_images") || isempty(begin_train_images)
    begin_train_images = 1; %% 1;  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("train_dir") || isempty(train_dir)
    train_dir = "canny";  %%  
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("list_dir") || isempty(list_dir)
    list_dir = ["list_", train_dir, num2str(2)];  %% 
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
  disp(['num_train_images = ', num2str(num_train_images)]);
  
  tot_train_images = length(begin_train_images:skip_train_images:num_train_images);
  disp(['tot_train_images = ', num2str(tot_train_images)]);
   

  if num_train < 0
    num_train = tot_train_images;
  endif

  if shuffle_flag
    [rank_ndx, write_train_ndx] = sort(rand(num_train_images,1));
    write_train_ndx = write_train_ndx(begin_train_images:skip_train_images:num_train_images);
  else
    write_train_ndx = begin_train_images:skip_train_images:num_train_images;
  endif

  if num_train < tot_train_images
    write_train_ndx = write_train_ndx(1:num_train);
  elseif num_train > tot_train_images
    num_train = tot_train_images;
  endif

  noclobber_flag = 0;
  if noclobber_flag
     num_fileOfFilenames_train = length(glob([filenames_path, "train_fileOfFilenames", "[0-9+].txt"]));
     fileOfFilenames_train = [filenames_path, "train_fileOfFilenames", num2str(num_fileOfFilenames_train+1), ".txt"];
  else
     fileOfFilenames_train = [filenames_path, "train_fileOfFilenames", ".txt"];
  endif
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
