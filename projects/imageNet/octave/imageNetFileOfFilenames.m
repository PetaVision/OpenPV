
function [train_filenames, ...
	  test_filenames, ...
	  tot_train_images, ...
	  tot_test_images,  ...
	  tot_time, ...
	  rand_state] = ...
      imageNetFileOfFilenames(imageNet_path, ...
			      object_name, ...
			      object_keyword, ...
			      num_train, ...
			      num_test, ...
			      train_dir, ...
			      test_dir, ...
			      anti_dir, ...
			      list_dir, ...
			      cross_category_flag, ...
			      shuffle_flag, ...
			      rand_state)

  %% makes list of paths to  imageNet image files for training and testing
  %% training files are drawn from images folder train_dir
  %% testing files are drawn from images folder train_dir

  begin_time = time();

  if nargin < 1 || ~exist(imageNet_path) || isempty(imageNet_path)
    imageNet_path = "~/Pictures/imageNet/";
  endif
  if nargin < 2 || ~exist(object_name) || isempty(object_name)
    object_name = "dog";  %% "cat"; %% 
  endif
  if nargin < 3 || ~exist(object_keyword) || isempty(object_keyword)
    object_keyword =  ["poodle"]; %%[]; %% ["terrier"];  %%
  endif
  if nargin < 4 || ~exist("num_train") || isempty(num_train)
    num_train = -1;  %% -1 use all images in train_dir
  endif
  if nargin < 5 || ~exist("num_test") || isempty(num_test)
    num_test = -1; %% -1;  %% -1 use all images in test_dir not in train_dir
  endif
  if nargin < 6 || ~exist(train_dir) || isempty(train_dir)
    train_dir = "DoGMask";  %%"masks"; %%  
  endif
  if nargin < 7 || ~exist(test_dir) || isempty(test_dir)
    test_dir = "DoG";  %% "standard"; %% 
  endif
  if nargin < 8 || ~exist(anti_dir) || isempty(anti_dir)
    anti_dir = "DoGAntiMask";  %% []; %% 
  endif
  if nargin < 9 || ~exist("list_dir") || isempty(list_dir)
    list_dir = "list";  %% 
  endif
  %% cross_category_flag == 0, only draw test images from test folders whose
  %% corresponding train folders contain at least abs(num_train) images
  %% cross_category_flag == 1, draw test images only from test folders that lack
  %% corresponding train folders--for testing generalization across
  %% imageNet sub categories
  if nargin < 10 || ~exist("cross_category_flag") || isempty(cross_category_flag)
    cross_category_flag = 0;  %% 
  endif
  if nargin < 11 || ~exist("shuffle_flag") || isempty(shuffle_flag)
    shuffle_flag = 1;  %% 
  endif
  if nargin < 12 || ~exist("rand_state") || isempty(rand_state)
    rand_state = rand("state");
  endif
  rand("state", rand_state);

 
  %%setenv('GNUTERM', 'x11');

  local_dir = pwd;
  image_type = "png";

  train_filenames = {};
  test_filenames = {};
  anti_filenames = {};
  filenames_path = [imageNet_path, list_dir, filesep, object_name, filesep];
  mkdir([imageNet_path, list_dir]);
  mkdir(filenames_path);
  if ~isempty(object_keyword)
    filenames_path = [filenames_path, object_keyword, filesep];
  endif
  mkdir(filenames_path);

  %% path to generic image processing routins
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);

  train_path = [imageNet_path, train_dir, filesep, object_name, filesep];
  test_path = [imageNet_path, test_dir, filesep, object_name, filesep];
  if ~isempty(anti_dir)
    anti_path = [imageNet_path, anti_dir, filesep, object_name, filesep];
  endif

  tot_train_images = 0;
  tot_test_images = 0;

  train_subdir_paths = glob([train_path,"*"]);
  num_train_subdirs = length(train_subdir_paths);
  disp(["num_train_subdirs = ", num2str(num_train_subdirs)]);

  test_subdir_paths = glob([test_path,"*"]);
  num_test_subdirs = length(test_subdir_paths);
  disp(["num_test_subdirs = ", num2str(num_test_subdirs)]);

  train_subdir_folders = ...
      cellfun(@strFolderFromPath, train_subdir_paths, "UniformOutput", false);
  test_subdir_folders = ...
      cellfun(@strFolderFromPath, test_subdir_paths, "UniformOutput", false);

  if cross_category_flag == 0
    [subdir_folders, subdir_index] = ...
	intersect(train_subdir_folders, test_subdir_folders);
  else
    [subdir_folders, subdir_index] = ...
	setdiff(train_subdir_folders, test_subdir_folders);
  endif

  num_subdirs = length(subdir_folders);
  disp(["num_subdirs = ", num2str(num_subdirs)]);

  for i_subdir = 1 : num_subdirs %%fix(num_subdirs/2)
    
    subdir_folder = subdir_folders{i_subdir};
    keyword_ndx = [0];
    if ~isempty(object_keyword)
      keyword_ndx = strfind( subdir_folder, object_keyword);
    endif
    if isempty(keyword_ndx)
      continue;
    endif

    disp(["i_subdir = ", num2str(i_subdir)]);
    disp(["subdir_name = ", subdir_folder]);

    train_subdir = [train_path, subdir_folder, filesep];
    train_search_str = ...
	[train_subdir, '*.', image_type];
    train_paths = glob(train_search_str);
    train_names = ...
	cellfun(@strFolderFromPath, train_paths, "UniformOutput", false);
    num_train_images = size(train_names,1);   
    tot_train_images = tot_train_images + num_train_images;
    disp(['num_train_images = ', num2str(num_train_images)]);
    if cross_category_flag == 0 && num_train_images == 0
      continue;
    endif
    train_filenames = [train_filenames; train_paths];
   
    test_subdir = [test_path, subdir_folder, filesep];
    test_search_str = ...
	[test_subdir, '*.', image_type];
    test_paths = glob(test_search_str);
    test_names = ...
	cellfun(@strFolderFromPath, test_paths, "UniformOutput", false);
    [unique_names, unique_index] = ...
	  setdiff(test_names, train_names);
    num_unique = length(unique_names);
    tot_test_images = tot_test_images + num_unique;
    unique_paths = strcat(repmat(test_subdir, num_unique, 1), unique_names);
    disp(['num_test_images = ', num2str(num_unique)]);
    if cross_category_flag == 1 && num_unique == 0
      continue;
    endif
    test_filenames = [test_filenames; unique_paths];

    if ~isempty(anti_dir)
      anti_subdir = [anti_path, subdir_folder, filesep];
      anti_paths = strcat(repmat(anti_subdir, num_train_images, 1), train_names);
      anti_filenames = [anti_filenames; anti_paths];
    endif

  endfor %% i_subdir

  tot_train_files = length(train_filenames);
  disp(["tot_train_files = ", num2str(tot_train_files)]);
  if tot_train_files ~= tot_train_images
    error("tot_train_files ~= tot_train_images");
  endif
  tot_test_files = length(test_filenames);
  disp(["tot_test_files = ", num2str(tot_test_files)]);
  if tot_test_files ~= tot_test_images
    error("tot_test_files ~= tot_test_images");
  endif

  if num_train < 0
    num_train = tot_train_files;
  endif
  if num_test < 0
    num_test = tot_test_files;
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
  
  if num_test < tot_test_files  || shuffle_flag
    [rank_ndx, write_test_ndx] = sort(rand(tot_test_images,1));
  else
    write_test_ndx = 1:num_test;
  endif

  num_fileOfFilenames_test = length(glob([filenames_path, "test_fileOfFilenames", "[0-9+].txt"]));
  fileOfFilenames_test = [filenames_path, "test_fileOfFilenames", num2str(num_fileOfFilenames_test+1), ".txt"];
  disp(["fileOfFilenames_test = ", fileOfFilenames_test]);
  fid_test = fopen(fileOfFilenames_test, "w", "native");
  for i_file = 1 : num_test
    fprintf(fid_test, "%s\n", test_filenames{write_test_ndx(i_file)});
  endfor %%
  fclose(fid_test);

  if ~isempty(anti_dir)
    fileOfFilenames_anti = [filenames_path, "anti_fileOfFilenames", num2str(num_fileOfFilenames_train+1), ".txt"];
    disp(["fileOfFilenames_anti = ", fileOfFilenames_anti]);
    fid_anti = fopen(fileOfFilenames_anti, "w", "native");
    for i_file = 1 : num_train
      fprintf(fid_anti, "%s\n", anti_filenames{write_train_ndx(i_file)});
    endfor %%
    fclose(fid_anti);
  endif

  num_rand_state = length(glob([filenames_path, "rand_state", "[0-9+].mat"]));
  rand_state_filename = [filenames_path, "rand_state", num2str(num_rand_state+1), ".mat"];
  disp(["rand_state_filename = ", rand_state_filename]);
  save("-binary", rand_state_filename, "rand_state");

  end_time = time;
  tot_time = end_time - begin_time;

 endfunction %% imageNetFileOfFilenames