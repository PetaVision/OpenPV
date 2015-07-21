
function [train_pos_images, ...
	  train_neg_images, ...
	  test_pos_images, ...
	  test_neg_images, ...
	  num_train_pos, ...
	  num_train_neg, ...
	  num_test_pos, ...
	  num_test_neg, ...
	  tot_time, ...
	  rand_state] = ...
      indexFiles(data_path, ...
		 base_name, ...
		 num_train, ...
		 num_test, ...
		 train_dir, ...
		 test_dir, ...
		 shuffle_flag, ...
		 rand_state)

  %% makes list of paths to PANN training and testing image files

  begin_time = time();

  if nargin < 1 || ~exist(imageNet_path) || isempty(imageNet_path)
    data_path = "~/PANN/neural_computing/data";
  endif
  if nargin < 2 || ~exist(base_name) || isempty(base_name)
    base_name = "demo_animals"; %%  
  endif
  if nargin < 3 || ~exist("num_train") || isempty(num_train)
    num_train = -1;  %% -1 use all images in train_dir
  endif
  if nargin < 4 || ~exist("num_test") || isempty(num_test)
    num_test = -1;  %% -1 use all images in test_dir not in train_dir
  endif
  if nargin < 5 || ~exist(train_dir) || isempty(train_dir)
    train_dir = "train";  %% 
  endif
  if nargin < 6 || ~exist(test_dir) || isempty(test_dir)
    test_dir = "test";  %% 
  endif
  if nargin < 7 || ~exist("shuffle_flag") || isempty(shuffle_flag)
    shuffle_flag = 1;  %% 
  endif
  if nargin < 8 || ~exist("rand_state") || isempty(rand_state)
    rand_state = rand("state");
  endif
  rand("state", rand_state);

 
  %%setenv('GNUTERM', 'x11');

  local_dir = pwd;
  image_type = "jpg";

  database_path = [data_path, filesep, base_name, filesep];
  if ~exist("database_path") 
    error(["database path does not exist: ", database_path]);
  endif
  index_path = [database_path, "index", filesep];
  mkdir(index_path);

  %% path to generic image processing routins
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);

  train_path = [database_path, train_dir, filesep];
  train_pos_path = [train_path, "pos", filesep];
  train_neg_path = [train_path, "neg",  filesep];
  test_path = [database_path, test_dir, filesep];
  test_pos_path = [test_path, "pos", filesep];
  test_neg_path = [test_path, "neg", filesep];

  num_train_pos = 0;
  num_train_neg = 0;
  num_test_pos = 0;
  num_test_neg = 0;

  train_pos_images = glob([train_pos_path,"*.", image_type]);
  num_train_pos = length(train_pos_images);
  disp(["num_train_pos = ", num2str(num_train_pos)]);

  train_neg_images = glob([train_neg_path,"*.", image_type]);
  num_train_neg = length(train_neg_images);
  disp(["num_train_neg = ", num2str(num_train_neg)]);

  test_pos_images = glob([test_pos_path,"*.", image_type]);
  num_test_pos = length(test_pos_images);
  disp(["num_test_pos = ", num2str(num_test_pos)]);

  test_neg_images = glob([test_neg_path,"*.", image_type]);
  num_test_neg = length(test_neg_images);
  disp(["num_test_neg = ", num2str(num_test_neg)]);

  if num_train < 1 
    tot_train_pos = num_train_pos;
  else
    tot_train_pos = min(num_train, num_train_pos);
  endif 
  if tot_train_pos < num_train_pos  || shuffle_flag
    [train_pos_rank, train_pos_ndx] = sort(rand(num_train_pos,1));
  else
    train_pos_ndx = 1:num_train_pos;
  endif

  num_train_pos_files = ...
      length(glob([database_path, "train_pos", "[0-9+].txt"]));
  train_pos_filename = ...
      [index_path, "train_pos_index", ...
       num2str(num_train_pos_files+1), ".txt"];
  disp(["train_pos_index = ", train_pos_filename]);
  fid_train_pos = fopen(train_pos_filename, "w", "native");
  for i_image = 1 : num_train_pos
    fprintf(fid_train_pos, "%s\n", train_pos_images{train_pos_ndx(i_image)});
  endfor %%
  fclose(fid_train_pos);
  
  if num_train < 1 
    tot_train_neg = num_train_neg;
  else
    tot_train_neg = min(num_train, num_train_neg);
  endif 
  if tot_train_neg < num_train_neg  || shuffle_flag
    [train_neg_rank, train_neg_ndx] = sort(rand(num_train_neg,1));
  else
    train_neg_ndx = 1:num_train_neg;
  endif

  num_train_neg_files = ...
      length(glob([database_path, "train_neg", "[0-9+].txt"]));
  train_neg_filename = ...
      [index_path, "train_neg_index", ...
       num2str(num_train_neg_files+1), ".txt"];
  disp(["train_neg_index = ", train_neg_filename]);
  fid_train_neg = fopen(train_neg_filename, "w", "native");
  for i_image = 1 : num_train_neg
    fprintf(fid_train_neg, "%s\n", train_neg_images{train_neg_ndx(i_image)});
  endfor %%
  fclose(fid_train_neg);
  
  if num_test < 1 
    tot_test_pos = num_test_pos;
  else
    tot_test_pos = min(num_test, num_test_pos);
  endif 
  if tot_test_pos < num_test_pos  || shuffle_flag
    [test_pos_rank, test_pos_ndx] = sort(rand(num_test_pos,1));
  else
    test_pos_ndx = 1:num_test_pos;
  endif

  num_test_pos_files = ...
      length(glob([database_path, "test_pos", "[0-9+].txt"]));
  test_pos_filename = ...
      [index_path, "test_pos_index", ...
       num2str(num_test_pos_files+1), ".txt"];
  disp(["test_pos_index = ", test_pos_filename]);
  fid_test_pos = fopen(test_pos_filename, "w", "native");
  for i_image = 1 : num_test_pos
    fprintf(fid_test_pos, "%s\n", test_pos_images{test_pos_ndx(i_image)});
  endfor %%
  fclose(fid_test_pos);
  
  if num_test < 1 
    tot_test_neg = num_test_neg;
  else
    tot_test_neg = min(num_test, num_test_neg);
  endif 
  if tot_test_neg < num_test_neg  || shuffle_flag
    [test_neg_rank, test_neg_ndx] = sort(rand(num_test_neg,1));
  else
    test_neg_ndx = 1:num_test_neg;
  endif

  num_test_neg_files = ...
      length(glob([database_path, "test_neg", "[0-9+].txt"]));
  test_neg_filename = ...
      [index_path, "test_neg_index", ...
       num2str(num_test_neg_files+1), ".txt"];
  disp(["test_neg_index = ", test_neg_filename]);
  fid_test_neg = fopen(test_neg_filename, "w", "native");
  for i_image = 1 : num_test_neg
    fprintf(fid_test_neg, "%s\n", test_neg_images{test_neg_ndx(i_image)});
  endfor %%
  fclose(fid_test_neg);
  
  num_rand_state = length(glob([index_path, "rand_state_index", "[0-9+].mat"]));
  rand_state_filename = [index_path, "rand_state_index", num2str(num_rand_state+1), ".mat"];
  disp(["rand_state_filename = ", rand_state_filename]);
  save("-binary", rand_state_filename, "rand_state");

  end_time = time;
  tot_time = end_time - begin_time;
 
 endfunction %% indexFile