
function [fileOfFilenames_train, ...
	  fileOfFilenames_test, ...
	  num_train, ...
	  num_test, ...
	  tot_time, ...
	  rand_state] = ...
      sampleFileOfFilenames(fileOfFilenames_all, ...
			    num_train, ...
			    num_test, ...
			    sample_method, ...
			    rand_state, ...
			    train_dir, ...
			    anti_dir)

  %% samples file of filenames containing lists of images for training and testing classification algorithms

  begin_time = time();

  if nargin < 1 || ~exist("fileOfFilenames_all") || isempty(fileOfFilenames_all)
    fileOfFilenames_all = "/Users/gkenyon/Pictures/imageNet/list/dog/poodle/train_fileOfFilenames3.txt";  
    %% fileOfFilenames_all = "/Users/gkenyon/Pictures/imageNet/list/cat/train_fileOfFilenames3.txt";  
  endif
  if nargin < 2 || ~exist("num_train") || isempty(num_train)
    num_train = -1;  %% -1 == extract all not used for test
  endif
  if nargin < 3 || ~exist("num_test") || isempty(num_test)
    num_test = 100;  %% -1 == extract all not used for train
  endif
  if nargin < 4 || ~exist("sample_method") || isempty(sample_method)
    sample_method = 1;  %% 0 == append, 1 == shuffle
  endif
  if nargin < 5 || ~exist("rand_state") || isempty(rand_state)
    if sample_method == 1
      rand_state = rand("state");
    else
      rand_state = [];
    endif
  endif
  if sample_method == 1
    rand("state", rand_state);
  endif
  if nargin < 6 || ~exist("train_dir") || isempty(train_dir)
    train_dir = "DoGMask"; %%"masks"; %% 
  endif
  if nargin < 7 || ~exist("anti_dir") || isempty(anti_dir)
    anti_dir = "DoGAntiMask"; %% []; %% 
  endif

 
  %%setenv('GNUTERM', 'x11');
  sample_path = strExtractPath(fileOfFilenames_all);
  disp(["sample_path = ", sample_path]);
  mkdir(sample_path);

  if ~exist(fileOfFilenames_all, "file")
    error(["~exist fileOfFilenames_all = ", fileOfFilenames_all]);
  endif
  fid_all = fopen(fileOfFilenames_all, "r", "native");
  %%[first_pathnames, num1] = fscanf(fid_all, "%s\n", "Inf");
  all_pathname = fgets(fid_all);
  all_pathnames = cell(1,1);
  i_path = 0;
  while all_pathname ~= -1
    i_path = i_path + 1;
    len_pathname = length(all_pathname);
    all_pathnames{i_path,1} = all_pathname(1:len_pathname-1);
    all_pathname = fgets(fid_all);
  endwhile
  fclose(fid_all);
  num_all = length(all_pathnames);

  if sample_method == 1
    [rank_ndx, write_ndx] = sort(rand(num_all,1));
  else
    write_ndx = 1:num_all;
  endif

  if num_test > 0 && num_train < 0 
    if num_test >= num_all
      num_train = 0;
    else
      num_train = num_all - num_test;
    endif
  elseif num_train > 0 && num_test < 0
    if num_train >= num_all
      num_test = 0;
    else
      num_test = num_all - num_train;
    endif
  elseif num_test > 0 && num_train > 0
    if (num_train + num_test) >= num_all
      num_test = floor(num_test*num_all/(num_train+num_test));
      num_train = floor(num_train*num_all/(num_train+num_test));
    endif    
  else
    error("num_train < 0 & num_test < 0");
  endif  %% num_test > 0
  num_tot = num_test + num_train;

  
  num_fileOfFilenames_train = ...
      length(glob([sample_path, ...
		   "train_fileOfFilenames", "[0-9+]", "_", num2str(num_train), ".txt"]));
  fileOfFilenames_train = ...
      [sample_path, ...
       "train_fileOfFilenames", ...
       num2str(num_fileOfFilenames_train+1), "_", num2str(num_train),".txt"];
  disp(["fileOfFilenames_train = ", fileOfFilenames_train]);
  fid_train = fopen(fileOfFilenames_train, "w", "native");
  for i_train = 1 : num_train
    i_image = write_ndx(i_train);
    write_pathname = all_pathnames{i_image};
    fprintf(fid_train, "%s\n", write_pathname);
  endfor %%
  fclose(fid_train);

  num_fileOfFilenames_test = ...
      length(glob([sample_path, ...
		   "test_fileOfFilenames", "[0-9+]", "_", num2str(num_test), ".txt"]));
  fileOfFilenames_test = ...
      [sample_path, ...
       "test_fileOfFilenames", ...
       num2str(num_fileOfFilenames_test+1), "_", num2str(num_test),".txt"];
  disp(["fileOfFilenames_test = ", fileOfFilenames_test]);
  fid_test = fopen(fileOfFilenames_test, "w", "native");
  for i_test = num_train + 1 : num_tot
    i_image = write_ndx(i_test);
    write_pathname = all_pathnames{i_image};
    fprintf(fid_test, "%s\n", write_pathname);
  endfor %%
  fclose(fid_test);

  if ~isempty(anti_dir)
    fileOfFilenames_anti = ...
	[sample_path, ...
	 "anti_fileOfFilenames", ...
	 num2str(num_fileOfFilenames_train+1), "_", num2str(num_train),".txt"];
    disp(["fileOfFilenames_anti = ", fileOfFilenames_anti]);
    fid_anti = fopen(fileOfFilenames_anti, "w", "native");
    for i_train = 1 : num_train
      i_image = write_ndx(i_train);
      write_pathname = all_pathnames{i_image};
      anti_pathname = strSwap(write_pathname, train_dir, anti_dir);
      fprintf(fid_anti, "%s\n", anti_pathname);
    endfor %%
    fclose(fid_anti);
  endif

  if sample_method == 1
    num_rand_state = length(glob([sample_path, "rand_state_sample", "[0-9+].mat"]));
    rand_state_filename = [sample_path, "rand_state_sample", num2str(num_rand_state+1), ".mat"];
    disp(["rand_state_filename = ", rand_state_filename]);
    save("-binary", rand_state_filename, "rand_state");
  endif

  end_time = time;
  tot_time = end_time - begin_time;

 endfunction %% imageNetFileOfFilenames