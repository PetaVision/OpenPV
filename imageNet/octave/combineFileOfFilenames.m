
function [fileOfFilenames_combine, ...
	  num1, ...
	  num2,  ...
	  tot_time, ...
	  rand_state] = ...
      combineFileOfFilenames(fileOfFilenames1, fileOfFilenames2, combine_dir, merge_method, rand_state)

  %% combines file of filenames containing lists of images for training and testing classification algorithms

  begin_time = time();

  if nargin < 1 || ~exist("fileOfFilenames1") || isempty(fileOfFilenames1)
    fileOfFilenames1 = "/Users/gkenyon/Pictures/imageNet/colorlist/dog/poodle/train_fileOfFilenames3.txt";  
  endif
  if nargin < 2 || ~exist("fileOfFilenames2") || isempty(fileOfFilenames2)
    fileOfFilenames2 = "/Users/gkenyon/Pictures/imageNet/colorlist/cat/train_fileOfFilenames3.txt";  
  endif
  if nargin < 3 || ~exist("combine_dir") || isempty(combine_dir)
    combine_dir = "color_poodle_cat";  
  endif
  if nargin < 4 || ~exist("merge_method") || isempty(merge_method)
    merge_method = 1;  %% 0 == append, 1 == shuffle
  endif
  if nargin < 5 || ~exist("rand_state") || isempty(rand_state)
    if merge_method == 1
      rand_state = rand("state");
    else
      rand_state = [];
    endif
  endif
  if merge_method == 1
    rand("state", rand_state);
  endif

 
  %%setenv('GNUTERM', 'x11');
  first_path = strExtractPath(fileOfFilenames1);
  second_path = strExtractPath(fileOfFilenames2);
  common_path = strCommon(first_path, second_path);
  if isempty(common_path)
    warning(["common_path is empty: ", "first_path = ", first_path, " second_path = ", second_path]);
  endif
  combine_path = [common_path, combine_dir, filesep];
  disp(["combine_path = ", combine_path]);
  mkdir(combine_path);

  if ~exist(fileOfFilenames1, "file")
    error(["~exist fileOfFilenames1 = ", fileOfFilenames1]);
  endif
  fid1 = fopen(fileOfFilenames1, "r", "native");
  %%[first_pathnames, num1] = fscanf(fid1, "%s\n", "Inf");
  first_pathname = fgets(fid1);
  first_pathnames = cell(1,1);
  i1 = 0;
  while first_pathname ~= -1
    i1 = i1 + 1;
    first_pathnames{i1,1} = first_pathname(1:end-1);
    first_pathname = fgets(fid1);
  endwhile
  fclose(fid1);
  num1 = length(first_pathnames);

  if ~exist(fileOfFilenames2, "file")
    error(["~exist fileOfFilenames2 = ", fileOfFilenames2]);
  endif
  fid2 = fopen(fileOfFilenames2, "r", "native");
  %%[second_pathnames, num2] = fscanf(fid2, "%s\n", "Inf");
  second_pathname = fgets(fid2);
  second_pathnames = cell(1,1);
  i2 = 0;
  while second_pathname ~= -1
    i2 = i2 + 1;
    second_pathnames{i2,1} = second_pathname(1:end-1);
    second_pathname = fgets(fid2);
  endwhile
  fclose(fid2);
  num2 = length(second_pathnames);

  num_tot = num1 + num2;

  if merge_method == 1
    [rank_ndx, write_ndx] = sort(rand(num_tot,1));
  else
    write_ndx = 1:num_tot;
  endif

  
  num_fileOfFilenames_combine = ...
      length(glob([combine_path, "combine_fileOfFilenames", "[0-9+].txt"]));
  fileOfFilenames_combine = ...
      [combine_path, "combine_fileOfFilenames", num2str(num_fileOfFilenames_combine+1), ".txt"];
  disp(["fileOfFilenames_combine = ", fileOfFilenames_combine]);
  fid_combine = fopen(fileOfFilenames_combine, "w", "native");
  for i_file = 1 : num_tot
    i_image = write_ndx(i_file);
    if i_image <= num1
      write_pathname = first_pathnames{i_image};
    else
      write_pathname = second_pathnames{i_image - num1};
    endif
    fprintf(fid_combine, "%s\n", write_pathname);
  endfor %%
  fclose(fid_combine);

  if merge_method == 1
    num_rand_state = length(glob([combine_path, "rand_state_combine", "[0-9+].mat"]));
    rand_state_filename = [combine_path, "rand_state_combine", num2str(num_rand_state+1), ".mat"];
    disp(["rand_state_filename = ", rand_state_filename]);
    save("-binary", rand_state_filename, "rand_state");
  endif

  end_time = time;
  tot_time = end_time - begin_time;

 endfunction %% imageNetFileOfFilenames