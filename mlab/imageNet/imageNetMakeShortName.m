
function [long_names, ...
	  short_names, ...
	  tot_time] = ...
      imageNetMakeShortName(imageNet_path, object_name, long_dir, short_dir)

  %% clip imageNet subfolder names to first key word
  %% e.g. "Cocker spaniel, English cocker spaniel, cocker" -> "Cocker spaniel"

  begin_time = time();

  if nargin < 1 || ~exist(imageNet_path) || isempty(imageNet_path)
    imageNet_path = "~/Pictures/imageNet/";
  endif
  if nargin < 2 || ~exist(object_name) || isempty(object_name)
    object_name = "dog";  %% could be a list?
  endif
  if nargin < 3 || ~exist(long_dir) || isempty(long_dir)
    long_dir = "DoG";
  endif
  if nargin < 4 || ~exist(short_dir) || isempty(short_dir)
    short_dir = long_dir;  %% default is to rename subdirectories,
    %% otherwise move 
  endif

  long_path = [imageNet_path, long_dir, filesep, object_name, filesep];
  short_path = [imageNet_path, short_dir, filesep, object_name, filesep];
  long_subdir_paths = glob([long_path,"*"]);
  num_long_subdirs = length(long_subdir_paths);
  disp(["num_long_subdirs = ", num2str(num_long_subdirs)]);
  long_names = ...
      cellfun(@strFolderFromPath, long_subdir_paths, "UniformOutput", false);
  short_names = ...
      cellfun(@strShortName, long_folder_names, "UniformOutput", false);
  long_paths = strcat(repmat(long_path, num_long_subdirs, 1), long_names);
  short_paths = strcat(repmat(short_path, num_long_subdirs, 1), short_names);

  if strcmp(long_dir == short_dir)
    cellfun(@renameFolders, long_path_names, short_path_names, ...
	    "UniformOutput", false);
  else
    cellfun(@moveFolders, long_path_names, short_path_names, ...
	    "UniformOutput", false);    
  endif

  end_time = time;
  tot_time = end_time - begin_time;

 endfunction %% imageNetFileOfFilenames