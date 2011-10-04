
function [tot_chips, ...
	  tot_DoG, ...
	  tot_canny, ...
	  tot_time] = ...
      padChips(chip_path, ...
	       object_name, ...
	       DoG_flag, ...
	       DoG_struct, ...
	       canny_flag, ...
	       canny_struct, ...
	       pad_size, ...
	       num_procs)
  
  %% perform edge filtering on DARPA NeoVis2 target chips, 
  %% mirror BCs used to pad images before edge extraction.
  %% also performs edge extraction on mask images if present

  global DoG_flag
  global canny_flag
  global DoG_dir
  global DoG_struct
  global canny_dir
  global canny_struct
  global image_margin
  global pad_size

  global VERBOSE_FLAG
  if ~exist("VERBOSE_FLAG") || isempty(VERBOSE_FLAG)
    VERBOSE_FLAG = 0;
  endif
 
  begin_time = time();

  if nargin < 1 || ~exist("chip_path") || isempty(chip_path)
    chip_path = "~/Pictures/HeliChips/";
  endif
  if nargin < 2 || ~exist("object_name") || isempty(object_name)
    object_name = "Car";  %% could be a list?
  endif
  if nargin < 3 || ~exist("DoG_flag") || isempty(DoG_flag)
    DoG_flag = 1;  %% 
  endif
  if nargin < 4 || ~exist("DoG_struct") || isempty(DoG_struct)
    DoG_struct = struct;  %% 
    DoG_struct.amp_center_DoG = 1;
    DoG_struct.sigma_center_DoG = 1;
    DoG_struct.amp_surround_DoG = 1;
    DoG_struct.sigma_surround_DoG = 2 * DoG_struct.sigma_center_DoG;
  endif
  if nargin < 5 || ~exist("canny_flag") || isempty(canny_flag)
    canny_flag = 1;  %% 
  endif
  if nargin < 6 || ~exist("canny_struct") || isempty(canny_struct)
    canny_struct = struct;  %% 
    canny_struct.sigma_canny = 1;
  endif
  if nargin < 7 || ~exist("pad_size") || isempty(pad_size)
    pad_size = [256 256];  %% 
  endif
  if nargin < 8 || ~exist("num_procs") || isempty(num_procs)
    num_procs = 4;  %% 
  endif
  
  setenv('GNUTERM', 'x11');

  local_dir = pwd;

  tot_chips = 0;
  tot_targets = 0;
  tot_DoG = 0;
  tot_canny = 0;

  %% path to generic image processing routines
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);

  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);

  chip_dir = [chip_path, "chips", filesep];
  if DoG_flag
    DoG_folder = [chip_path, "DoG", filesep];
    mkdir(DoG_folder);
    DoG_dir = [DoG_folder, object_name, filesep];
    mkdir(DoG_dir);
  endif %% DoG_flag
  if canny_flag
    canny_folder = [chip_path, "canny", filesep];
    mkdir(canny_folder);
    canny_dir = [canny_folder, object_name, filesep];
    mkdir(canny_dir);
  endif %% canny_flag

  image_type = ".png";
  image_margin = 8;

  target_dir = ...
      [chip_dir, object_name, filesep];  %%

  target_path = ...
      [target_dir, '*', image_type];
  target_pathnames = glob(target_path);
  num_chips = size(target_pathnames,1);
  disp(['num_chips = ', num2str(num_chips)]);

  tot_chips = tot_chips + num_chips;
    
  if num_procs > 1
    [status_info] = ...
	parcellfun(num_procs, @padChipKernel, target_pathnames, "UniformOutput", false);
  else
    [status_info] = ...
	cellfun(@padChipKernel, target_pathnames, "UniformOutput", false);
  endif

  for i_chip = 1 : num_chips
      tot_targets = tot_targets + status_info{i_chip}.target_flag;
      tot_DoG = tot_DoG + status_info{i_chip}.DoG_flag;
      tot_canny = tot_canny + status_info{i_chip}.canny_flag;
   endfor %% i_chip


  disp(["tot_chips = ", ...
	num2str(tot_chips)]);
  disp(["tot_DoG = ", ...
	num2str(tot_DoG)]);
  disp(["tot_canny = ", ...
	num2str(tot_canny)]);
  
  end_time = time();
  tot_time = end_time - begin_time;
  disp(["tot_time = ", num2str(tot_time)]);


endfunction %% imageNetEdgeFilter