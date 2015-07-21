
function [tot_images, ...
	  tot_masks, ...
	  tot_DoG, ...
	  tot_DoG_masks, ...
	  tot_canny, ...
	  tot_canny_masks, ...
	  tot_time] = ...
      imageNetEdgeFilter2(imageNet_path, ...
			  object_name, ...
			  mask_flag, ...
			  mask_only_flag, ...
			  DoG_flag, ...
			  DoG_struct, ...
			  canny_flag, ...
			  canny_struct, ...
			  num_procs, ...
			  antimask_flag)

  %% perform edge filtering on standardized image net images, 
  %% mirror BCs used to pad images before edge extraction.
  %% also performs edge extraction on mask images if present

  global mask_flag
  global antimask_flag
  global DoG_flag
  global canny_flag
  global DoG_subdir
  global DoG_mask_subdir
  global DoG_antimask_subdir
  global DoG_struct
  global canny_subdir
  global canny_mask_subdir
  global canny_antimask_subdir
  global canny_struct
  global image_margin

  global VERBOSE_FLAG
  if ~exist("VERBOSE_FLAG") || isempty(VERBOSE_FLAG)
    VERBOSE_FLAG = 0;
  endif
 
  begin_time = time();

  if nargin < 1 || ~exist("imageNet_path") || isempty(imageNet_path)
    imageNet_path = "~/Pictures/imageNet/";
  endif
  if nargin < 2 || ~exist("object_name") || isempty(object_name)
    object_name = "car";  %% could be a list?
  endif
  if nargin < 3 || ~exist("mask_flag") || isempty(mask_flag)
    mask_flag = 1;  %% apply edge filtering to masked images 
  endif
  %% mask_only_flag
  %% if set to 1, only process object subdirs with bounding box masks
  %% if set to -1, only process object subdirs without bounding box masks
  %% if set to 0, process all object subdirs
  if nargin < 4 || ~exist("mask_only_flag") || isempty(mask_only_flag)
    mask_only_flag = mask_flag;  %% 
  endif
  if nargin < 5 || ~exist("DoG_flag") || isempty(DoG_flag)
    DoG_flag = 1;  %% 
  endif
  if nargin < 6 || ~exist("DoG_struct") || isempty(DoG_struct)
    DoG_struct = struct;  %% 
    DoG_struct.amp_center_DoG = 1;
    DoG_struct.sigma_center_DoG = 1;
    DoG_struct.amp_surround_DoG = 1;
    DoG_struct.sigma_surround_DoG = 2 * DoG_struct.sigma_center_DoG;
  endif
  if nargin < 7 || ~exist("canny_flag") || isempty(canny_flag)
    canny_flag = 1;  %% 
  endif
  if nargin < 8 || ~exist("canny_struct") || isempty(canny_struct)
    canny_struct = struct;  %% 
    canny_struct.sigma_canny = 1;
  endif
  if nargin < 9 || ~exist("num_procs") || isempty(num_procs)
    num_procs = 2;  %% 
  endif
  if nargin < 10 || ~exist("antimask_flag") || isempty(antimask_flag)
    antimask_flag = mask_flag;  %% apply edge filtering to unmasked portion of images 
  endif
  
  %%setenv('GNUTERM', 'x11');

  local_dir = pwd;

  tot_images = 0;
  tot_targets = 0;
  tot_masks = 0;
  tot_DoG = 0;
  tot_DoG_masks = 0;
  tot_canny = 0;
  tot_canny_masks = 0;

  %% path to generic image processing routins
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);

  standard_dir = [imageNet_path, "standard", filesep];
  if mask_flag 
    masks_dir = [imageNet_path, "masks", filesep];
    mkdir(masks_dir);
  endif %% mask_flag
  if DoG_flag
    DoG_dir = [imageNet_path, "DoG", filesep, object_name, filesep];
    mkdir(DoG_dir);
    if mask_flag 
      DoG_mask_dir = [imageNet_path, "DoGMask", filesep, object_name, filesep];
      mkdir(DoG_mask_dir);
      if antimask_flag
	DoG_antimask_dir = [imageNet_path, "DoGAntiMask", filesep, object_name, filesep];
	mkdir(DoG_antimask_dir);
      endif
    endif %% mask_flag
  endif %% DoG_flag
  if canny_flag
    canny_dir = [imageNet_path, "canny", filesep, object_name, filesep];
    mkdir(canny_dir);
    if mask_flag 
      canny_mask_dir = [imageNet_path, "cannyMask", filesep, object_name, filesep];
      mkdir(canny_mask_dir);
      if antimask_flag
	canny_antimask_dir = [imageNet_path, "cannyAntiMask", filesep, object_name, filesep];
	mkdir(canny_antimask_dir);
      endif
    endif %% mask_flag
  endif %% canny_flag

  image_type = ".png";
  mask_type = ".png";
  image_margin = 8;

  target_dir = ...
      [standard_dir, object_name, filesep];  %%
  if mask_flag 
    target_mask_dir = ...
	[masks_dir, object_name, filesep];  %%
  endif %% mask_flag

  subdir_pathnames = glob([target_dir,"*"]);
  num_subdirs = length(subdir_pathnames);
  disp(["num_subdirs = ", num2str(num_subdirs)]);
  for i_subdir = 1 : num_subdirs %%fix(num_subdirs/2)
    disp(["i_subdir = ", num2str(i_subdir)]);
    disp(["subdir_name = ", subdir_pathnames{i_subdir}]);

    subdir_path = subdir_pathnames{i_subdir};
    subdir_folder = strFolderFromPath(subdir_path);

    target_subdir = [subdir_path, filesep];
    target_path = ...
	[target_subdir, '*', image_type];
    target_pathnames = glob(target_path);
    num_images = size(target_pathnames,1);
    disp(['num_images = ', num2str(num_images)]);

    if mask_flag
      target_mask_subdir = ...
	  [target_mask_dir, ...
	   subdir_folder, filesep];
      target_mask_path = ...
	  [target_mask_subdir, '*', mask_type];
      mask_images = glob(target_mask_path);
      num_masks = size(mask_images,1);
      disp(['num_masks = ', num2str(num_masks)]);
    endif %% mask_flag
    if mask_flag && mask_only_flag == 1 && num_masks == 0
      continue;
    elseif ~mask_flag && mask_only_flag == -1 && num_masks > 0
      continue;
    endif

    if mask_flag 
      target_filenames = cellfun(@strFolderFromPath, target_pathnames, "UniformOutput", false);
      target_base = cellfun(@strRemoveExtension, target_filenames, "UniformOutput", false);
      mask_base = strcat(target_mask_subdir, target_base);
      mask_pathnames = strcat(mask_base, repmat(mask_type, num_images, 1));
    else
      mask_pathnames = cell(num_images, 1);
    endif %% mask_flag

    tot_images = tot_images + num_images;
    
    if DoG_flag
      DoG_subdir = ...
	  [DoG_dir, ...
	   subdir_folder, filesep];
      mkdir(DoG_subdir);
      if mask_flag
	DoG_mask_subdir = ...
	    [DoG_mask_dir, ...
	     subdir_folder, filesep];
	mkdir(DoG_mask_subdir);
	if antimask_flag
	  DoG_antimask_subdir = ...
	      [DoG_antimask_dir, ...
	       subdir_folder, filesep];
	  mkdir(DoG_antimask_subdir);
	endif
      endif  %% mask_flag
    endif %% DoG_flag

    if canny_flag
      canny_subdir = ...
	  [canny_dir, ...
	   subdir_folder, "/"];
      mkdir(canny_subdir);
      if mask_flag
	canny_mask_subdir = ...
	    [canny_mask_dir, ...
	     subdir_folder, "/"];
	mkdir(canny_mask_subdir);
	if antimask_flag
	  canny_antimask_subdir = ...
	      [canny_antimask_dir, ...
	       subdir_folder, filesep];
	  mkdir(canny_antimask_subdir);
	endif
      endif  %% mask_flag
    endif %% canny_flag

    if num_procs > 1
      [status_info] = ...
	  parcellfun(num_procs, @imageNetEdgeKernel, target_pathnames, mask_pathnames, "UniformOutput", false);
    else
      [status_info] = ...
	  cellfun(@imageNetEdgeKernel, target_pathnames, mask_pathnames, "UniformOutput", false);
    endif

    for i_image = 1 : num_images
      tot_targets = tot_targets + status_info{i_image}.target_flag;
      tot_masks = tot_masks + status_info{i_image}.mask_flag;
      tot_DoG = tot_DoG + status_info{i_image}.DoG_flag;
      tot_DoG_masks = tot_DoG_masks + status_info{i_image}.DoG_mask_flag;
      tot_canny = tot_canny + status_info{i_image}.canny_flag;
      tot_canny_masks = tot_canny_masks + status_info{i_image}.canny_mask_flag;
   endfor %% i_image

  endfor  %% i_subdir

  disp(["tot_images = ", ...
	num2str(tot_images)]);
  disp(["tot_masks = ", ...
	num2str(tot_masks)]);
  disp(["tot_DoG = ", ...
	num2str(tot_DoG)]);
  disp(["tot_DoG_masks = ", ...
	num2str(tot_DoG_masks)]);
  disp(["tot_canny = ", ...
	num2str(tot_canny)]);
  disp(["tot_canny_masks = ", ...
	num2str(tot_canny_masks)]);
  
  end_time = time();
  tot_time = end_time - begin_time;
  disp(["tot_time = ", num2str(tot_time)]);


endfunction %% imageNetEdgeFilter