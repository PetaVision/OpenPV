
function [tot_images, ...
	  tot_masks, ...
	  tot_time] = ...
      imageNetEdgeFilter(imageNet_path, object_name, ...
		     mask_flag, mask_only_flag, ...
		     DoG_flag, DoG_struct, ...
		     canny_flag, canny_struct)

  %% perform edge filtering on standardized image net images, 
  %% mirror BCs used to pad images before edge extraction.
  %% also performs edge extraction on mask images if present

  begin_time = time();

  if nargin < 1 || ~exist(imageNet_path) || isempty(imageNet_path)
    imageNet_path = "~/Pictures/imageNet/";
  endif
  if nargin < 2 || ~exist(object_name) || isempty(object_name)
    object_name = "cat";  %% could be a list?
  endif
  if nargin < 3 || ~exist(mask_flag) || isempty(mask_flag)
    mask_flag = 1;  %% 
  endif
  %% mask_only_flag
  %% if set to 1, only process object subdirs with bounding box masks
  %% if set to -1, only process object subdirs without bounding box masks
  %% if set to 0, process all object subdirs
  if nargin < 4 || ~exist(mask_only_flag) || isempty(mask_only_flag)
    mask_only_flag = 1;  %% 
  endif
  if nargin < 5 || ~exist(DoG_flag) || isempty(DoG_flag)
    DoG_flag = 1;  %% 
  endif
  if nargin < 6 || ~exist(DoG_struct) || isempty(DoG_struct)
    DoG_struct = struct;  %% 
    DoG_struct.amp_center_DoG = 1;
    DoG_struct.sigma_center_DoG = 1;
    DoG_struct.amp_surround_DoG = 1;
    DoG_struct.sigma_surround_DoG = 2 * DoG_struct.sigma_center_DoG;
  endif
  if nargin < 7 || ~exist(canny_flag) || isempty(canny_flag)
    canny_flag = 1;  %% 
  endif
  if nargin < 8 || ~exist(canny_struct) || isempty(canny_struct)
    canny_struct = struct;  %% 
    canny_struct.sigma_canny = 1;
  endif
  
  %%setenv('GNUTERM', 'x11');

  local_dir = pwd;

  %% path to generic image processing routins
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);

  standard_dir = [imageNet_path, "standard/"];
  mask_flag = 1;
  if mask_flag 
    masks_dir = [imageNet_path, "masks/"];
    mkdir(masks_dir);
  endif %% mask_flag
  DoG_flag = 1;
  if DoG_flag
    DoG_dir = [imageNet_path, "DoG/", object_name, "/"];
    mkdir(DoG_dir);
    if mask_flag 
      DoG_mask_dir = [imageNet_path, "DoGMask/", object_name, "/"];
      mkdir(DoG_mask_dir);
    endif %% mask_flag
  endif %% DoG_flag
  canny_flag = 0.0;
  if canny_flag
    canny_dir = [imageNet_path, "canny/", object_name, "/"];
    mkdir(canny_dir);
    if mask_flag 
      canny_mask_dir = [imageNet_path, "canny_mask/", object_name, "/"];
      mkdir(canny_mask_dir);
    endif %% mask_flag
  endif %% canny_flag

  image_type = "png";
  image_margin = 8;
  tot_images = 0;
  tot_masks = 0;

  target_dir = ...
      [standard_dir, object_name, "/"];  %%
  if mask_flag 
    target_mask_dir = ...
	[masks_dir, object_name, "/"];  %%
  endif %% mask_flag

  subdir_names = glob([target_dir,"*"]);
  num_subdirs = length(subdir_names);
  disp(["num_subdirs = ", num2str(num_subdirs)]);
  for i_subdir = 1 : num_subdirs %%fix(num_subdirs/2)
    disp(["i_subdir = ", num2str(i_subdir)]);
    disp(["subdir_name = ", subdir_names{i_subdir}]);

    subdir_path = subdir_names{i_subdir};
    subdir_sep_ndx = strfind(subdir_path, filesep);
    subdir_folder = subdir_path(subdir_sep_ndx(end)+1:end);

    target_subdir = [subdir_path, "/"];
    target_path = ...
	[target_subdir, '*.', image_type];

    if mask_flag 
      target_mask_subdir = ...
	  [target_mask_dir, ...
	   subdir_folder, "/"];
      target_mask_path = ...
	  [target_mask_subdir, '*.', image_type];
    endif %% mask_flag

    target_images = glob(target_path);
    mask_images = glob(target_mask_path);

    num_images = size(target_images,1);
    disp(['num_images = ', num2str(num_images)]);
    
    num_masks = size(mask_images,1);
    disp(['num_masks = ', num2str(num_masks)]);

    if mask_only_flag == 1 && num_masks == 0
      continue;
    elseif
      mask_only_flag == -1 && num_masks > 0
    endif

    tot_images = tot_images + num_images;
    tot_masks = tot_masks + num_masks;
    
    if DoG_flag
      DoG_subdir = ...
	  [DoG_dir, ...
	   subdir_folder, "/"];
      mkdir(DoG_subdir);
      if mask_flag
	DoG_mask_subdir = ...
	    [DoG_mask_dir, ...
	     subdir_folder, "/"];
	mkdir(DoG_mask_subdir);
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
      endif  %% mask_flag
    endif %% canny_flag

    for i_image = 1 : num_images
      image_name = target_images{i_image};
      sep_ndx = strfind(image_name, filesep);
      base_name = image_name(sep_ndx(end)+1:end);
      image_color = ...
	  imread(image_name);
      image_gray = col2gray(image_color);
      extended_image_gray = addMirrorBC(image_gray, image_margin);

      if mask_flag
	mask_exists = 0;
	mask_name = [target_mask_subdir, base_name];    
	if exist( mask_name, "file")
	  mask_exists = 1;
	  mask_color = ...
	      imread(mask_name);
	  mask_gray = col2gray(mask_color);
	endif %% exist(mask_name)
      endif %% mask_flag

      if DoG_flag 
	[extended_image_DoG, DoG_gray_val] = ...
	    DoG(extended_image_gray, ...
		DoG_struct.amp_center_DoG, ...
		DoG_struct.sigma_center_DoG, ...
		DoG_struct.amp_surround_DoG, ...
		DoG_struct.sigma_surround_DoG);
	image_DoG = ...
	    extended_image_DoG(image_margin+1:end-image_margin, ...
			       image_margin+1:end-image_margin);
	%%[image_DoG] = grayBorder(image_DoG, DoG_margin_width, DoG_gray_val);
	DoG_filename = ...
	    [DoG_subdir, base_name];
	imwrite(uint8(image_DoG), DoG_filename);
	if mask_flag && mask_exists
	  mask_DoG = ...
	      image_DoG .* (mask_gray > 0) + ...
	      DoG_gray_val .* (mask_gray == 0);
	  DoG_mask_filename = ...
	      [DoG_mask_subdir, base_name];
	  imwrite(uint8(mask_DoG), DoG_mask_filename);
	endif  %% mask_flag
      endif
      
      if canny_flag
	[extended_image_canny, image_orient] = ...
	    canny(extended_image_gray, canny_struct.sigma_canny);
	%% [image_canny] = grayBorder(image_canny, canny_margin_width, canny_gray_val);
	image_canny = ...
	    extended_image_canny(image_margin+1:end-image_margin, ...
				 image_margin+1:end-image_margin);
	canny_filename = ...
	    [canny_subdir, base_name];
	imwrite(uint8(image_canny), canny_filename);
	if mask_flag && mask_exists
	  canny_gray_val = 0;
	  mask_canny = ...
	      image_canny .* (mask_gray > 0) + ...
	      canny_gray_val .* (mask_gray == 0);
	  canny_mask_filename = ...
	      [canny_mask_subdir, base_name];
	  imwrite(uint8(mask_canny), canny_mask_filename);
	endif  %% mask_flag
      endif %% canny_flag

    endfor %% i_image

  endfor  %% i_subdir

  disp(["tot_images = ", ...
	num2str(tot_images)]);
  disp(["tot_masks = ", ...
	num2str(tot_masks)]);
  
  end_time = time();
  tot_time = end_time - begin_time;
  disp(["tot_time = ", num2str(tot_time)]);


endfunction %% imageNetEdgeFilter