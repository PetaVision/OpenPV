function [tot_images ...
	  tot_masks ...
	  tot_discarded ...
	  num_BB_mismatch ...
	  ave_height ...
	  ave_width ...
	  std_height ...
	  std_width ...
	  tot_images_landscape ...
	  tot_masks_landscape ...
	  ave_height_landscape ...
	  ave_width_landscape ...
	  std_height_landscape ...
	  std_width_landscape ...
	  tot_time] = ...
      imageNetResize2(imageNet_path, ...
		      image_resize, ...
		      object_list, ...
		      image_type, ...
		      grabcut_flag, ...
		      num_procs, ...
		      BB_only_flag)

  %% reformat directory of image net images to be of uniform size with
  %% mirror BCs for padding if necessary
  %% also produces masked images from bounding
  %% box data
  %% uses parcellfun to execute in parallel

  if nargin < 1 || ~exist(imageNet_path) || isempty(imageNet_path)
    imageNet_path = "/Users/dylanpaiton/Documents/Work/LANL/Image_Net/Database/";
  endif
  if nargin < 2 || ~exist(image_resize) || isempty(image_resize)
    image_resize = [0 0];
  endif
  if nargin < 3 || ~exist(object_list) || isempty(object_list)
    object_list{1} = "flag"; 
  endif
  if nargin < 4 || ~exist(image_type) || isempty(image_type)
    image_type = ".png";  %% 
  endif
  if nargin < 5 || ~exist(grabcut_flag) || isempty(grabcut_flag)
    grabcut_flag = 0;  %% uses openCV segmentation algorithm to focus bounding boxes
  endif
  if nargin < 6 || ~exist(num_procs) || isempty(num_procs)
    num_procs = 1;  %% number of processors to use
  endif
  if nargin < 7 || ~exist(BB_only_flag) || isempty(BB_only_flag)
    BB_only_flag = 1;  %% (-)1 -> only process images with(without) bounding boxes
  endif

  global NUM_FIGS
  NUM_FIGS = 0;
  
  global VERBOSE_FLAG
  if ~exist("VERBOSE_FLAG") || isempty(VERBOSE_FLAG)
    VERBOSE_FLAG = 1;
  endif

  global UNAVAILABLE_INFO
  UNAVAILABLE_INFO = [];

  global STANDARD_DIR
  global ANNOTATION_DIR
  global MASKS_DIR
  global TMP_DIR
  global TMP_MASK_DIR
  global IMAGE_RESIZE
  global IMAGE_TYPE
  global GRABCUT_FLAG
  global RESIZE_FLAG

  RESIZE_FLAG = prod (image_resize) > 0;

  IMAGE_RESIZE = image_resize;
  IMAGE_TYPE = image_type;
  GRABCUT_FLAG = grabcut_flag;

  begin_time = time();
  more off

  output_path = [imageNet_path, "imageNetMasks", filesep];
  mkdir(output_path);

  TMP_DIR = [output_path, "tmp", filesep];
  mkdir(TMP_DIR);

  TMP_MASK_DIR = [output_path, "tmpMask", filesep];
  mkdir(TMP_MASK_DIR);

  matFiles_dir = ...
      [output_path, "matFiles", filesep];
  mkdir(matFiles_dir); 

  disp("");
  disp(["num_procs = ", num2str(num_procs)]);

  num_objects = length(object_list);
  for i_object = 1 : num_objects
    object_name = object_list{i_object};
    disp(["object_name = ", object_name]);
    object_dir = ...
	[ imageNet_path, "img", filesep, object_name, filesep];
    if ~exist(object_dir, "dir")
      error(["object_dir does not exist: ", object_dir]);
    endif

    mkdir([ output_path, "standard" ]);
    standard_parent_dir = ...
	[ output_path, "standard", filesep, object_name, filesep ];
    mkdir(standard_parent_dir);
    mkdir([ output_path, "masks" ]);
    masks_parent_dir = ...
	[ output_path, "masks", filesep, object_name, filesep ];
    mkdir(masks_parent_dir);
    tot_images = zeros(1,num_objects);
    tot_masks = zeros(1,num_objects);
    tot_discarded = zeros(1,num_objects);
    num_BB_mismatch = zeros(1,num_objects);
    ave_height = zeros(1,num_objects);
    ave_width = zeros(1,num_objects);
    std_height = zeros(1,num_objects);
    std_width = zeros(1,num_objects);
    tot_images_landscape = zeros(1,num_objects);
    tot_masks_landscape = zeros(1,num_objects);
    ave_height_landscape = zeros(1,num_objects);
    ave_width_landscape = zeros(1,num_objects);
    std_height_landscape = zeros(1,num_objects);
    std_width_landscape = zeros(1,num_objects);
    
    %% imageNet sub directories are drawn from wordNet categories
    subdir_pathnames = glob([object_dir, "*"]);
    num_subdirs = length(subdir_pathnames);
    disp(["num_subdirs = ", num2str(num_subdirs)]);
    subdir_foldernames = cellfun(@strFolderFromPath, subdir_pathnames, "UniformOutput", false);
    for i_subdir = 1 : num_subdirs %% 
      disp(["i_subdir = ", num2str(i_subdir)]);
      disp(["subdir_pathname = ", subdir_pathnames{i_subdir}]);
      images_path = ...
	  [object_dir, ...
	   subdir_foldernames{i_subdir}, filesep];
      annotation_path = ...
	  [object_dir, ...
	   subdir_foldernames{i_subdir}, filesep, ...
	   "Annotation", filesep];
      annotation_flag = exist(annotation_path, "dir");
      if ~annotation_flag && BB_only_flag == 1
	continue;
      elseif annotation_flag && BB_only_flag == -1
	continue;
      endif
      subdir_ids = [];
      subdir_id = [];
      num_subdir_ids = 0;
      if ~annotation_flag
	if VERBOSE_FLAG
	  warning(["annoation_path does not exist: ", annotation_path]);
	endif
      else
	  subdir_ids = glob([annotation_path, filesep, "n[0-9]*"]);
	  num_subdir_ids = length(subdir_ids);
	  subdir_id = subdir_ids{num_subdir_ids};
	  base_id = strFolderFromPath(subdir_id);
	  annotation_dir = [annotation_path, base_id, filesep];
      endif
      standard_dir = ...
	  [standard_parent_dir, ...
	   subdir_foldernames{i_subdir}, filesep];
      STANDARD_DIR = standard_dir;
      mkdir(standard_dir); %% does not clobber
      masks_dir = ...
	  [masks_parent_dir, ...
	   subdir_foldernames{i_subdir}, filesep];
      MASKS_DIR = masks_dir;
      mkdir(masks_dir); 

      original_pathnames = glob([images_path, "n*.*"]);
      num_images = length(original_pathnames);
      if VERBOSE_FLAG
	disp(["num_images = ", num2str(num_images)]);
      endif
      if annotation_flag
	original_filenames = cellfun(@strFolderFromPath, original_pathnames, "UniformOutput", false);
	base_filenames = cellfun(@strRemoveExtension, original_filenames, "UniformOutput", false);
	xml_filenames = strcat(base_filenames, repmat(".xml", num_images, 1));
	xml_pathnames = strcat(annotation_dir, xml_filenames);
      else
	xml_pathnames = cell(num_images,1);
      endif

      %%disp(["size(original_pathnames) = ", num2str(size(original_pathnames))]);
      %%disp(["size(xml_pathnames) = ", num2str(size(xml_pathnames))]);
      if num_procs > 1
	[image_info] = ...
	    parcellfun(num_procs, @imageNetStandardize, original_pathnames, xml_pathnames, ...
		       "UniformOutput", false);
      else
	[image_info] = ...
	    cellfun(@imageNetStandardize, original_pathnames, xml_pathnames, ...
		       "UniformOutput", false);
      endif

      tot_discarded(i_object) = ...
	  tot_discarded(i_object) + num_images;
      if exist("image_info") && ~isempty(image_info) 
      for i_image = 1 : num_images
	if image_info{i_image}.image_flag == 0
	  continue;
	endif
	tot_images(1,i_object) = ...
	    tot_images(1,i_object) + image_info{i_image}.image_flag;
	ave_height(1,i_object) = ...
	    ave_height(1,i_object) + image_info{i_image}.Height;
	ave_width(1,i_object) = ...
	    ave_width(1,i_object) + image_info{i_image}.Width;
	std_height(1,i_object) = ...
	    std_height(1,i_object) + image_info{i_image}.Height.^2;
	std_width(1,i_object) = ...
	    std_width(1,i_object) + image_info{i_image}.Width.^2;
	if image_info{i_image}.Height < image_info{i_image}.Width
	  tot_images_landscape(1,i_object) = ...
	      tot_images_landscape(1,i_object) + 1;
	  ave_height_landscape(1,i_object) = ...
	      ave_height_landscape(1,i_object) + image_info{i_image}.Height;
	  ave_width_landscape(1,i_object) = ...
	      ave_width_landscape(1,i_object) + image_info{i_image}.Width;
	  std_height_landscape(1,i_object) = ...
	      std_height_landscape(1,i_object) + image_info{i_image}.Height.^2;
	  std_width_landscape(1,i_object) = ...
	      std_width_landscape(1,i_object) + image_info{i_image}.Width.^2;
	endif
	if image_info{i_image}.image_flag == 0
	  continue;
	endif
	tot_masks(1,i_object) = ...
	    tot_masks(1,i_object) + image_info{i_image}.mask_flag;
	num_BB_mismatch(i_object) = ...
	    num_BB_mismatch(i_object) + ...
	    ((image_info{i_image}.BB_scale_x ~= 1) || (image_info{i_image}.BB_scale_y ~= 1));	    
      endfor %% i_image
    endif%% exist("image_info")

    endfor %% i_subdir
    
    tot_discarded(i_object) = ...
	tot_discarded(i_object) - tot_images(i_object);
    disp(["tot_discarded = ", ...
	  num2str(tot_discarded(i_object))]);

    ave_height(i_object) = ...
	ave_height(i_object) / tot_images(i_object);
    ave_width(i_object) = ...
	ave_width(i_object) / tot_images(i_object);
    std_height(i_object) = ...
	sqrt((std_height(i_object) / tot_images(i_object)) - ...
	     ave_height(i_object).^2);
    std_width(i_object) = ...
	sqrt((std_width(i_object) / tot_images(i_object)) - ...
	     ave_width(i_object).^2);
    disp(["tot_images = ", ...
	  num2str(tot_images(i_object))]);
    disp(["tot_masks = ", ...
	  num2str(tot_masks(i_object))]);
    disp(["ave_height = ", ...
	  num2str(ave_height(i_object)), " +/- ", num2str(std_height(i_object))]);
    disp(["ave_width = ", ...
	  num2str(ave_width(i_object)), " +/- ", num2str(std_width(i_object))]);

    ave_height_landscape(i_object) = ...
	ave_height_landscape(i_object) / tot_images_landscape(i_object);
    ave_width_landscape(i_object) = ...
	ave_width_landscape(i_object) / tot_images_landscape(i_object);
    std_height_landscape(i_object) = ...
	sqrt((std_height_landscape(i_object) / tot_images_landscape(i_object)) - ...
	     ave_height_landscape(i_object).^2);
    std_width_landscape(i_object) = ...
	sqrt((std_width_landscape(i_object) / tot_images_landscape(i_object)) - ...
	     ave_width_landscape(i_object).^2);
    disp(["tot_images_landscape = ", ...
	  num2str(tot_images_landscape(i_object))]);
    disp(["tot_masks_landscape = ", ...
	  num2str(tot_masks_landscape(i_object))]);
    disp(["ave_height_landscape = ", ...
	  num2str(ave_height_landscape(i_object)), " +/- ", ...
	  num2str(std_height_landscape(i_object))]);
    disp(["ave_width_landscape = ", ...
	  num2str(ave_width_landscape(i_object)), " +/- ", ...
	  num2str(std_width_landscape(i_object))]);
    
  endfor  %% i_object
  
  matFile_name = [matFiles_dir, date, ".mat"];
  save( "-binary", matFile_name, ...
      "tot_images", ...
      "tot_masks", ...
      "tot_discarded", ...
      "num_BB_mismatch", ...
      "ave_height", ...
      "ave_width", ...
      "std_height", ...
      "std_width", ...
      "tot_images_landscape", ...
      "tot_masks_landscape", ...
      "ave_height_landscape", ...
      "ave_width_landscape", ...
      "std_height_landscape", ...
      "std_width_landscape");
  end_time = time();
  tot_time = end_time - begin_time;
  disp(["tot_time = ", num2str(tot_time)]);

  rmdir(TMP_DIR);
  rmdir(TMP_MASK_DIR);

endfunction%% imageNetResize2
