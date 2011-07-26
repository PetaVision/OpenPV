function [tot_images, ...
	  tot_discarded, ...
	  tot_masks, ...
	  num_BB_mismatch, ...
	  ave_height, ...
	  ave_width, ...
	  ave_area] = ...
      imageNetResize(imageNet_path, image_resize, object_list)

  %% reformat directory of image net images to be of uniform size with
  %% gray padding if necessary
  %% also produces masked images from either eye tracking or bounding
  %% box data

  if nargin < 1 || ~exist(imageNet_path) || isempty(imageNet_path)
    imageNet_path = "~/Pictures/imageNet/";
  endif
  if nargin < 2 || ~exist(image_resize) || isempty(image_resize)
    image_resize = [256 256];
  endif
  if nargin < 3 || ~exist(object_name) || isempty(object_name)
    object_list{1} = "automobile";  %% could be a list?
  endif

  global NUM_FIGS
  NUM_FIGS = 0;
  
  num_objects = length(object_list);
  num_BB_mismatch = 0;
  ave_height = 0;
  ave_width = 0;
  ave_area = 0;
  tot_images = 0;
  tot_discarded = 0;
  tot_masks = 0;
  for i_object = 1 : num_objects
    object_name = object_list{i_object};
    
    object_dir = ...
	[ imageNet_path, object_name, "/" ];
    if ~exist(object_dir, "dir")
      error(["object_dir does not exist: ", object_dir]);
    endif
    standard_dir = [object_dir, "standard/"];
    mkdir(standard_dir); %% does not clobber
    masks_dir = [object_dir, "masks/"];
    mkdir(masks_dir); 
    
    %% imageNet directories start with the letter n
    subdir_struct = dir([object_dir,"n*"]);
    num_subdirs = length(subdir_struct);
    
    for i_subdir = 1 : num_subdirs
      if ~strcmp(subdir_struct(i_subdir).name(1),"n")
	continue;
      endif
      disp(["i_subdir = ", num2str(i_subdir)]);
      images_path = ...
	  [object_dir, ...
	   subdir_struct(i_subdir).name, "/", ...
	   subdir_struct(i_subdir).name, "/"];
      annotation_path = ...
	  [object_dir, ...
	   subdir_struct(i_subdir).name, "/", ...
	   "Annotation", "/"];
      annotation_flag = exist(annotation_path, "dir");
      if ~annotation_flag
	warning(["annoation_path does not exist: ", annotation_path]);
      endif
      images_struct = dir([images_path, "*.*"]);
      num_images = length(images_struct);
      for i_image = 1 : num_images
	tot_discarded = tot_discarded + 1;
	disp(["i_image = ", num2str(i_image)]);
	original_name = ...
	    [images_path, images_struct(i_image).name];
	disp(["original_name = ", original_name]);
	base_ndx = strfind(images_struct(i_image).name, ".") - 1;
	base_name = images_struct(i_image).name(1:base_ndx);
	try
	  [original_image, original_map, original_alpha] = imread(original_name);
	catch 
	  continue;
	end
	original_info = imfinfo(original_name);
	if (( original_info.Height < image_resize(1)/(256/194)) || ...
	    (original_info.Width < image_resize(2)/(256/194)))
	  continue;
	endif
	%% ignore transparancy channel: 4th "color")
	if (strcmp( original_info.Format, "GIF" ) || ...
	    strcmp( original_info.Format, "PNG" ))
	  if size(original_image,3) == 4
	    original_image = original_image(:,:,1:3);
	  endif
	endif
	[pad_image] = ...
	    imageNetPad(original_image, ...
			original_info, ...
			image_resize);
	if isempty(pad_image)
	  continue;
	endif
	standard_name = [standard_dir, base_name, ".png"];
	imwrite(pad_image, standard_name);
	tot_images = tot_images + 1;
	ave_height = ave_height + original_info.Height;
	ave_width = ave_width + original_info.Width;
	ave_area = ave_area + original_info.Height * original_info.Width;
	if annotation_flag
	  xml_file = [annotation_path, ...
		      subdir_struct(i_subdir).name, "/", ...
		      base_name, ".xml"];
	  xml_flag = exist(xml_file, "file");
	  if xml_flag
	    %%keyboard
	    [mask_image, BB_mask, BB_scale_x, BB_scale_y] = ...
		imageNetMaskBB(original_image, original_info, xml_file);
	    if isempty(mask_image)
	      continue;
	    endif
	    [pad_mask_image] = ...
		imageNetPad(mask_image, ...
			    original_info, ...
			    image_resize);
	    if isempty(pad_image)
	      continue;
	    endif
	    mask_name = [masks_dir, base_name, ".png"];
	    imwrite(pad_mask_image, mask_name);
	    tot_masks = tot_masks + 1;
	    num_BB_mismatch = ...
		num_BB_mismatch + ((BB_scale_x ~= 1) || (BB_scale_y ~= 1));	    
	  endif
	endif
      endfor %% i_image
    endfor %% i_subdir
    
  endfor  %% i_object
  tot_discarded = tot_discarded - tot_images;
  ave_height = ave_height / tot_images;
  ave_width = ave_width / tot_images;
  ave_area = ave_area / tot_images;
  
  
endfunction%% imageNetResize