function [image_info] = imageNetStandardize(original_pathname, xml_pathname)

  global VERBOSE_FLAG
  if ~exist("VERBOSE_FLAG") || isempty(VERBOSE_FLAG)
    VERBOSE_FLAG = 0;
  endif
  if VERBOSE_FLAG
    disp([" original_name = ", original_pathname]);
  endif

  global UNAVAILABLE_INFO

  global STANDARD_DIR
  global ANNOTATION_DIR
  global MASKS_DIR
  global IMAGE_RESIZE
  global RESIZE_FLAG
  global IMAGE_TYPE
  global GRABCUT_FLAG

  image_info = struct;
  image_info.image_flag = 0;
  image_info.mask_flag = 0;
  image_info.Height = [];
  image_info.Width = [];
  image_info.BB_scale_x = 1;
  image_info.BB_scale_y = 1;

  original_filename = strFolderFromPath(original_pathname);
  base_name = strRemoveExtension(original_filename);

  try
    %%[original_image, original_map, original_alpha] = imread(original_pathname);
    original_image = imread(original_pathname);
  catch 
    disp(["failed imageNetStandardized::imread: ", original_pathname]);
    return;
  end
  image_info.Height = size(original_image, 1);
  image_info.Width = size(original_image, 2);
  image_info.NumChannels = size(original_image, 3);

  %% check if image unavaliable
  %% original_info = imfinfo(original_pathname);  %% imfinfo currently broken
  [unavailable_flag, unavailable_info] = ...
      imageNetUnavailableFLCKR2(image_info, original_image, UNAVAILABLE_INFO);
  if unavailable_flag
    disp(["imageNetStandardized::unavailable_flag = ", num2str(unavailable_flag)]);
    return;
  endif
  %% ignore transparancy channel: 4th "color")
  if size(original_image,3) >= 3
    original_image = original_image(:,:,1:3);
  endif
  
  if RESIZE_FLAG
      [pad_image] = ...
          imageNetPad2(original_image, ...
              image_info, ...
              IMAGE_RESIZE);
  else
    pad_image = original_image;
  endif

  if isempty(pad_image)
    disp(["failed imageNetStandardized::imageNetPad2: ", original_pathname]);
    return;
  endif

  standard_name = [STANDARD_DIR, base_name, IMAGE_TYPE];
  try
    imwrite(pad_image, standard_name);
  catch
    disp(["failed imageNetStandardized::imwrite: ", standard_name]);
    return;
  end
  image_info.image_flag = 1;
  
  if ~isempty(xml_pathname) && exist(xml_pathname, "file")
    if VERBOSE_FLAG
      disp(["xml_file = ", xml_pathname]);
    endif
    [mask_image, BB_scale_x, BB_scale_y] = ...
	imageNetMaskBB2(original_image, original_pathname, image_info, xml_pathname);
    if isempty(mask_image)
      return;
    endif

    if RESIZE_FLAG
        [pad_mask_image] = ...
        imageNetPad2(mask_image, ...
                image_info, ...
                IMAGE_RESIZE);
    else
        pad_mask_image = mask_image;
    endif

    if isempty(pad_mask_image)
      disp(["failed imageNetStandardized::imageNetPad2: ", xml_pathname]);
      return;
    endif
    mask_name = [MASKS_DIR, base_name, ".png"];
    try
      imwrite(pad_mask_image, mask_name);
    catch
      disp(["failed imageNetStandardized::imwrite: ", mask_name]);
      return;
    end
    image_info.mask_flag = 1;
    image_info.BB_scale_x = BB_scale_x;
    image_info.BB_scale_y = BB_scale_y;
  endif %% isempty(xml_pathname)

endfunction %% imageNetStandardize

