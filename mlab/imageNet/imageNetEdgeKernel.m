function [status_info] = imageNetEdgeKernel(target_pathname, mask_pathname)

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

  status_info = struct;
  status_info.target_flag = 0;
  status_info.mask_flag = 0;
  status_info.DoG_flag = 0;
  status_info.DoG_mask_flag = 0;
  status_info.canny_flag = 0;
  status_info.canny_mask_flag = 0;

  if VERBOSE_FLAG
    disp(["target_pathname = ", target_pathname]);
  endif
  target_filename = strFolderFromPath(target_pathname);
  try
    image_color = ...
	imread(target_pathname);
  catch
    disp([" failed: imageNetEdgeExtract::imread: ", target_pathname]);
    return;
  end
  status_info.target_flag = 1;
  image_gray = col2gray(image_color);
  extended_image_gray = addMirrorBC(image_gray, image_margin);

  if mask_flag
    if VERBOSE_FLAG
      disp(["mask_pathname = ", mask_pathname]);
    endif
    mask_exists = 0;
    if exist( mask_pathname, "file")
      if VERBOSE_FLAG
	disp(["mask_flag = ", num2str(mask_flag)]);
      endif
      mask_exists = 1;
      status_info.mask_flag = 1;
      try
	mask_color = ...
	    imread(mask_pathname);
      catch
	disp([" failed: imageNetEdgeExtract::imread: ", mask_pathname]);
	return;
      end
      status_info.mask_flag = 1;
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
	extended_image_DoG(image_margin+1:size(extended_image_DoG,1)-image_margin, ...
			   image_margin+1:size(extended_image_DoG,2)-image_margin);
    %%[image_DoG] = grayBorder(image_DoG, DoG_margin_width, DoG_gray_val);
    DoG_pathname = ...
	[DoG_subdir, target_filename];
    try
      imwrite(uint8(image_DoG), DoG_pathname);
    catch
      disp([" failed: imageNetEdgeExtract::imwrite: ", DoG_pathname]);
      return;
    end
    status_info.DoG_flag = 1;
    if mask_flag && mask_exists
      mask_DoG = ...
	  image_DoG .* (mask_gray > 0) + ...
	  DoG_gray_val .* (mask_gray == 0);
      DoG_mask_pathname = ...
	  [DoG_mask_subdir, target_filename];
      try
	imwrite(uint8(mask_DoG), DoG_mask_pathname);
      catch
	disp([" failed: imageNetEdgeExtract::imwrite: ", DoG_mask_pathname]);
	return;
      end
      if antimask_flag
	antimask_DoG = ...
	    image_DoG .* (mask_gray == 0) + ...
	    DoG_gray_val .* (mask_gray > 0);
	DoG_antimask_pathname = ...
	    [DoG_antimask_subdir, target_filename];
	try
	  imwrite(uint8(antimask_DoG), DoG_antimask_pathname);
	catch
	  disp([" failed: imageNetEdgeExtract::imwrite: ", DoG_antimask_pathname]);
	  return;
	end
      endif
      status_info.DoG_mask_flag = 1;
    endif  %% mask_flag
  endif
      
  if canny_flag
    [extended_image_canny, image_orient] = ...
	canny(extended_image_gray, canny_struct.sigma_canny);
    %% [image_canny] = grayBorder(image_canny, canny_margin_width, canny_gray_val);
    image_canny = ...
	extended_image_canny(image_margin+1:size(extended_image_canny,1)-image_margin, ...
			     image_margin+1:size(extended_image_canny,2)-image_margin);
    canny_pathname = ...
	[canny_subdir, target_filename];
    try
      imwrite(uint8(image_canny), canny_pathname);
    catch
      disp([" failed: imageNetEdgeExtract::imwrite: ", canny_pathname]);
      return;
    end
    status_info.canny_flag = 1;
    if mask_flag && mask_exists
      canny_gray_val = 0;
      mask_canny = ...
	  image_canny .* (mask_gray > 0) + ...
	  canny_gray_val .* (mask_gray == 0);
      canny_mask_pathname = ...
	  [canny_mask_subdir, target_filename];
      try
	imwrite(uint8(mask_canny), canny_mask_pathname);
      catch
	disp([" failed: imageNetEdgeExtract::imwrite: ", canny_mask_pathname]);
	return;
      end
      if antimask_flag
	antimask_canny = ...
	    image_canny .* (mask_gray == 0) + ...
	    canny_gray_val .* (mask_gray > 0);
	canny_antimask_pathname = ...
	    [canny_antimask_subdir, target_filename];
	try
	  imwrite(uint8(antimask_canny), canny_antimask_pathname);
	catch
	  disp([" failed: imageNetEdgeExtract::imwrite: ", canny_antimask_pathname]);
	  return;
	end
      endif
      status_info.canny_mask_flag = 1;
    endif %% mask_flag
  endif %% canny_flag



endfunction  %% imageNetEdgeExtract