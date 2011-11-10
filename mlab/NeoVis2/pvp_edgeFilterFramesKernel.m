function [status_info] = pvp_edgeFilterFramesKernel(frame_pathname)

  global DoG_flag
  global canny_flag
  global DoG_dir
  global DoG_struct
  global canny_dir
  global canny_struct
  global image_margin

  global VERBOSE_FLAG

  status_info = struct;
  status_info.unread_flag = 1;
  status_info.DoG_flag = 0;
  status_info.canny_flag = 0;
  status_info.rejected_flag = 1;

  if VERBOSE_FLAG
    disp(["frame_pathname = ", frame_pathname]);
  endif
  frame_filename = strFolderFromPath(frame_pathname);
  status_info.framename = frame_filename;
  status_info.pathname = frame_pathname;
  try
    image_color = ...
	imread(frame_pathname);
  catch
    disp([" failed: pvp_edgeFilterFramesKernel::imread: ", frame_pathname]);
    return;
  end
  status_info.unread_flag = 0;
  image_gray = col2gray(image_color);
  
  status_info.mean = mean(image_gray(:));
  status_info.std = std(image_gray(:));

  extended_image_gray = addMirrorBC(image_gray, image_margin);

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
    DoG_pathname = ...
	[DoG_dir, frame_filename];
    try
      imwrite(uint8(image_DoG), DoG_pathname);
    catch
      disp([" failed: pvp_edgeFilterFramesKernel::imwrite: ", DoG_pathname]);
      return;
    end
    status_info.DoG_flag = 1;
  endif %% DoG_flag
      
  if canny_flag
    canny_gray_val = 0;
    [extended_image_canny, image_orient] = ...
	canny(extended_image_gray, canny_struct.sigma_canny);
    image_canny = ...
	extended_image_canny(image_margin+1:size(extended_image_canny,1)-image_margin, ...
			     image_margin+1:size(extended_image_canny,2)-image_margin);    
    canny_pathname = ...
	[canny_dir, frame_filename];
    try
      imwrite(uint8(image_canny), canny_pathname);
    catch
      disp([" failed: pvp_edgeFilterFramesKernel::imwrite: ", canny_pathname]);
      return;
    end
    status_info.canny_flag = 1;
  endif %% canny_flag



endfunction  %% pvp_edgeFilterFramesKernel