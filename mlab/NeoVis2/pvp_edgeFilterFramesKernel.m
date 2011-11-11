function [pvp_status_info] = pvp_edgeFilterFramesKernel(frame_pathname)

  global pvp_DoG_flag
  global canny_flag
  global pvp_DoG_dir
  global DoG_struct
  global canny_dir
  global canny_struct
  global pvp_image_margin

  global VERBOSE_FLAG

  pvp_status_info = struct;
  pvp_status_info.unread_flag = 1;
  pvp_status_info.pvp_DoG_flag = 0;
  pvp_status_info.canny_flag = 0;
  pvp_status_info.rejected_flag = 1;

  if VERBOSE_FLAG
    disp(["frame_pathname = ", frame_pathname]);
  endif
  frame_filename = strFolderFromPath(frame_pathname);
  pvp_status_info.framename = frame_filename;
  pvp_status_info.pathname = frame_pathname;
  try
    image_color = ...
	imread(frame_pathname);
  catch
    disp([" failed: pvp_edgeFilterFramesKernel::imread: ", frame_pathname]);
    return;
  end
  pvp_status_info.unread_flag = 0;
  image_gray = col2gray(image_color);
  
  pvp_status_info.mean = mean(image_gray(:));
  pvp_status_info.std = std(image_gray(:));

  extended_image_gray = addMirrorBC(image_gray, pvp_image_margin);

  if pvp_DoG_flag 
    [extended_image_DoG, DoG_gray_val] = ...
	DoG(extended_image_gray, ...
	    DoG_struct.amp_center_DoG, ...
	    DoG_struct.sigma_center_DoG, ...
	    DoG_struct.amp_surround_DoG, ...
	    DoG_struct.sigma_surround_DoG);
    image_DoG = ...
	extended_image_DoG(pvp_image_margin+1:size(extended_image_DoG,1)-pvp_image_margin, ...
			   pvp_image_margin+1:size(extended_image_DoG,2)-pvp_image_margin);
    DoG_pathname = ...
	[pvp_DoG_dir, frame_filename];
    try
      imwrite(uint8(image_DoG), DoG_pathname);
    catch
      disp([" failed: pvp_edgeFilterFramesKernel::imwrite: ", DoG_pathname]);
      return;
    end
    pvp_status_info.pvp_DoG_flag = 1;
  endif %% pvp_DoG_flag
      
  if canny_flag
    canny_gray_val = 0;
    [extended_image_canny, image_orient] = ...
	canny(extended_image_gray, canny_struct.sigma_canny);
    image_canny = ...
	extended_image_canny(pvp_image_margin+1:size(extended_image_canny,1)-pvp_image_margin, ...
			     pvp_image_margin+1:size(extended_image_canny,2)-pvp_image_margin);    
    canny_pathname = ...
	[canny_dir, frame_filename];
    try
      imwrite(uint8(image_canny), canny_pathname);
    catch
      disp([" failed: pvp_edgeFilterFramesKernel::imwrite: ", canny_pathname]);
      return;
    end
    pvp_status_info.canny_flag = 1;
  endif %% canny_flag



endfunction  %% pvp_edgeFilterFramesKernel