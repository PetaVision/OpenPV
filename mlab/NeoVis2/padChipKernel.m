function [status_info] = padChipKernel(target_pathname)

  global DoG_flag
  global canny_flag
  global DoG_dir
  global DoG_struct
  global canny_dir
  global canny_struct
  global image_margin
  global pad_size

  global VERBOSE_FLAG

  status_info = struct;
  status_info.target_flag = 0;
  status_info.DoG_flag = 0;
  status_info.canny_flag = 0;

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
    pad_DoG = padGray(image_DoG, pad_size, DoG_gray_val);

    %% blend edge artifacts
    num_blend = 8;
    image_DoG_mirrorBC = image_DoG;
    pad_DoG_blend = pad_DoG;
    mirror_width = 1;
    for i_blend = 1 : num_blend
      image_DoG_mirrorBC = addMirrorBC(image_DoG, mirror_width*i_blend);
      pad_DoG_mirror = padGray(image_DoG_mirrorBC, pad_size, DoG_gray_val);
      pad_DoG_blend = ...
	  pad_DoG_blend * (1/2) + ...
	  pad_DoG_mirror * (1/2);
    endfor %% i_blend

    DoG_pathname = ...
	[DoG_dir, target_filename];
    try
      imwrite(uint8(pad_DoG_blend), DoG_pathname);
    catch
      disp([" failed: imageNetEdgeExtract::imwrite: ", DoG_pathname]);
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
    pad_canny = padGray(image_canny, pad_size, canny_gray_val);

    %% blend edge artifacts
    num_blend = 4;
    image_canny_mirrorBC = image_canny;
    pad_canny_blend = pad_canny;
    mirror_width = 2;
    for i_blend = 1 : num_blend
      image_canny_mirrorBC = addMirrorBC(image_canny, mirror_width*i_blend);
      pad_canny_mirror = padGray(image_canny_mirrorBC, pad_size, canny_gray_val);
      pad_canny_blend = ...
	  pad_canny_blend * (1/2) + ...
	  pad_canny_mirror * (1/2);
    endfor %% i_blend
    
    canny_pathname = ...
	[canny_dir, target_filename];
    try
      imwrite(uint8(pad_canny_blend), canny_pathname);
    catch
      disp([" failed: imageNetEdgeExtract::imwrite: ", canny_pathname]);
      return;
    end
    status_info.canny_flag = 1;
  endif %% canny_flag



endfunction  %% padChipKernel