function [status_info] = padChipKernel(target_pathname)

  global DoG_flag
  global canny_flag
  global DoG_dir
  global DoG_struct
  global canny_dir
  global canny_struct
  global image_margin
  global pad_size
  global cropped_dir

  global VERBOSE_FLAG

  status_info = struct;
  status_info.target_flag = 0;
  status_info.DoG_flag = 0;
  status_info.canny_flag = 0;

  if VERBOSE_FLAG
    disp(["target_pathname = ", target_pathname]);
  endif
  target_filename = strFolderFromPath(target_pathname);
  status_info.chipname = target_filename;
  status_info.pathname = target_pathname;
  try
    image_color = ...
	imread(target_pathname);
  catch
    disp([" failed: imageNetEdgeExtract::imread: ", target_pathname]);
    return;
  end
  status_info.target_flag = 1;
  image_gray = col2gray(image_color);
  %%keyboard;

  status_info.mean = mean(image_gray(:));
  status_info.std = std(image_gray(:));

  status_info.original_size = size(image_color);  

  %% crop artifacts arising from image borders
  border_artifact_thresh = 1.25;
  status_info.num_border_artifact_top = 0;
  while (mean(squeeze(image_gray(2,:))) - mean(squeeze(image_gray(1,:)))) / ...
	(status_info.std + (status_info.std==0)) > ...
	border_artifact_thresh
    status_info.num_border_artifact_top = status_info.num_border_artifact_top + 1;
    image_gray = image_gray(2:end,:);
    image_color = image_color(2:end,:,:);
  endwhile
  status_info.num_border_artifact_bottom = 0;
  while (mean(squeeze(image_gray(end-1,:))) - mean(squeeze(image_gray(end,:)))) / ...
	(status_info.std + (status_info.std==0)) > ...
	border_artifact_thresh
    status_info.num_border_artifact_bottom = status_info.num_border_artifact_bottom + 1;
    image_gray = image_gray(1:end-1,:);
    image_color = image_gray(1:end-1,:,:);
  endwhile
  status_info.num_border_artifact_left = 0;
  while (mean(squeeze(image_gray(:,2))) - mean(squeeze(image_gray(:,1)))) / ...
	(status_info.std + (status_info.std==0)) > ...
	border_artifact_thresh
    status_info.num_border_artifact_left = status_info.num_border_artifact_left + 1;
    image_gray = image_gray(:,2:end);
    image_color = image_color(:,2:end,:);
  endwhile
  status_info.num_border_artifact_right = 0;
  while (mean(squeeze(image_gray(:, end-1))) - mean(squeeze(image_gray(:,end)))) / ...
	(status_info.std + (status_info.std==0)) > ...
	border_artifact_thresh
    status_info.num_border_artifact_right = status_info.num_border_artifact_right + 1;
    image_gray = image_gray(:,1:end-1);
    image_color = image_color(:,1:end-1,:);
  endwhile
  status_info.cropped_size = size(image_gray);
  if any(status_info.cropped_size ~= status_info.original_size(1:2))
    cropped_pathname = [cropped_dir, status_info.chipname];
    imwrite(image_color, cropped_pathname);
  endif

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
	  pad_DoG_blend * (i_blend/num_blend) + ...
	  pad_DoG_mirror * ((num_blend-i_blend)/num_blend);
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
	  pad_canny_blend * (i_blend/num_blend) + ...
	  pad_canny_mirror * ((num_blend-i_blend)/num_blend);
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