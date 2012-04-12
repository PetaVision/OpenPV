function [num_target_chips, num_distractor_chips] = ...
      pvp_bootstrapChips(frame_pathname, hit_list, miss_list);

  global target_bootstrap_dir
  global distractor_bootstrap_dir
  global pvp_max_patch_size
  global pvp_min_patch_size
  global NFEATURES NCOLS NROWS N

  %% path to generic image processing routines
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);
  
  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);
    
  num_miss_BB = length(miss_list);
  num_distractor_chips = 0;
  if num_miss_BB > 0
    ave_miss_confidence = 0;
    std_miss_confidence = 0;
    for i_miss_BB = 1 : num_miss_BB
      ave_miss_confidence = ave_miss_confidence + miss_list{i_miss_BB}.Confidence;
      std_miss_confidence = std_miss_confidence + miss_list{i_miss_BB}.Confidence.^2;
    endfor
    ave_miss_confidence = ave_miss_confidence / num_miss_BB;
    std_miss_confidence = sqrt((std_miss_confidence / num_miss_BB) - (ave_miss_confidence.^2));
    for i_miss_BB = 1 : num_miss_BB
      if miss_list{i_miss_BB}.Confidence < ave_miss_confidence
	continue;
      endif
      x_BB_min = miss_list{i_miss_BB}.patch_X1;
      x_BB_max = miss_list{i_miss_BB}.patch_X2;
      y_BB_min = miss_list{i_miss_BB}.patch_Y1;
      y_BB_max = miss_list{i_miss_BB}.patch_Y3;

      chip_width = x_BB_max - x_BB_min + 1;
      chip_height = y_BB_max - y_BB_min + 1;
      chip_size = ceil([chip_height, chip_width]);
      if any(chip_size <= pvp_min_patch_size)
	continue;
      endif
      pvp_image = imread(frame_pathname);
      num_subchips = ceil(chip_size ./ pvp_max_patch_size);
      for i_subchip_row = 1 : num_subchips(1)
	y_BB_min2 = ceil(y_BB_min + (i_subchip_row - 1 ) * chip_size(1) / num_subchips(1));
	y_BB_min2 = max(1, y_BB_min2);
	y_BB_max2 = floor(y_BB_min + ((i_subchip_row) * chip_size(1) / num_subchips(1)) - 1);
	y_BB_max2 = min(NROWS, y_BB_max2);
	for j_subchip_col = 1 : num_subchips(2)
	  x_BB_min2 = ceil(x_BB_min + (j_subchip_col - 1 ) * chip_size(2) / num_subchips(2));
	  x_BB_min2 = max(1, x_BB_min2);
	  x_BB_max2 = floor(x_BB_min + ((j_subchip_col) * chip_size(2) / num_subchips(2)) - 1);
	  x_BB_max2 = min(NCOLS, x_BB_max2);
	  miss_chip = pvp_image(y_BB_min2:y_BB_max2, x_BB_min2:x_BB_max2);
	  miss_chip_pathname = strExtractPath(frame_pathname);
	  miss_chip_parent = strFolderFromPath(miss_chip_pathname);
	  miss_chip_parent_root = miss_chip_parent(1:(length(miss_chip_parent)-1));
	  miss_chip_imagename = strFolderFromPath(frame_pathname);
	  miss_chip_rootname = strRemoveExtension(miss_chip_imagename);
	  chip_id = i_miss_BB + ...
	      (i_subchip_row - 1) * num_miss_BB + ...
	      (j_subchip_col - 1) * num_subchips(1) * num_miss_BB;
	  miss_chip_title = [miss_chip_parent_root, "_", miss_chip_rootname, "_", num2str(chip_id, "%3.3d"), ".png"];
	  miss_chip_pathname = [distractor_bootstrap_dir, miss_chip_title];
	  imwrite(miss_chip, miss_chip_pathname);
	  num_distractor_chips = num_distractor_chips + 1;
	endfor %% j_subchip_col
      endfor %% i_subchip_row
    endfor  %% i_BB
  endif  %% num_miss_BB > 0

  num_hit_BB = length(hit_list);
  num_target_chips = 0;
  if num_hit_BB > 0
    ave_hit_confidence = 0;
    std_hit_confidence = 0;
    for i_hit_BB = 1 : num_hit_BB
      if isempty(hit_list{i_hit_BB})
	continue;
      endif
      ave_hit_confidence = ave_hit_confidence + hit_list{i_hit_BB}.Confidence;
      std_hit_confidence = std_hit_confidence + hit_list{i_hit_BB}.Confidence.^2;
    endfor
    ave_hit_confidence = ave_hit_confidence / num_hit_BB;
    std_hit_confidence = sqrt((std_hit_confidence / num_hit_BB) - (ave_hit_confidence.^2));
    for i_hit_BB = 1 : num_hit_BB
      if isempty(hit_list{i_hit_BB})
	continue;
      endif
      if hit_list{i_hit_BB}.Confidence > 0 %% ave_hit_confidence
	continue;
      endif
      x_BB_min = hit_list{i_hit_BB}.BoundingBox_X1;
      x_BB_max = hit_list{i_hit_BB}.BoundingBox_X2;
      y_BB_min = hit_list{i_hit_BB}.BoundingBox_Y1;
      y_BB_max = hit_list{i_hit_BB}.BoundingBox_Y3;

      %% expand chip size
      chip_size_x = (x_BB_max - x_BB_min); 
      chip_size_y = (y_BB_max - y_BB_min);
      chip_center_x = (x_BB_max + x_BB_min) / 2; 
      chip_center_y = (y_BB_max + y_BB_min) / 2; 
      target_radius_x = pvp_max_patch_size(2) / 2;
      target_radius_y = pvp_max_patch_size(1) / 2;
      x_BB_min2 = ceil(chip_center_x - target_radius_x);
      x_BB_max2 = floor(chip_center_x + target_radius_x);
      x_BB_max2 = x_BB_max2 - fix(length(x_BB_min2:x_BB_max2) - pvp_max_patch_size(2));
      y_BB_min2 = ceil(chip_center_y - target_radius_y);
      y_BB_max2 = floor(chip_center_y + target_radius_y);
      y_BB_max2 = y_BB_max2 - fix(length(y_BB_min2:y_BB_max2) - pvp_max_patch_size(1));
      if x_BB_min2 < 1
	pad_x_min = 1 - x_BB_min2;
	x_BB_min2 = 1;
      else
	pad_x_min = 0;
      endif
      if y_BB_min2 < 1
	pad_y_min = 1 - y_BB_min2;
	y_BB_min2 = 1;
      else
	pad_y_min = 0;
      endif
      if x_BB_max2 > NCOLS
	pad_x_max = x_BB_max2 - NCOLS;
	x_BB_max2 = NCOLS;
      else
	pad_x_max = 0;
      endif
      if y_BB_max2 > NROWS
	pad_y_max = y_BB_max2 - NROWS;
	y_BB_max2 = NROWS;
      else
	pad_y_max = 0;
      endif
      
      %%keyboard;
      pvp_image = imread(frame_pathname);
      hit_chip = uint8(zeros(pvp_max_patch_size));
      if any(size([1+pad_y_min:pvp_max_patch_size(1)-pad_y_max, 1+pad_x_min:pvp_max_patch_size(2)-pad_x_max]) ~= ...
	     size([y_BB_min2:y_BB_max2, x_BB_min2:x_BB_max2]))
	disp(["size hit_chip = ", ...
	      num2str(size([1+pad_y_min:pvp_max_patch_size(1)-pad_y_max, 1+pad_x_min:pvp_max_patch_size(2)-pad_x_max]))]);
	disp(["size BB = ", num2str(size([y_BB_min2:y_BB_max2, x_BB_min2:x_BB_max2]))]);
	keyboard;
	error(["size of hit_chip != size of bounding box: ", frame_pathname]);
      endif
      hit_chip(1+pad_y_min:pvp_max_patch_size(1)-pad_y_max, 1+pad_x_min:pvp_max_patch_size(2)-pad_x_max) = ...
	  pvp_image(y_BB_min2:y_BB_max2, x_BB_min2:x_BB_max2);
      if pad_x_min > 0
	hit_chip(1+pad_y_min:pvp_max_patch_size(1)-pad_y_max, 1:pad_x_min) = ...
	    fliplr(pvp_image(y_BB_min2:y_BB_max2, 1:pad_x_min));
      endif
      if pad_x_max > 0
	hit_chip(1+pad_y_min:pvp_max_patch_size(1)-pad_y_max, pvp_max_patch_size(2)-pad_x_max:pvp_max_patch_size(2)) = ...
	    fliplr(pvp_image(y_BB_min2:y_BB_max2, NCOLS-pad_x_max:NCOLS));
      endif
      if pad_y_min > 0
	hit_chip(1:pad_y_min, 1+pad_x_min:pvp_max_patch_size(2)-pad_x_max) = ...
	    flipud(pvp_image(1:pad_y_min, x_BB_min2:x_BB_max2));
      endif
      if pad_y_max > 0
	hit_chip(pvp_max_patch_size(1)-pad_y_max:pvp_max_patch_size(1), 1+pad_x_min:pvp_max_patch_size(2)-pad_x_max) = ...
	    flipud(pvp_image(NROWS-pad_y_max:NROWS, x_BB_min2:x_BB_max2));
      endif
      if pad_x_min > 0 && pad_y_min > 0
	hit_chip(1:pad_y_min, 1:pad_x_min) = ...
	    fliplr(pvp_image(1:pad_y_min, 1:pad_x_min));
      endif
      if pad_x_max > 0 && pad_y_min > 0
	hit_chip(1:pad_y_min, pvp_max_patch_size(2)-pad_x_max:pvp_max_patch_size(2)) = ...
	    fliplr(pvp_image(1:pad_y_min, pvp_max_patch_size(2)-pad_x_max:pvp_max_patch_size(2)));
      endif
      if pad_x_min > 0 && pad_y_max > 0
	hit_chip(pvp_max_patch_size(1)-pad_y_max:pvp_max_patch_size(1), 1:pad_x_min) = ...
	    fliplr(pvp_image(pvp_max_patch_size(1)-pad_y_max:pvp_max_patch_size(1), 1:pad_x_min));
      endif
      if pad_x_max > 0 && pad_y_max > 0
	hit_chip(pvp_max_patch_size(1)-pad_y_max:pvp_max_patch_size(1), ...
		 pvp_max_patch_size(2)-pad_x_max:pvp_max_patch_size(2)) = ...
	    fliplr(pvp_image(pvp_max_patch_size(1)-pad_y_max:pvp_max_patch_size(1), ...
			     pvp_max_patch_size(2)-pad_x_max:pvp_max_patch_size(2)));
      endif
      hit_chip_pathname = strExtractPath(frame_pathname);
      hit_chip_parent = strFolderFromPath(hit_chip_pathname);
      hit_chip_parent_root = hit_chip_parent(1:(length(hit_chip_parent)-1));
      hit_chip_imagename = strFolderFromPath(frame_pathname);
      hit_chip_rootname = strRemoveExtension(hit_chip_imagename);
      chip_id = i_hit_BB;
      hit_chip_title = [hit_chip_parent_root, "_", hit_chip_rootname, "_", num2str(chip_id, "%3.3d"), ".png"];
      hit_chip_pathname = [target_bootstrap_dir, hit_chip_title];
      imwrite(hit_chip, hit_chip_pathname);
      num_distractor_chips = num_target_chips + 1;
    endfor  %% i_BB
  endif  %% num_hit_BB > 0

endfunction %% pvp_bootstrapChips  

