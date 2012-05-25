function [] = ...
      pvp_makeTargetMask(frame_pathname, truth_CSV_struct, DCR_CSV_struct);

  global target_mask_dir
  global distractor_mask_dir
  global frame_mask_dir
  global pvp_max_patch_size
  global pvp_min_patch_size
  global NFEATURES NCOLS NROWS N

  %% skip frame if not annotated
  if isempty(truth_CSV_struct)
    return;
  endif

  %% path to generic image processing routines
  img_proc_dir = "~/workspace-indigo/PetaVision/mlab/imgProc/";
  addpath(img_proc_dir);
  
  %% path to string manipulation kernels for use with parcellfun
  str_kernel_dir = "~/workspace-indigo/PetaVision/mlab/stringKernels/";
  addpath(str_kernel_dir);

  %%keyboard;
  bounding_quad = 0;
  largest_rect = 1;
  mask_method = bounding_quad;
  
%%disp(frame_pathname);
  num_BB = length(truth_CSV_struct);
  disp(["num_BB = ", num2str(num_BB)]);
  num_DCR = length(DCR_CSV_struct);
  pvp_image = imread(frame_pathname);
  size_image = size(pvp_image);
  target_mask = uint8(zeros(size_image(1:2)));
  for i_BB = 1 : num_BB
    if isempty(truth_CSV_struct{i_BB})
      continue;
    endif
    x1_truth = truth_CSV_struct{i_BB}.BoundingBox_X1;
    x2_truth = truth_CSV_struct{i_BB}.BoundingBox_X2;
    x3_truth = truth_CSV_struct{i_BB}.BoundingBox_X3;
    x4_truth = truth_CSV_struct{i_BB}.BoundingBox_X4;
    y1_truth = truth_CSV_struct{i_BB}.BoundingBox_Y1;
    y2_truth = truth_CSV_struct{i_BB}.BoundingBox_Y2;
    y3_truth = truth_CSV_struct{i_BB}.BoundingBox_Y3;
    y4_truth = truth_CSV_struct{i_BB}.BoundingBox_Y4;
      
    x_BB_min = min([x1_truth; ...
		    x2_truth; ...
		    x3_truth; ...
		    x4_truth]);
    if x_BB_min <= 0
      x_BB_min = 1;
    endif
    x_BB_max = max([x1_truth; ...
		    x2_truth; ...
		    x3_truth; ...
		    x4_truth]);
    if x_BB_max > NCOLS
      x_BB_max = NCOLS;
    endif
    y_BB_min = min([y1_truth; ...
		    y2_truth; ...
		    y3_truth; ...
		    y4_truth]);
    if y_BB_min <= 0
      y_BB_min = 1;
    endif
    y_BB_max = max([y1_truth; ...
		    y2_truth; ...
		    y3_truth; ...
		    y4_truth]);
    if y_BB_max > NROWS
      y_BB_max = NROWS;
    endif
    
    %%keyboard;
    if mask_method == largest_rect
      x_poly = [x_BB_min, x_BB_max, x_BB_max, x_BB_min];
      y_poly = [y_BB_min, y_BB_min, y_BB_max, y_BB_max];
    elseif (mask_method == bounding_quad)
      x_poly = [x1_truth, x2_truth, x3_truth, x4_truth];
      y_poly = [y1_truth, y2_truth, y3_truth, y4_truth];
    endif
    target_mask_tmp = poly2mask(x_poly, y_poly, size_image(1), size_image(2));
    target_mask_tmp = uint8(target_mask_tmp);
    target_mask(find(target_mask_tmp)) = 255;
  endfor  %% i_BB


  DCR_mask = uint8(zeros(size_image(1:2)));
  disp(["num_DCR = ", num2str(num_DCR)]);
  for i_DCR = 1 : num_DCR
    if isempty(DCR_CSV_struct{i_DCR})
      continue;
    endif
    x1_DCR = DCR_CSV_struct{i_DCR}.BoundingBox_X1;
    x2_DCR = DCR_CSV_struct{i_DCR}.BoundingBox_X2;
    x3_DCR = DCR_CSV_struct{i_DCR}.BoundingBox_X3;
    x4_DCR = DCR_CSV_struct{i_DCR}.BoundingBox_X4;
    y1_DCR = DCR_CSV_struct{i_DCR}.BoundingBox_Y1;
    y2_DCR = DCR_CSV_struct{i_DCR}.BoundingBox_Y2;
    y3_DCR = DCR_CSV_struct{i_DCR}.BoundingBox_Y3;
    y4_DCR = DCR_CSV_struct{i_DCR}.BoundingBox_Y4;
      
    x_DCR_min = min([x1_DCR; ...
		    x2_DCR; ...
		    x3_DCR; ...
		    x4_DCR]);
    if x_DCR_min <= 0
      x_DCR_min = 1;
    endif
    x_DCR_max = max([x1_DCR; ...
		    x2_DCR; ...
		    x3_DCR; ...
		    x4_DCR]);
    if x_DCR_max > NCOLS
      x_DCR_max = NCOLS;
    endif
    y_DCR_min = min([y1_DCR; ...
		    y2_DCR; ...
		    y3_DCR; ...
		    y4_DCR]);
    if y_DCR_min <= 0
      y_DCR_min = 1;
    endif
    y_DCR_max = max([y1_DCR; ...
		    y2_DCR; ...
		    y3_DCR; ...
		    y4_DCR]);
    if y_DCR_max > NROWS
      y_DCR_max = NROWS;
    endif
    
    %%keyboard;
    if mask_method == largest_rect
      x_poly = [x_DCR_min, x_DCR_max, x_DCR_max, x_DCR_min];
      y_poly = [y_DCR_min, y_DCR_min, y_DCR_max, y_DCR_max];
    elseif (mask_method == bounding_quad)
      x_poly = [x1_DCR, x2_DCR, x3_DCR, x4_DCR];
      y_poly = [y1_DCR, y2_DCR, y3_DCR, y4_DCR];
    endif
    DCR_mask_tmp = poly2mask(x_poly, y_poly, size_image(1), size_image(2));
    DCR_mask_tmp = uint8(DCR_mask_tmp);
    DCR_mask(find(DCR_mask_tmp)) = 255;
  endfor  %% i_DCR
      
  mask_pathname = strExtractPath(frame_pathname);
  mask_parent = strFolderFromPath(mask_pathname);
  mask_parent_root = mask_parent(1:(length(mask_parent)-1));
  mask_imagename = strFolderFromPath(frame_pathname);
  mask_rootname = strRemoveExtension(mask_imagename);
  mask_title = [mask_parent_root, "_", mask_rootname, "_", "mask", ".png"];

  target_mask_pathname = [target_mask_dir, "target_", mask_title];
  %%disp(target_mask_pathname);
  imwrite(target_mask, target_mask_pathname);
  
  distractor_mask = max(target_mask(:)) + min(target_mask(:)) - target_mask;
  distractor_mask(DCR_mask == 255) = 0;
  distractor_mask_pathname = [distractor_mask_dir, "distractor_", mask_title];
  %%disp(distractor_mask_pathname);
  imwrite(distractor_mask, distractor_mask_pathname);
  
  frame_mask_pathname = [frame_mask_dir, mask_title];
  %%disp(frame_mask_pathname);
  imwrite(pvp_image, frame_mask_pathname);
  
endfunction %% pvp_makeTargetMask

