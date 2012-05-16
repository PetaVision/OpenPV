function [pvp_num_active_BB_mask, ...
	  pvp_num_active_BB_notmask, ...
	  pvp_num_BB_mask, ...
	  pvp_num_BB_notmask, ...
	  pvp_hit_list, ...
	  pvp_miss_list, ...
	  pvp_max_confidence] = ...
      pvp_numActiveInBoundingBox2(pvp_activity, ...
				  truth_CSV_struct, ...
				  other_CSV_struct, ...
				  DCR_CSV_struct);
  %%keyboard;
  global NFEATURES NCOLS NROWS N
  global pvp_patch_size
  global pvp_use_PANN_boundingBoxes
  global miss_list_flag;

  pvp_num_active_BB_mask = 0;
  pvp_num_active_BB_notmask = 0;
  pvp_num_BB_mask = 0;
  pvp_num_BB_notmask = 0;
  pvp_hit_list = [];
  pvp_miss_list = [];  
  pvp_max_confidence = -1;

  num_truth_BBs = length(truth_CSV_struct);
  num_other_BBs = length(other_CSV_struct);
  num_DCR_BBs = length(DCR_CSV_struct);
  if num_truth_BBs == 0 && num_DCR_BBs == 0 && num_other_BBs == 0
    return;
  endif
  pvp_hit_list = cell(num_truth_BBs,1);
  pvp_miss_list = cell(1,1);
  pvp_BB_count = zeros(num_truth_BBs,1);
  BB_mask = zeros(NROWS, NCOLS);
  pvp_activity3D = reshape(full(pvp_activity), [NFEATURES NCOLS NROWS]);
  pvp_activity2D = squeeze(sum(pvp_activity3D, 1))';
  pvp_max_confidence = 0;
  for i_BB = 1 : num_truth_BBs
    if isempty(truth_CSV_struct)
      break;
    endif
    if isempty(truth_CSV_struct{i_BB})
      continue;
    endif
    x_BB_min = min([truth_CSV_struct{i_BB}.BoundingBox_X1; ...
		   truth_CSV_struct{i_BB}.BoundingBox_X2; ...
		   truth_CSV_struct{i_BB}.BoundingBox_X3; ...
		   truth_CSV_struct{i_BB}.BoundingBox_X4]);
    if x_BB_min <= 0
      x_BB_min = 1;
    endif
    x_BB_max = max([truth_CSV_struct{i_BB}.BoundingBox_X1; ...
		   truth_CSV_struct{i_BB}.BoundingBox_X2; ...
		   truth_CSV_struct{i_BB}.BoundingBox_X3; ...
		   truth_CSV_struct{i_BB}.BoundingBox_X4]);
    if x_BB_max > NCOLS
      x_BB_max = NCOLS;
    endif
    y_BB_min = min([truth_CSV_struct{i_BB}.BoundingBox_Y1; ...
		   truth_CSV_struct{i_BB}.BoundingBox_Y2; ...
		   truth_CSV_struct{i_BB}.BoundingBox_Y3; ...
		   truth_CSV_struct{i_BB}.BoundingBox_Y4]);
    if y_BB_min <= 0
      y_BB_min = 1;
    endif
    y_BB_max = max([truth_CSV_struct{i_BB}.BoundingBox_Y1; ...
		   truth_CSV_struct{i_BB}.BoundingBox_Y2; ...
		   truth_CSV_struct{i_BB}.BoundingBox_Y3; ...
		   truth_CSV_struct{i_BB}.BoundingBox_Y4]);
    if y_BB_max > NROWS
      y_BB_max = NROWS;
    endif

    if pvp_use_PANN_boundingBoxes
      x_BB_center = mean(x_BB_min, x_BB_max);
      y_BB_center = mean(y_BB_min, y_BB_max);
      x_BB_width = pvp_patch_size(2); %% x_BB_max - x_BB_min + 1;
      y_BB_width = pvp_patch_size(1); %% y_BB_max - y_BB_min + 1;
      
      BB_scale = 1.0/2.0;
      x_BB_min2 = floor(max(1, x_BB_center - BB_scale * x_BB_width));
      x_BB_max2 = ceil(min(NCOLS, x_BB_center + BB_scale * x_BB_width));
      y_BB_min2 = floor(max(1, y_BB_center - BB_scale * y_BB_width));
      y_BB_max2 = ceil(min(NROWS, y_BB_center + BB_scale * y_BB_width));
    else
      x_BB_min2 = x_BB_min;
      x_BB_max2 = x_BB_max;
      y_BB_min2 = y_BB_min;
      y_BB_max2 = y_BB_max;
    endif      

    
    BB_mask2 = zeros(NROWS, NCOLS);   
    BB_mask2(y_BB_min2:y_BB_max2, x_BB_min2:x_BB_max2) = 1;
    BB_mask_area = nnz(BB_mask2);

    pvp_BB_count(i_BB) = nnz(pvp_activity2D .* (BB_mask2==1));
    pvp_hit_list{i_BB}.Confidence = pvp_BB_count(i_BB) / BB_mask_area;
    pvp_max_confidence = max(pvp_max_confidence, pvp_hit_list{i_BB}.Confidence);
    pvp_hit_list{i_BB}.BoundingBox_X1 = x_BB_min2;
    pvp_hit_list{i_BB}.BoundingBox_Y1 = y_BB_min2;
    pvp_hit_list{i_BB}.BoundingBox_X2 = x_BB_max2;
    pvp_hit_list{i_BB}.BoundingBox_Y2 = y_BB_min2;
    pvp_hit_list{i_BB}.BoundingBox_X3 = x_BB_max2;
    pvp_hit_list{i_BB}.BoundingBox_Y3 = y_BB_max2;
    pvp_hit_list{i_BB}.BoundingBox_X4 = x_BB_min2;
    pvp_hit_list{i_BB}.BoundingBox_Y4 = y_BB_max2;
    pvp_hit_list{i_BB}.hit_density = pvp_hit_list{i_BB}.Confidence;


    BB_mask(y_BB_min2:y_BB_max2, x_BB_min2:x_BB_max2) = 1;

  endfor %% i_BB
  for i_BB = 1 : num_truth_BBs
    if isempty(pvp_hit_list{i_BB})
      continue;
    endif
    pvp_hit_list{i_BB}.Confidence = ...
	pvp_hit_list{i_BB}.Confidence / (pvp_max_confidence +(pvp_max_confidence==0));
  endfor
  pvp_num_active_BB_mask = nnz(pvp_activity2D .* (BB_mask==1));
  pvp_num_BB_mask = nnz(BB_mask);

  for i_BB = 1 : num_DCR_BBs
    if isempty(DCR_CSV_struct{i_BB})
      continue;
    endif
    x_BB_min = min([DCR_CSV_struct{i_BB}.BoundingBox_X1; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_X2; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_X3; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_X4]);
    if x_BB_min <= 0
      x_BB_min = 1;
    endif
    x_BB_max = max([DCR_CSV_struct{i_BB}.BoundingBox_X1; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_X2; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_X3; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_X4]);
    if x_BB_max > NCOLS
      x_BB_max = NCOLS;
    endif
    y_BB_min = min([DCR_CSV_struct{i_BB}.BoundingBox_Y1; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_Y2; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_Y3; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_Y4]);
    if y_BB_min <= 0
      y_BB_min = 1;
    endif
    y_BB_max = max([DCR_CSV_struct{i_BB}.BoundingBox_Y1; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_Y2; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_Y3; ...
		   DCR_CSV_struct{i_BB}.BoundingBox_Y4]);
    if y_BB_max > NROWS
      y_BB_max = NROWS;
    endif

    BB_mask(y_BB_min:y_BB_max, x_BB_min:x_BB_max) = 1;

  endfor %% i_BB

  if miss_list_flag
  BB_mask3D = repmat(BB_mask, [1, 1, NFEATURES]);
  BB_mask3D = permute(BB_mask3D, [3, 2, 1]);
  false_pos_activity3D = pvp_activity3D .* (BB_mask3D==0);
  false_pos_activity = sparse(false_pos_activity3D(:));
  [pvp_miss_list] = ...
      pvp_dbscan(false_pos_activity);
  pvp_num_active_BB_notmask = 0;
  pvp_num_BB_notmask = 0;
  if isempty(pvp_miss_list)
    return;
  endif
  BB_mask = zeros(NROWS, NCOLS);
  pvp_num_miss = length(pvp_miss_list);
  disp(["num_miss = ", num2str(pvp_num_miss)]);
  for i_BB = 1 : pvp_num_miss
    x_BB_min2 = pvp_miss_list{i_BB}.patch_X1;
    x_BB_max2 = pvp_miss_list{i_BB}.patch_X2;
    y_BB_min2 = pvp_miss_list{i_BB}.patch_Y1;
    y_BB_max2 = pvp_miss_list{i_BB}.patch_Y3;
    BB_mask(y_BB_min2:y_BB_max2, x_BB_min2:x_BB_max2) = 1;    
  endfor %% i_BB
  pvp_num_active_BB_notmask = nnz(pvp_activity2D .* (BB_mask==1));
  pvp_num_BB_notmask = nnz(BB_mask);
  endif

endfunction %% pvp_numActiveInBoundingBox2
