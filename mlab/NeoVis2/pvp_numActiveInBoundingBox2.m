function [pvp_num_active_BB_mask, ...
	  pvp_num_active_BB_notmask, ...
	  pvp_num_BB_mask, ...
	  pvp_num_BB_notmask, ...
	  pvp_hit_list, ...
	  pvp_miss_list] = ...
      pvp_numActiveInBoundingBox2(pvp_activity, ...
				 true_CSV_struct);
  %%keyboard;
  global NFEATURES NCOLS NROWS N
  global pvp_patch_size

  num_BBs = length(true_CSV_struct);
  pvp_hit_list = cell(num_BBs,1);
  pvp_miss_list = cell(num_BBs,1);
  pvp_BB_count = zeros(num_BBs,1);
  BB_mask = zeros(NROWS, NCOLS);
  pvp_activity3D = reshape(full(pvp_activity), [NFEATURES NCOLS NROWS]);
  pvp_activity3D = permute(pvp_activity3D, [3,2,1]);
  pvp_activity2D = squeeze(sum(pvp_activity3D, 3));
  max_confidence = 0;
  for i_BB = 1 : num_BBs
    x_BB_min = min([true_CSV_struct{i_BB}.BoundingBox_X1; ...
		   true_CSV_struct{i_BB}.BoundingBox_X2; ...
		   true_CSV_struct{i_BB}.BoundingBox_X3; ...
		   true_CSV_struct{i_BB}.BoundingBox_X4]);
    if x_BB_min <= 0
      x_BB_min = 1;
    endif
    x_BB_max = max([true_CSV_struct{i_BB}.BoundingBox_X1; ...
		   true_CSV_struct{i_BB}.BoundingBox_X2; ...
		   true_CSV_struct{i_BB}.BoundingBox_X3; ...
		   true_CSV_struct{i_BB}.BoundingBox_X4]);
    if x_BB_max > NCOLS
      x_BB_max = NCOLS;
    endif
    y_BB_min = min([true_CSV_struct{i_BB}.BoundingBox_Y1; ...
		   true_CSV_struct{i_BB}.BoundingBox_Y2; ...
		   true_CSV_struct{i_BB}.BoundingBox_Y3; ...
		   true_CSV_struct{i_BB}.BoundingBox_Y4]);
    if y_BB_min <= 0
      y_BB_min = 1;
    endif
    y_BB_max = max([true_CSV_struct{i_BB}.BoundingBox_Y1; ...
		   true_CSV_struct{i_BB}.BoundingBox_Y2; ...
		   true_CSV_struct{i_BB}.BoundingBox_Y3; ...
		   true_CSV_struct{i_BB}.BoundingBox_Y4]);
    if y_BB_max > NROWS
      y_BB_max = NROWS;
    endif
    x_BB_center = mean(x_BB_min, x_BB_max);
    y_BB_center = mean(y_BB_min, y_BB_max);
    x_BB_width = pvp_patch_size(2); %% x_BB_max - x_BB_min + 1;
    y_BB_width = pvp_patch_size(1); %% y_BB_max - y_BB_min + 1;
    
    BB_scale = 1.0/2.0;
    x_BB_min2 = floor(max(1, x_BB_center - BB_scale * x_BB_width));
    x_BB_max2 = ceil(min(NCOLS, x_BB_center + BB_scale * x_BB_width));
    y_BB_min2 = floor(max(1, y_BB_center - BB_scale * y_BB_width));
    y_BB_max2 = ceil(min(NROWS, y_BB_center + BB_scale * y_BB_width));

    
    BB_mask2 = zeros(NROWS, NCOLS);   
    BB_mask2(y_BB_min2:y_BB_max2, x_BB_min2:x_BB_max2) = 1;
    BB_mask_area = nnz(BB_mask2);

    pvp_BB_count(i_BB) = nnz(pvp_activity2D .* (BB_mask2==1));
    pvp_hit_list{i_BB}.Confidence = pvp_BB_count(i_BB) / BB_mask_area;
    max_confidence = max(max_confidence, pvp_hit_list{i_BB}.Confidence);
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

  endfor
  for i_BB = 1 : num_BBs
    pvp_hit_list{i_BB}.Confidence = pvp_hit_list{i_BB}.Confidence / max_confidence;
  endfor
  pvp_num_active_BB_mask = nnz(pvp_activity2D .* (BB_mask==1));
  pvp_num_active_BB_notmask = nnz(pvp_activity2D .* (BB_mask==0));
  pvp_num_BB_mask = nnz(BB_mask);
  pvp_num_BB_notmask = nnz(~BB_mask);
  
endfunction %% pvp_numActiveInBoundingBoxy
