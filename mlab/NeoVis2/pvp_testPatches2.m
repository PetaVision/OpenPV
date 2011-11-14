function [hit_list, miss_list] = pvp_testPatches2(pvp_activity)

  global NFEATURES NCOLS NROWS N
  global pvp_patch_size
  global pvp_density_thresh 
  global pvp_nnz_dist_min pvp_ave_dist_min pvp_std_dist_min
  global pvp_nnz_dist_max pvp_ave_dist_max pvp_std_dist_max

  %%keyboard;
  hist_list = cell(1);
  miss_list = [];
  nnz_activity = nnz(pvp_activity);
  if nnz_activity == 0
    return;
  endif
  
  num_hits = 0;

  if pvp_nnz_dist_min < 0
    nnz_dist_min = nnz_activity * prod(patch_size) * NFEATURES / N;
  else
    nnz_dist_min = pvp_nnz_dist_min;
  endif
  if pvp_ave_dist_min < 0
    ave_dist_min = mean(patch_size) / 2;
  else
    ave_dist_min = pvp_ave_dist_min;
  endif
  if pvp_std_dist_min < 0
    std_dist_min = mean(patch_size) / 2;
  else
    std_dist_min = pvp_std_dist_min;
  endif

  [active_feature_ndx, active_col_ndx, active_row_ndx] = ...
      ind2sub([NFEATURES NCOLS NROWS], pvp_activity);  %% returns row vectors
  col_toeplitz = toeplitz(active_col_ndx');
  col_repmat = repmat(active_col_ndx, [nnz_active, 1]);
  col_dist = col_toeplitz - col_repmat;
  col_dist(abs(col_dist > patch_size(2))) = 0;
  row_toeplitz = toeplitz(active_row_ndx');
  row_repmat = repmat(active_row_ndx, [nnz_active, 1]);
  row_dist = row_toeplitz - row_repmat;
  row_dist(abs(row_dist > patch_size(1))) = 0;
  euclid_dist = sqrt(col_dist.^2 + row_dist.^2);
  nnz_dist = sum(euclid_dist > 0);
  ave_dist = sum(euclid_dist) ./ nnz_dist;
  std_dist = sqrt((sum(euclid_dist.^2)) ./ nnz_dist - ave_dist.^2);
  hit_ndx = ...
      find((nnz_dist > nnz_dist_min) && ...
	   (ave_dist < ave_dist_min) && ...
	   (std_dist < std_dist_min));
  num_hits = nnz(hit_ndx);
  num_miss = nnz_activity - num_hits;
  %% could find largest eigenvectors of num_hits x num_hits distance matrix
  

  

 
  delta_patch_row = (NROWS - pvp_patch_size(1)) / (num_patch_rows - 1);
  delta_patch_col = (NCOLS - pvp_patch_size(2)) / (num_patch_cols - 1);
  for i_patch_row = 1 : num_patch_rows
    patch_row_min = ceil(1 + delta_patch_row * (i_patch_row - 1));
    patch_row_max = floor(pvp_patch_size(1) + delta_patch_row * (i_patch_row - 1));
    for i_patch_col = 1 : num_patch_cols
      patch_col_min = ceil(1 + delta_patch_col * (i_patch_col - 1));
      patch_col_max = floor(pvp_patch_size(2) + delta_patch_col * (i_patch_col - 1));
      [feature_mesh, col_mesh, row_mesh] = ...
	  meshgrid((1 : NFEATURES), (patch_col_min : patch_col_max), (patch_row_min : patch_row_max));
      patch_ndx = ...
	  sub2ind([NFEATURES NCOLS NROWS], feature_mesh(:), col_mesh(:), row_mesh(:));
      patch_num_active = nnz(pvp_activity(patch_ndx));
      if patch_num_active == 0
	continue;
      endif
      patch_area = length(patch_ndx);
      patch_density = patch_num_active / patch_area;
      patch_active_ndx = find(pvp_activity(patch_ndx));
      [patch_active_feature, patch_active_col, patch_active_row] = ...
	  ind2sub([NFEATURES NCOLS NROWS], patch_ndx(patch_active_ndx));
      patch_row_ave = mean(patch_active_row);
      patch_col_ave = mean(patch_active_col);
      patch_row_std = std(patch_active_row);
      patch_col_std = std(patch_active_col);
      if patch_density >= pvp_density_thresh
	num_hits = num_hits + 1; 
	hit_list{num_hits} = struct;
	hit_list{num_hits}.hit_density = patch_density;
	hit_list{num_hits}.patch_X1 = patch_col_min;
	hit_list{num_hits}.patch_Y1 = patch_row_min;
	hit_list{num_hits}.patch_X2 = patch_col_max;
	hit_list{num_hits}.patch_Y2 = patch_row_min;
	hit_list{num_hits}.patch_X3 = patch_col_max;
	hit_list{num_hits}.patch_Y3 = patch_row_max;
	hit_list{num_hits}.patch_X4 = patch_col_min;
	hit_list{num_hits}.patch_Y4 = patch_row_max;
	hit_list{num_hits}.Confidence = 1.0;
	hit_list{num_hits}.BoundingBox_X1 = max(ceil(patch_col_ave - patch_col_std), patch_col_min);
	hit_list{num_hits}.BoundingBox_Y1 = max(ceil(patch_row_ave - patch_row_std), patch_row_min);
	hit_list{num_hits}.BoundingBox_X2 = min(floor(patch_col_ave + patch_col_std), patch_col_max);
	hit_list{num_hits}.BoundingBox_Y2 = max(ceil(patch_row_ave - patch_row_std), patch_row_min);
	hit_list{num_hits}.BoundingBox_X3 = min(floor(patch_col_ave + patch_col_std), patch_col_max);
	hit_list{num_hits}.BoundingBox_Y3 = min(floor(patch_row_ave + patch_row_std), patch_row_max);
	hit_list{num_hits}.BoundingBox_X4 = max(ceil(patch_col_ave - patch_col_std), patch_col_min);
	hit_list{num_hits}.BoundingBox_Y4 = min(floor(patch_row_ave + patch_row_std), patch_row_max);
      else
	miss_list(i_patch_row, i_patch_col) = patch_density;
      endif
    endfor %% i_patch_col
  endfor %% i_patch_row


endfunction %% pvp_testPatches