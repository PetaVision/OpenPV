function ...
  [Sparse_time, hist_pool] = ...
  calcSparseHistPVPArray(Sparse_struct, ...
			 nx_full, ny_full, nf_full, ...
			 nx_GT, ny_GT, ...
			 min_val, max_val, mean_val, std_val, median_val, num_bins)
  
  if isempty(Sparse_struct.values) 
    hist_pool = nan;
    return;
  endif
  
  Sparse_time = squeeze(Sparse_struct.time);
  size_values = size(Sparse_struct.values);
  n_sparse = nx_full * ny_full * nf_full;
  full_vals = squeeze(Sparse_struct.values);
  if numel(size_values) <= 2  %% convert to full
    Sparse_active_ndx = full_vals(:,1);
    num_active = numel(Sparse_active_ndx);
    if columns(full_vals) == 2
      Sparse_active_vals = full_vals(:,2);
    else
      Sparse_active_vals = ones(size(Sparse_active_ndx),1);
    endif
    f_index_active = mod(Sparse_active_ndx, nf_full) + 1;
    x_index_active = mod(floor(Sparse_active_ndx / nf_full), nx_full) + 1;
    y_index_active = mod(floor(Sparse_active_ndx / (nf_full*nx_full)), ny_full) + 1;
    full_vals = zeros(ny_full, nx_full, nf_full);
    linear_active = sub2ind([ny_full, nx_full, nf_full], y_index_active(:), x_index_active(:), f_index_active(:));
    full_vals(linear_active(:)) = Sparse_active_vals(:);
  else
    full_vals = permute(full_vals, [2,1,3]);
  endif

  halfstep_ratio = (max_val/median_val)^(1/(num_bins-3));
  hist_centers = [0, min_val, median_val * halfstep_ratio.^([0:num_bins-3])];
  
  %% need to map activity so that each column contains all the activations for a given feature within one GT output tile
  %%keyboard;
  hist_pool = zeros(num_bins, nf_full, ny_GT, nx_GT);
  x_GT_size = floor(nx_full / nx_GT);
  y_GT_size = floor(ny_full / ny_GT);
  for j_yGT = 1 : ny_GT
    for i_xGT = 1 : nx_GT
      GT_vals3D = full_vals((j_yGT-1)*y_GT_size+1:j_yGT*y_GT_size, (i_xGT-1)*x_GT_size+1:i_xGT*x_GT_size, :);
      GT_vals2D = reshape(GT_vals3D, [y_GT_size*x_GT_size, nf_full]);
      [i_row_nnz, j_col_nnz] = find(GT_vals2D);
      [GT_hist, GT_bins] = hist(GT_vals2D, hist_centers);
      hist_pool(:, :, j_yGT, i_xGT) = GT_hist;
    endfor
  endfor
endfunction
