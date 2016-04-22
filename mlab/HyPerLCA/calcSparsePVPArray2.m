function ...
      [Sparse_time, ...
       Sparse_percent_active, ...
       Sparse_std, ...
       Sparse_hist_frame, ...
       Sparse_percent_change, ...
       Sparse_max_val, ...
       Sparse_min_val, ...
       Sparse_mean_val, ...
       Sparse_std_val, ...
       Sparse_median_val] = ...
      calcSparsePVPArray2(Sparse_struct, ...
			 Sparse_previous_struct, ...
			 n_Sparse, ...
			 nf_Sparse)
  
  if isempty(Sparse_struct.values) 
    Sparse_time = nan;
    Sparse_percent_active = nan;
    Sparse_std = nan;
    Sparse_hist_frame = [];
    Sparse_percent_change = nan;
    return;
  endif

  size_values = size(Sparse_struct.values);
  if numel(size_values) > 2
    %% convert to sparse, values has dimensions nx,ny,nf
    nx = size_values(1);    
    ny = size_values(2);    
    nf = size_values(3);
    if (nx * ny * nf ~= n_Sparse) || (nf ~= nf_Sparse)
      error(["non-sparse array dimensions do not match input values"]);
      return;
    endif
    [sparse_row_ndx, sparse_col_ndx, sparse_values] = find(Sparse_struct.values(:));
    %%keyboard;
    Sparse_struct.values = zeros(length(sparse_values(:)),2);
    Sparse_struct.values(:,1) = sparse_row_ndx-1;
    Sparse_struct.values(:,2) = sparse_values;
  endif

  Sparse_hist_edges = [0:1:nf_Sparse]+0.5;
  Sparse_time = squeeze(Sparse_struct.time);
  Sparse_values_tmp = squeeze(Sparse_struct.values);
  Sparse_active_ndx = Sparse_values_tmp(:,1);
  if columns(Sparse_values_tmp) == 2
    Sparse_active_vals = Sparse_values_tmp(:,2);
  else
    Sparse_active_vals = ones(size(Sparse_active_ndx));
  endif
  %%Sparse_current = full(sparse(Sparse_active_ndx+1,1,Sparse_active_vals,n_Sparse,1,n_Sparse));
  Sparse_current = sparse(Sparse_active_ndx+1,1,Sparse_active_vals,n_Sparse,1,n_Sparse);
  Sparse_current_active = nnz(Sparse_current(:));
  Sparse_percent_active = Sparse_current_active / n_Sparse;
  Sparse_std = sqrt(mean(Sparse_current(:).^2));
  Sparse_max_val = max(Sparse_active_vals(:));
  Sparse_min_val = min(Sparse_active_vals(:));
  Sparse_mean_val = mean(Sparse_active_vals(:));
  Sparse_std_val = std(Sparse_active_vals(:));
  Sparse_median_val = median(Sparse_active_vals(:));
  Sparse_active_kf = mod(Sparse_active_ndx, nf_Sparse) + 1;
  if Sparse_current_active > 0
    Sparse_hist_frame = histc(Sparse_active_kf, Sparse_hist_edges)';
  else
    Sparse_hist_frame = zeros(1,nf_Sparse+1);
  endif
  
  if isempty(Sparse_previous_struct.values) 
    Sparse_percent_change = nan;
    return;
  endif

  Sparse_previous_values_tmp = squeeze(Sparse_previous_struct.values);
  Sparse_previous_active_ndx = Sparse_previous_values_tmp(:,1);
  if columns(Sparse_previous_values_tmp) == 2
    Sparse_previous_active_vals = Sparse_previous_values_tmp(:,2);
  else
    Sparse_previous_active_vals = ones(size(Sparse_previous_active_ndx));
  endif
  %%Sparse_current = full(sparse(Sparse_active_ndx+1,1,Sparse_active_vals,n_Sparse,1,n_Sparse));
  Sparse_previous = sparse(Sparse_previous_active_ndx+1,1,Sparse_previous_active_vals,n_Sparse,1,n_Sparse);
  Sparse_previous_active = nnz(Sparse_previous(:));

  Sparse_abs_change = sum((Sparse_current(:)~=0) ~= (Sparse_previous(:)~=0));
  Sparse_OR_active = sum((Sparse_current(:)~=0) | (Sparse_previous(:)~=0));
  Sparse_percent_change = ...
      Sparse_abs_change / (Sparse_OR_active + (Sparse_OR_active==0));
  %%keyboard;

endfunction
