function ...
      [Sparse_time, ...
       Sparse_percent_active, ...
       Sparse_std, ...
       Sparse_hist_frame] = ...
      calcSparsePVPArray(Sparse_struct, ...
                         n_Sparse, ...
                         nf_Sparse)
  
  if isempty(Sparse_struct.values) 
    Sparse_time = nan;
    Sparse_percent_active = nan;
    Sparse_std = nan;
    Sparse_hist_frame = [];
    return;
  end%if

  size_values = size(Sparse_struct.values);
  if numel(size_values) > 2
    %% convert to sparse, values has dimensions nx,ny,nf
    nx = size_values(1);    
    ny = size_values(2);    
    nf = size_values(3);
    if (nx * ny * nf ~= n_Sparse) || (nf ~= nf_Sparse)
      error(["non-sparse array dimensions do not match input values"]);
      return;
    end%if
    [sparse_row_ndx, sparse_col_ndx, sparse_values] = find(Sparse_struct.values(:));
    Sparse_struct.values = zeros(length(sparse_values(:)),2);
    Sparse_struct.values(:,1) = sparse_row_ndx-1;
    Sparse_struct.values(:,2) = sparse_values;
  end%if

  Sparse_hist_centers = 1:nf_Sparse;
  Sparse_time = squeeze(Sparse_struct.time);
  Sparse_values_tmp = squeeze(Sparse_struct.values);
  Sparse_active_ndx = Sparse_values_tmp(:,1);
  if columns(Sparse_values_tmp) == 2
    Sparse_active_vals = Sparse_values_tmp(:,2);
  else
    Sparse_active_vals = ones(size(Sparse_active_ndx));
  end%if
  if max(Sparse_active_ndx(:) > n_Sparse)
    keyboard;
  end%if
  Sparse_current = sparse(Sparse_active_ndx+1,1,Sparse_active_vals,n_Sparse,1,n_Sparse);
  Sparse_current_active = nnz(Sparse_current(:));
  Sparse_percent_active = Sparse_current_active / n_Sparse;
  Sparse_std = sqrt(mean(Sparse_current.^2));
  Sparse_active_kf = mod(Sparse_active_ndx, nf_Sparse) + 1;
  if Sparse_current_active > 0
    Sparse_hist_frame = hist(Sparse_active_kf, Sparse_hist_centers)';
  else
    Sparse_hist_frame = zeros(1,nf_Sparse);
  end%if
  
  %%keyboard;

endfunction
