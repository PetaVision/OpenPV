  ErrorVsSparse_list = [nonSparse_list; ReconError_list];
  num_ErrorVsSparse_list = size(ErrorVsSparse_list,1);
  Sparse_axis_index = ones(num_ErrorVsSparse_list,1);
  if strcmp(run_type, "Heli_C1") 
    Sparse_axis_index(2) = 2;
    Sparse_axis_index(3) = 2;
    Sparse_axis_index(4) = 1;
    Sparse_axis_index(5) = 2;
    Sparse_axis_index(6) = 2;
    Sparse_axis_index(7) = 3;
  elseif strcmp(run_type, "Heli_DPTX3")
    Sparse_axis_index = zeros(num_ErrorVsSparse_list,3);
    Sparse_axis_index(1,1) = 1;
    Sparse_axis_index(2,1) = 2;
    Sparse_axis_index(3,1) = 2;
    Sparse_axis_index(4,1) = 3;
    Sparse_axis_index(5,1) = 3;
    Sparse_axis_index(6,1) = 1;
    Sparse_axis_index(7,1) = 2;
    Sparse_axis_index(8,1) = 3;
    Sparse_axis_index(:,2) = 2;
    Sparse_axis_index(:,3) = 3;
  elseif strcmp(run_type, "Heli_DTX3") || strcmp(run_type, "Heli_DX3")   
    Sparse_axis_index(2) = 2;
    Sparse_axis_index(3) = 3;
    Sparse_axis_index(4) = 1;
    Sparse_axis_index(5) = 2;
    Sparse_axis_index(6) = 3;
  elseif strcmp(run_type, "Heli_V1V2V4") 
    Sparse_axis_index(2) = 2;
    Sparse_axis_index(3) = 3;
    Sparse_axis_index(4) = 1;
    Sparse_axis_index(5) = 2;
    Sparse_axis_index(6) = 3;
  elseif strcmp(run_type, "CIFAR_C1")
    Sparse_axis_index(2) = 2;
    Sparse_axis_index(3) = 2;
    Sparse_axis_index(4) = 1;
    Sparse_axis_index(5) = 2;
  endif



  ErrorVsSparse_dir = [output_dir, filesep, "ErrorVsSparse"]
  [status, msg, msgid] = mkdir(ErrorVsSparse_dir);
  if status ~= 1
    warning(["mkdir(", ErrorVsSparse_dir, ")", " msg = ", msg]);
  endif 
  for i_ErrorVsSparse = 1 : num_ErrorVsSparse_list
    nonSparse_times = nonSparse_times_array{i_ErrorVsSparse};
    nonSparse_RMS = nonSparse_RMS_array{i_ErrorVsSparse};
    nonSparse_norm_RMS = nonSparse_norm_RMS_array{i_ErrorVsSparse};
    num_nonSparse_frames = length(nonSparse_times);
    if num_nonSparse_frames < 2
      continue;
    endif



    %% get percent active bins
    i_Sparse = Sparse_axis_index(i_ErrorVsSparse,1);
    Sparse_times = Sparse_times_array{i_Sparse};
    Sparse_percent_active = Sparse_percent_active_array{i_Sparse};
    num_Sparse_neurons = ...
	Sparse_hdr{i_Sparse}.nxGlobal * Sparse_hdr{i_Sparse}.nyGlobal * Sparse_hdr{i_Sparse}.nf;
    Sparse_sum_norm = 1;
    Sparse_sum_offset = 0;
    for i_Sparse_sum = 2 : size(Sparse_axis_index,2)
      i_Sparse = Sparse_axis_index(i_ErrorVsSparse,i_Sparse_sum);
      if i_Sparse <= 0
	continue;
      endif
      Sparse_times_sum = Sparse_times_array{i_Sparse};
      while(Sparse_times_sum(end-Sparse_sum_offset) > Sparse_times(end))
	Sparse_sum_offset = Sparse_sum_offset + 1;
      endwhile
      Sparse_times_sum = Sparse_times_sum(1:end-Sparse_sum_offset);
      %%Sparse_times_ndx = find(Sparse_times_sum ~= Sparse_times);
      %%Sparse_times_sum =Sparse_times_sum(Sparse_times_ndx);
      %%if any(Sparse_times_sum ~= Sparse_times)
      %%	keyboard;
      %%endif
      num_Sparse_neurons_sum = ...
	  Sparse_hdr{i_Sparse}.nxGlobal * Sparse_hdr{i_Sparse}.nyGlobal * Sparse_hdr{i_Sparse}.nf;
      if length(Sparse_percent_active) > length(Sparse_percent_active_array{i_Sparse}(1:end-Sparse_sum_offset))
	Sparse_percent_active = Sparse_percent_active(1+Sparse_sum_offset:end) + (num_Sparse_neurons_sum / num_Sparse_neurons) * Sparse_percent_active_array{i_Sparse}(1:end-Sparse_sum_offset);
      else
	Sparse_percent_active = Sparse_percent_active + (num_Sparse_neurons_sum / num_Sparse_neurons) * Sparse_percent_active_array{i_Sparse}(1:end-Sparse_sum_offset);
      endif      
      Sparse_sum_norm = Sparse_sum_norm + (num_Sparse_neurons_sum / num_Sparse_neurons);
    endfor
    Sparse_percent_active = Sparse_percent_active / Sparse_sum_norm;
    first_nonSparse_time = nonSparse_times(2);
    second_nonSparse_time = nonSparse_times(3);
    last_nonSparse_time = nonSparse_times(end);    
    [first_Sparse_ndx1, ~, first_Sparse_diff1] = ...
	find((Sparse_times - first_nonSparse_time) >= 0, 1, "first");
    [first_Sparse_ndx2, ~, first_Sparse_diff2] = ...
	find((Sparse_times - first_nonSparse_time) <= 0, 1, "last");
    if abs(first_Sparse_diff1) < abs(first_Sparse_diff2)
      first_Sparse_ndx = first_Sparse_ndx1;
      first_Sparse_diff = first_Sparse_diff1;
    else
      first_Sparse_ndx = first_Sparse_ndx2;
      first_Sparse_diff = first_Sparse_diff2;
    endif      
    [second_Sparse_ndx1, ~, second_Sparse_diff1] = ...
	find(Sparse_times - second_nonSparse_time >= 0, 1, "first");
    [second_Sparse_ndx2, ~, second_Sparse_diff2] = ...
	find(Sparse_times - second_nonSparse_time <= 0, 1, "last");
    if abs(second_Sparse_diff1) < abs(second_Sparse_diff2)
      second_Sparse_ndx = second_Sparse_ndx1;
      second_Sparse_diff = second_Sparse_diff1;
    else
      second_Sparse_ndx = second_Sparse_ndx2;
      second_Sparse_diff = second_Sparse_diff2;
    endif      
    if max(Sparse_times(:)) >= last_nonSparse_time
      [last_Sparse_ndx, ~, last_Sparse_diff] = ...
	  find(Sparse_times - last_nonSparse_time < 0, 1, "last");
      %%last_nonSparse_ndx = num_nonSparse_frames;
    else
      %%[last_nonSparse_ndx, ~, last_nonSparse_diff] = find(nonSparse_times - Sparse_times(end) < 0, 1, "last");
      last_Sparse_ndx = length(Sparse_times);
    endif
    skip_Sparse_ndx = max(second_Sparse_ndx - first_Sparse_ndx, 1);
    Sparse_vals = 1-Sparse_percent_active(first_Sparse_ndx:skip_Sparse_ndx:last_Sparse_ndx);
    num_Sparse_vals = length(Sparse_vals);
    if num_Sparse_vals < 1
      continue;
    endif
    num_Sparse_bins = 5; %%10;
    min_Sparse_val = min(Sparse_vals(:));
    max_Sparse_val = max(Sparse_vals(:));
    skip_Sparse_val = (max_Sparse_val - min_Sparse_val) / num_Sparse_bins;
    if skip_Sparse_val == 0
      skip_Sparse_val = 1;
    endif
    Sparse_bins = min_Sparse_val : skip_Sparse_val : max_Sparse_val;
    Sparse_bins = Sparse_bins(1:end-1);
    Sparse_bin_ndx = ceil((Sparse_vals - min_Sparse_val) / skip_Sparse_val);
    Sparse_bin_ndx(Sparse_bin_ndx < 1) = 1;
    Sparse_bin_ndx(Sparse_bin_ndx > num_Sparse_bins) = num_Sparse_bins;


    mean_nonSparse_RMS = zeros(num_Sparse_bins, 1); 
    std_nonSparse_RMS = zeros(num_Sparse_bins, 1); 
    last_nonSparse_ndx = length(Sparse_vals);
    for i_Sparse_bin = 1 : num_Sparse_bins
      if ~isempty(nonSparse_norm_RMS(Sparse_bin_ndx == i_Sparse_bin))
	mean_nonSparse_RMS(i_Sparse_bin) = ...
	    mean(nonSparse_RMS(Sparse_bin_ndx == i_Sparse_bin) ./ ...
		 nonSparse_norm_RMS(Sparse_bin_ndx == i_Sparse_bin));
	std_nonSparse_RMS(i_Sparse_bin) = ...
	    std(nonSparse_RMS(Sparse_bin_ndx == i_Sparse_bin) ./ ...
		nonSparse_norm_RMS(Sparse_bin_ndx == i_Sparse_bin));
      endif
    endfor %% i_Sparse_bin
    last_nonSparse_ndx = length(Sparse_vals);
    ErrorVsSparse_name = ...
	["ErrorVsSparse_", ErrorVsSparse_list{i_ErrorVsSparse,1}, ErrorVsSparse_list{i_ErrorVsSparse,2}, "_", ...
	 num2str(nonSparse_times(num_nonSparse_frames), "%08d")];
    if plot_flag
      normalized_nonSparse_RMS = nonSparse_RMS(1:last_nonSparse_ndx) ./ ...
       	       (nonSparse_norm_RMS(1:last_nonSparse_ndx) + (nonSparse_norm_RMS(1:last_nonSparse_ndx) == 0));
      %%normalized_nonSparse_RMS = nonSparse_RMS(1:last_nonSparse_ndx);
      max_nonSparse_RMS = max(normalized_nonSparse_RMS(:));
      ErrorVsSparse_fig = figure;
      ErrorVsSparse_hndl = ...
	  plot(Sparse_vals, ...
	       normalized_nonSparse_RMS, ...
	       "."); 
      if max_nonSparse_RMS <= 1.0
	axis([min(min_Sparse_val,0.90) 1.0 0 1.0]);
      else
	axis([min(min_Sparse_val,0.90) 1.0 0 max_nonSparse_RMS]);
      endif
      hold on
      eh = errorbar(Sparse_bins+skip_Sparse_val/2, mean_nonSparse_RMS, std_nonSparse_RMS);
      set(eh, "color", [0 0 0]);
      set(eh, "linewidth", 1.5);
      set(ErrorVsSparse_fig, "name", ...
	  ErrorVsSparse_name);
      saveas(ErrorVsSparse_fig, ...
	     [ErrorVsSparse_dir, filesep, ...
	      ErrorVsSparse_name, ".", "png"], "png");
    endif %% plot_flag
    save("-mat", ...
	 [ErrorVsSparse_dir, filesep, ErrorVsSparse_name, ".mat"], ...
	 "nonSparse_times", "Sparse_vals", "nonSparse_RMS", "nonSparse_norm_RMS", ...
	 "Sparse_bins", "mean_nonSparse_RMS", "std_nonSparse_RMS");	 
    %keyboard;
  endfor  %% i_ErrorVsSparse
  drawnow;
