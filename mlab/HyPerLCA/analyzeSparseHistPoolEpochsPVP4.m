function [Sparse_hdr, ...
	  Sparse_hist_pool_array, ...
	  Sparse_times_array, ...
	  Sparse_max_pool_array, ...
	  Sparse_mean_pool_array, ...
	  Sparse_max2X2_pool_array, ...
	  Sparse_mean2X2_pool_array, ...
	  Sparse_max4X4_pool_array, ...
	  Sparse_mean4X4_pool_array] = ...
      analyzeSparseHistPoolEpochsPVP4(Sparse_list, ...
				     output_dir, ...
				     Sparse_hist_rank_array, ...
				     load_Sparse_flag, ...
				     plot_Sparse_flag, ...
				     fraction_Sparse_frames_read, ...
				     min_Sparse_skip, ...
				     fraction_Sparse_progress, ...
				     Sparse_min_val_array, ...
				     Sparse_max_val_array, ...
				     Sparse_mean_val_array, ...
				     Sparse_std_val_array, ...
				     Sparse_median_val_array, ...
				     nx_GT, ny_GT, ...
				     num_Sparse_hist_pool_bins, ...
				     save_Sparse_hist_pool_flag, ...
				     hist_pool_flag, ...
				     max_pool_flag, ...
				     mean_pool_flag, ...
				     max2X2_pool_flag, ...
				     mean2X2_pool_flag, ...
				     max4X4_pool_flag, ...
				     mean4X4_pool_flag, ...
				     num_procs, ...
				     num_epochs)
  %% analyze sparse activity pvp file.  
  %% Sparse_hist_pool_array returns a histogram over activity (activity distribution) for each feature for an output grid of size nx_GT X ny_GT.  num_Sparse_hist_pool_bins gives the number of bins in the histogram, which range from Sparse_min_val to Sparse_max_val.  
  
  if ~exist("load_Sparse_flag") || isempty(load_Sparse_flag)
    load_Sparse_flag = false;  %% true -> load Sparse arrays from file
  endif
  if ~exist("plot_Sparse_flag") || isempty(plot_Sparse_flag)
    plot_Sparse_flag = false;
  endif
  if ~exist("min_Sparse_skip") || isempty(min_Sparse_skip)
    min_Sparse_skip = 0;  %% 0 -> skip no frames
  endif
  if ~exist("fraction_Sparse_frames_read") || isempty(fraction_Sparse_frames_read)
    fraction_Sparse_frames_read = 1;  %% 1 -> read all frames
  endif
  if ~exist("fraction_Sparse_progress") || isempty(fraction_Sparse_progress)
    fraction_Sparse_progress = 10;
  endif
  if ~exist("hist_pool_flag") || isempty(hist_pool_flag)
    hist_pool_flag = true;
  endif
  if ~exist("max_pool_flag") || isempty(max_pool_flag)
    max_pool_flag = true;
  endif
  if ~exist("mean_pool_flag") || isempty(mean_pool_flag)
    mean_pool_flag = true;
  endif
  if ~exist("num_procs") || isempty(num_procs)
    num_procs = 1;
  endif
  if ~exist("num_epochs") || isempty(num_epochs)
    num_epochs = 1;
  endif
  
  Sparse_hist_pool_array = [];
  Sparse_times_array = [];
  Sparse_max_pool_array = [];
  Sparse_mean_pool_array = [];
  Sparse_max2X2_pool_array = [];
  Sparse_mean2X2_pool_array = [];
  Sparse_max4X4_pool_array = [];
  Sparse_mean4X4_pool_array = [];
  
  num_Sparse_list = size(Sparse_list,1);
  if num_Sparse_list ==0
    warning(["analyzeSparseHistPoolEpochsPVP:num_Sparse_list == 0"]);
    return;
  endif
  Sparse_hdr = cell(num_Sparse_list,1);
  Sparse_hist_pool_array = cell(num_Sparse_list,1);
  Sparse_times_array = cell(num_Sparse_list,1);
  Sparse_max_pool_array = cell(num_Sparse_list,1);
  Sparse_mean_pool_array = cell(num_Sparse_list,1);
  Sparse_max2X2_pool_array = cell(num_Sparse_list,1);
  Sparse_mean2X2_pool_array = cell(num_Sparse_list,1);
  Sparse_max4X4_pool_array = cell(num_Sparse_list,1);
  Sparse_mean4X4_pool_array = cell(num_Sparse_list,1);
  Sparse_struct_array = cell(num_Sparse_list,1);
  Sparse_dir = [output_dir, filesep, "Sparse"]
  [status, msg, msgid] = mkdir(Sparse_dir);
  if status ~= 1
    warning(["mkdir(", Sparse_dir, ")", " msg = ", msg]);
  endif 
  for i_Sparse = 1 : num_Sparse_list
    Sparse_file = [output_dir, filesep, Sparse_list{i_Sparse,1}, Sparse_list{i_Sparse,2}, ".pvp"]
    if ~exist(Sparse_file, "file")
      warning(["file does not exist: ", Sparse_file]);
      continue
    endif
    Sparse_fid = fopen(Sparse_file);
    Sparse_hdr{i_Sparse} = readpvpheader(Sparse_fid);
    fclose(Sparse_fid);
    tot_Sparse_frames = Sparse_hdr{i_Sparse}.nbands;
    num_Sparse_skip = tot_Sparse_frames - fix(tot_Sparse_frames/fraction_Sparse_frames_read);  %% number of activity frames to analyze, counting backward from last frame, maximum is tot_Sparse_frames
    num_Sparse_skip = max(num_Sparse_skip, min_Sparse_skip);
    progress_step = ceil(tot_Sparse_frames / fraction_Sparse_progress);

    %% analyze sparse (or non-sparse) activity from pvp file
    if ~load_Sparse_flag
      nx_Sparse = Sparse_hdr{i_Sparse}.nx;
      ny_Sparse = Sparse_hdr{i_Sparse}.ny;
      nf_Sparse = Sparse_hdr{i_Sparse}.nf;
      n_Sparse = nx_Sparse * ny_Sparse * nf_Sparse;
      Sparse_max_val = max(Sparse_max_val_array{i_Sparse}(:));
      Sparse_min_val = min(Sparse_min_val_array{i_Sparse}(:));;
      Sparse_mean_val = mean(Sparse_mean_val_array{i_Sparse}(:));
      Sparse_std_val = mean(Sparse_std_val_array{i_Sparse}(:));
      Sparse_median_val = median(Sparse_median_val_array{i_Sparse}(:));
      
      nx_full_cell = cell(1);
      nx_full_cell{1} = nx_Sparse;
      ny_full_cell = cell(1);
      ny_full_cell{1} = ny_Sparse;
      nf_full_cell = cell(1);
      nf_full_cell{1} = nf_Sparse;
      nx_GT_cell = cell(1);
      nx_GT_cell{1} = nx_GT;
      ny_GT_cell = cell(1);
      ny_GT_cell{1} = ny_GT;
      Sparse_max_val_cell = cell(1);
      Sparse_max_val_cell{1} = Sparse_max_val;
      Sparse_min_val_cell = cell(1);
      Sparse_min_val_cell{1} = Sparse_min_val;
      Sparse_mean_val_cell = cell(1);
      Sparse_mean_val_cell{1} = Sparse_mean_val;
      Sparse_std_val_cell = cell(1);
      Sparse_std_val_cell{1} = Sparse_std_val;
      Sparse_median_val_cell = cell(1);
      Sparse_median_val_cell{1} = Sparse_median_val;
      num_Sparse_hist_pool_bins_cell = cell(1);
      num_Sparse_hist_pool_bins_cell{1} = num_Sparse_hist_pool_bins * hist_pool_flag;
      

      if num_epochs == 1

	[Sparse_struct, Sparse_hdr_tmp] = ...
	    readpvpfile(Sparse_file, progress_step, tot_Sparse_frames, num_Sparse_skip,1);
	num_Sparse_frames = size(Sparse_struct,1);
      
	if num_procs == 1
	  [Sparse_times_list, ...
	   Sparse_hist_pool, ...
	   Sparse_max_pool, ...
	   Sparse_mean_pool, ...
	   Sparse_max2X2_pool, ...
	   Sparse_mean2X2_pool, ...
	   Sparse_max4X4_pool, ...
	   Sparse_mean4X4_pool] = ...
	      cellfun(@calcSparseHistPVPArray4, ...
		      Sparse_struct, ...
			 nx_full_cell, ...
			 ny_full_cell, ...
			 nf_full_cell, ...
			 nx_GT_cell, ...
			 ny_GT_cell, ...
			 Sparse_min_val_cell, ...
			 Sparse_max_val_cell, ...
			 Sparse_mean_val_cell, ...
			 Sparse_std_val_cell, ...
			 Sparse_median_val_cell, ...
			 num_Sparse_hist_pool_bins_cell, ...
		      "UniformOutput", false);
	elseif num_procs > 1
	  [Sparse_times_list, ...
	   Sparse_hist_pool, ...
	   Sparse_max_pool, ...
	   Sparse_mean_pool, ...
	   Sparse_max2X2_pool, ...
	   Sparse_mean2X2_pool, ...
	   Sparse_max4X4_pool, ...
	   Sparse_mean4X4_pool] = ...
	      parcellfun(num_procs, ...
			 @calcSparseHistPVPArray4, ...
			 Sparse_struct, ...
			 nx_full_cell, ...
			 ny_full_cell, ...
			 nf_full_cell, ...
			 nx_GT_cell, ...
			 ny_GT_cell, ...
			 Sparse_min_val_cell, ...
			 Sparse_max_val_cell, ...
			 Sparse_mean_val_cell, ...
			 Sparse_std_val_cell, ...
			 Sparse_median_val_cell, ...
			 num_Sparse_hist_pool_bins_cell, ...
			 "UniformOutput", false);
	endif  %% num_procs

 	Sparse_times = cell2mat(Sparse_times_list);
	num_times = length(Sparse_times);

     else %% num_epochs

	num_frames_per_epoch = floor((tot_Sparse_frames-1)/num_epochs);
	for i_epoch = 1 : num_epochs

	  [Sparse_struct_epoch, Sparse_hdr_tmp] = ...
	      readpvpfile(Sparse_file, progress_step, i_epoch*num_frames_per_epoch, 1+(i_epoch-1)*num_frames_per_epoch,1);
	  num_Sparse_frames_epoch = size(Sparse_struct_epoch,1);

 	  if num_procs == 1
	    [Sparse_times_list_epoch, ...
	     Sparse_hist_pool_list_epoch, ...
	     Sparse_max_pool_list_epoch, ...
	     Sparse_mean_pool_list_epoch, ...
	     Sparse_max2X2_pool_list_epoch, ...
	     Sparse_mean2X2_pool_list_epoch, ...
	     Sparse_max4X4_pool_list_epoch, ...
	     Sparse_mean4X4_pool_list_epoch] = ...
		cellfun(@calcSparseHistPVPArray4, ...
			Sparse_struct_epoch, ...
			 nx_full_cell, ...
			 ny_full_cell, ...
			 nf_full_cell, ...
			 nx_GT_cell, ...
			 ny_GT_cell, ...
			 Sparse_min_val_cell, ...
			 Sparse_max_val_cell, ...
			 Sparse_mean_val_cell, ...
			 Sparse_std_val_cell, ...
			 Sparse_median_val_cell, ...
			 num_Sparse_hist_pool_bins_cell, ...
			"UniformOutput", false);
	  else
	    [Sparse_times_list_epoch, ...
	     Sparse_hist_pool_list_epoch, ...
	     Sparse_max_pool_list_epoch, ...
	     Sparse_mean_pool_list_epoch, ...
	     Sparse_max2X2_pool_list_epoch, ...
	     Sparse_mean2X2_pool_list_epoch, ...
	     Sparse_max4X4_pool_list_epoch, ...
	     Sparse_mean4X4_pool_list_epoch] = ...
		parcellfun(num_procs, ...
			   @calcSparsePVPArray4, ...
			   Sparse_struct_epoch, ...
			   nx_full_cell, ...
			   ny_full_cell, ...
			   nf_full_cell, ...
			   nx_GT_cell, ...
			   ny_GT_cell, ...
			   Sparse_min_val_cell, ...
			   Sparse_max_val_cell, ...
			   Sparse_mean_val_cell, ...
			   Sparse_std_val_cell, ...
			   Sparse_median_val_cell, ...
			   num_Sparse_hist_pool_bins_cell, ...
			   "UniformOutput", false);
	  endif %% num_procs	
	  if i_epoch == 1
	    Sparse_times = cell2mat(Sparse_times_list_epoch);
	    if ~isempty(Sparse_hist_pool_list_epoch)
	      Sparse_hist_pool = cell2mat(Sparse_hist_pool_list_epoch); 
	      Sparse_max_pool = cell2mat(Sparse_max_pool_list_epoch); 
	      Sparse_mean_pool = cell2mat(Sparse_mean_pool_list_epoch); 
	      Sparse_max2X2_pool = cell2mat(Sparse_max2X2_pool_list_epoch); 
	      Sparse_mean2X2_pool = cell2mat(Sparse_mean2X2_pool_list_epoch); 
	      Sparse_max4X4_pool = cell2mat(Sparse_max4X4_pool_list_epoch); 
	      Sparse_mean4X4_pool = cell2mat(Sparse_mean4X4_pool_list_epoch); 
	    else
	      Sparse_hist_pool = zeros(num_Sparse_hist_pool_bins, nf_Sparse, ny_GT, nx_GT, length(Sparse_times));
	    endif
	  else
	    Sparse_times = [Sparse_times; cell2mat(Sparse_times_list_epoch)];
 	    if ~isempty(Sparse_hist_pool_list_epoch)
	      Sparse_hist_pool = [Sparse_hist_pool; cell2mat(Sparse_hist_pool_list_epoch)]; 
	      Sparse_max_pool = [Sparse_max_pool; cell2mat(Sparse_max_pool_list_epoch)]; 
	      Sparse_mean_pool = [Sparse_mean_pool; cell2mat(Sparse_mean_pool_list_epoch)]; 
	      Sparse_max2X2_pool = [Sparse_max2X2_pool; cell2mat(Sparse_max2X2_pool_list_epoch)]; 
	      Sparse_mean2X2_pool = [Sparse_mean2X2_pool; cell2mat(Sparse_mean2X2_pool_list_epoch)]; 
	      Sparse_max4X4_pool = [Sparse_max4X4_pool; cell2mat(Sparse_max4X4_pool_list_epoch)]; 
	      Sparse_mean4X4_pool = [Sparse_mean4X4_pool; cell2mat(Sparse_mean4X4_pool_list_epoch)]; 
	    endif
	  endif %% i_epoch == 1
	endfor %% i_epoch
      endif %% num_epochs

      num_Sparse_frames = size(Sparse_times,1);      
      
      Sparse_filename_id = ["pool", "_", Sparse_list{i_Sparse,2}, "_",  ...
			    num2str(Sparse_times(num_Sparse_frames), "%08d")];


      if save_Sparse_hist_pool_flag && hist_pool_flag
	save("-mat", ...
	     [Sparse_dir, filesep, "hist_", Sparse_filename_id, "_bins", ".mat"], ...
	     "Sparse_hist_pool_bins");
	save("-mat", ...
	     [Sparse_dir, filesep, "hist_", Sparse_filename_id, ".mat"], ...
	     "Sparse_hist_pool");
      endif
      if max_pool_flag
	save("-mat", ...
	     [Sparse_dir, filesep, "max_", Sparse_filename_id, ".mat"], ...
	     "Sparse_max_pool");
      endif
      if mean_pool_flag
	save("-mat", ...
	     [Sparse_dir, filesep, "mean_", Sparse_filename_id, ".mat"], ...
	     "Sparse_mean_pool");
      endif
      if max2X2_pool_flag
	save("-mat", ...
	     [Sparse_dir, filesep, "max2X2_", Sparse_filename_id, ".mat"], ...
	     "Sparse_max2X2_pool");
      endif
      if mean2X2_pool_flag
	save("-mat", ...
	     [Sparse_dir, filesep, "mean2X2_", Sparse_filename_id, ".mat"], ...
	     "Sparse_mean2X2_pool");
      endif
      if max4X4_pool_flag
	save("-mat", ...
	     [Sparse_dir, filesep, "max4X4_", Sparse_filename_id, ".mat"], ...
	     "Sparse_max4X4_pool");
      endif
      if mean4X4_pool_flag
	save("-mat", ...
	     [Sparse_dir, filesep, "mean4X4_", Sparse_filename_id, ".mat"], ...
	     "Sparse_mean4X4_pool");
      endif

    else  %% load Sparse data structures from file


      Sparse_hist_pool_bins_str = ...
	  [Sparse_dir, filesep, "hist_pool_", Sparse_list{i_Sparse,2}, "*", "_bins", ".mat"];
      Sparse_hist_pool_bins_glob = glob(Sparse_hist_pool_bins_str);
      num_Sparse_hist_pool_bins_glob = length(Sparse_hist_pool_bins_glob);
      if num_Sparse_hist_pool_bins_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_hist_pool_bins_str]);
	break;
      endif
      load("-mat", Sparse_hist_pool_bins_glob{num_Sparse_hist_pool_bins_glob});
      
      Sparse_hist_pool_str = ...
	  [Sparse_dir, filesep, "hist_pool_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_hist_pool_glob = glob(Sparse_hist_pool_str);
      num_Sparse_hist_pool_glob = length(Sparse_hist_pool_glob);
      if num_Sparse_hist_pool_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_hist_pool_str]);
	break;
      endif
      load("-mat", Sparse_hist_pool_glob{num_Sparse_hist_pool_glob});
      
      Sparse_max_pool_str = ...
	  [Sparse_dir, filesep, "max_pool_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_max_pool_glob = glob(Sparse_max_pool_str);
      num_Sparse_max_pool_glob = length(Sparse_max_pool_glob);
      if num_Sparse_max_pool_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_max_pool_str]);
	break;
      endif
      load("-mat", Sparse_max_pool_glob{num_Sparse_max_pool_glob});

      Sparse_mean_pool_str = ...
      [Sparse_dir, filesep, "mean_pool_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_mean_pool_glob = glob(Sparse_mean_pool_str);
      num_Sparse_mean_pool_glob = length(Sparse_mean_pool_glob);
      if num_Sparse_mean_pool_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_mean_pool_str]);
	break;
      endif
      load("-mat", Sparse_mean_pool_glob{num_Sparse_mean_pool_glob});

      Sparse_max2X2_pool_str = ...
	  [Sparse_dir, filesep, "max2X2_pool_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_max2X2_pool_glob = glob(Sparse_max2X2_pool_str);
      num_Sparse_max2X2_pool_glob = length(Sparse_max2X2_pool_glob);
      if num_Sparse_max2X2_pool_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_max2X2_pool_str]);
	break;
      endif
      load("-mat", Sparse_max2X2_pool_glob{num_Sparse_max2X2_pool_glob});

      Sparse_mean2X2_pool_str = ...
      [Sparse_dir, filesep, "mean2X2_pool_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_mean2X2_pool_glob = glob(Sparse_mean2X2_pool_str);
      num_Sparse_mean2X2_pool_glob = length(Sparse_mean2X2_pool_glob);
      if num_Sparse_mean2X2_pool_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_mean2X2_pool_str]);
	break;
      endif
      load("-mat", Sparse_mean2X2_pool_glob{num_Sparse_mean2X2_pool_glob});

      Sparse_max4X4_pool_str = ...
	  [Sparse_dir, filesep, "max4X4_pool_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_max4X4_pool_glob = glob(Sparse_max4X4_pool_str);
      num_Sparse_max4X4_pool_glob = length(Sparse_max4X4_pool_glob);
      if num_Sparse_max4X4_pool_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_max4X4_pool_str]);
	break;
      endif
      load("-mat", Sparse_max4X4_pool_glob{num_Sparse_max4X4_pool_glob});

      Sparse_mean4X4_pool_str = ...
      [Sparse_dir, filesep, "mean4X4_pool_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_mean4X4_pool_glob = glob(Sparse_mean4X4_pool_str);
      num_Sparse_mean4X4_pool_glob = length(Sparse_mean4X4_pool_glob);
      if num_Sparse_mean4X4_pool_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_mean4X4_pool_str]);
	break;
      endif
      load("-mat", Sparse_mean4X4_pool_glob{num_Sparse_mean4X4_pool_glob});

    endif %% load_Sparse_flag

    num_Sparse_frames = length(Sparse_times);
    if plot_Sparse_flag 
      if ~exist("Sparse_hist_rank_array") || isempty("Sparse_hist_rank_array")
	Sparse_rank = 1:nf_Sparse;
      else
	Sparse_rank = Sparse_hist_rank_array{i_Sparse};
	if isempty(Sparse_rank)
	  Sparse_rank = 1:nf_Sparse;
	endif
      endif
      
      if hist_pool_flag
	Sparse_hist_pool_fig = figure;
	axis off; box off;
	k_subplot = 0;
	Sparse_hist_pool_hndl = zeros(ny_GT, nx_GT);
	Sparse_hist_pool_axis = zeros(ny_GT, nx_GT);
	for j_yGT = 1 : ny_GT
	  for i_xGT = 1 : nx_GT
	    k_subplot = k_subplot + 1;
	    Sparse_hist_pool_axis(j_yGT,i_xGT) = subplot(ny_GT, nx_GT, k_subplot);
	    hist_pool_tmp = squeeze(Sparse_hist_pool{num_Sparse_frames}(:,:,j_yGT,i_xGT));
	    ranked_hist_pool = hist_pool_tmp(:, Sparse_rank);
	    bar(ranked_hist_pool(2:num_Sparse_hist_pool_bins,2:ceil(nf_Sparse/48):nf_Sparse)', 'stacked');
	    axis tight;
	    axis off;
	    box off;
	    colormap(prism(num_Sparse_hist_pool_bins))
	  endfor
	endfor
	set(Sparse_hist_pool_fig, "name", ["Hist_", Sparse_filename_id]);
	saveas(Sparse_hist_pool_fig, ...
	       [Sparse_dir, filesep, "Hist_", Sparse_filename_id], "png");
      endif
      
      if max_pool_flag
	Sparse_max_pool_fig = figure;
	axis off; box off;
	k_subplot = 0;
	Sparse_max_pool_hndl = zeros(ny_GT, nx_GT);
	Sparse_max_pool_axis = zeros(ny_GT, nx_GT);
	for j_yGT = 1 : ny_GT
	  for i_xGT = 1 : nx_GT
	    k_subplot = k_subplot + 1;
	    Sparse_max_pool_axis(j_yGT,i_xGT) = subplot(ny_GT, nx_GT, k_subplot);
	    max_pool_tmp = squeeze(Sparse_max_pool{num_Sparse_frames}(:,j_yGT,i_xGT));
	    ranked_max_pool = max_pool_tmp(Sparse_rank);
	    bar(ranked_max_pool(1:ceil(nf_Sparse/48):nf_Sparse));
	    axis tight;
	    axis off;
	    box off;
	    colormap("default")
	  endfor
	endfor
	set(Sparse_max_pool_fig, "name", ["max_", Sparse_filename_id]);
	saveas(Sparse_max_pool_fig, ...
	       [Sparse_dir, filesep, "max_", Sparse_filename_id], "png");
      endif

      if mean_pool_flag
	Sparse_mean_pool_fig = figure;
	axis off; box off;
	k_subplot = 0;
	Sparse_mean_pool_hndl = zeros(ny_GT, nx_GT);
	Sparse_mean_pool_axis = zeros(ny_GT, nx_GT);
	for j_yGT = 1 : ny_GT
	  for i_xGT = 1 : nx_GT
	    k_subplot = k_subplot + 1;
	    Sparse_mean_pool_axis(j_yGT,i_xGT) = subplot(ny_GT, nx_GT, k_subplot);
	    mean_pool_tmp = squeeze(Sparse_mean_pool{num_Sparse_frames}(:,j_yGT,i_xGT));
	    ranked_mean_pool = mean_pool_tmp(Sparse_rank);
	    bar(ranked_mean_pool(1:ceil(nf_Sparse/48):nf_Sparse));
	    axis tight;
	    axis off;
	    box off;
	    colormap("default")
	  endfor
	endfor
	set(Sparse_mean_pool_fig, "name", ["mean_", Sparse_filename_id]);
	saveas(Sparse_mean_pool_fig, ...
	       [Sparse_dir, filesep, "mean_", Sparse_filename_id], "png");
      endif
	
      if max2X2_pool_flag
	Sparse_max2X2_pool_fig = figure;
	axis off; box off;
	k_subplot = 0;
	Sparse_max2X2_pool_hndl = zeros(ny_GT*2, nx_GT*2);
	Sparse_max2X2_pool_axis = zeros(ny_GT*2, nx_GT*2);
	for j_yGT = 1 : ny_GT*2
	  for i_xGT = 1 : nx_GT*2
	    k_subplot = k_subplot + 1;
	    Sparse_max2X2_pool_axis(j_yGT,i_xGT) = subplot(ny_GT*2, nx_GT*2, k_subplot);
	    max2X2_pool_tmp = squeeze(Sparse_max2X2_pool{num_Sparse_frames}(1+mod(j_yGT,2), 1+mod(i_xGT,2), :, ceil(j_yGT/2), ceil(i_xGT/2)));
	    ranked_max2X2_pool = max2X2_pool_tmp(Sparse_rank);
	    bar(ranked_max2X2_pool(1:ceil(nf_Sparse/48):nf_Sparse));
	    axis tight;
	    axis off;
	    box off;
	    colormap("default")
	  endfor
	endfor
	set(Sparse_max2X2_pool_fig, "name", ["max2X2_", Sparse_filename_id]);
	saveas(Sparse_max2X2_pool_fig, ...
	       [Sparse_dir, filesep, "max2X2_", Sparse_filename_id], "png");
      endif

      if mean2X2_pool_flag
	Sparse_mean2X2_pool_fig = figure;
	axis off; box off;
	k_subplot = 0;
	Sparse_mean2X2_pool_hndl = zeros(ny_GT*2, nx_GT*2);
	Sparse_mean2X2_pool_axis = zeros(ny_GT*2, nx_GT*2);
	for j_yGT = 1 : ny_GT*2
	  for i_xGT = 1 : nx_GT*2
	    k_subplot = k_subplot + 1;
	    Sparse_mean2X2_pool_axis(j_yGT,i_xGT) = subplot(ny_GT*2, nx_GT*2, k_subplot);
	    mean2X2_pool_tmp = squeeze(Sparse_mean2X2_pool{num_Sparse_frames}(1+mod(j_yGT,2), 1+mod(i_xGT,2), :, ceil(j_yGT/2), ceil(i_xGT/2)));
	    ranked_mean2X2_pool = mean2X2_pool_tmp(Sparse_rank);
	    bar(ranked_mean2X2_pool(1:ceil(nf_Sparse/48):nf_Sparse));
	    axis tight;
	    axis off;
	    box off;
	    colormap("default")
	  endfor
	endfor
	set(Sparse_mean2X2_pool_fig, "name", ["mean2X2_", Sparse_filename_id]);
	saveas(Sparse_mean2X2_pool_fig, ...
	       [Sparse_dir, filesep, "mean2X2_", Sparse_filename_id], "png");
      endif

      if max4X4_pool_flag
	Sparse_max4X4_pool_fig = figure;
	axis off; box off;
	k_subplot = 0;
	Sparse_max4X4_pool_hndl = zeros(ny_GT*4, nx_GT*4);
	Sparse_max4X4_pool_axis = zeros(ny_GT*4, nx_GT*4);
	for j_yGT = 1 : ny_GT*4
	  for i_xGT = 1 : nx_GT*4
	    k_subplot = k_subplot + 1;
	    Sparse_max4X4_pool_axis(j_yGT,i_xGT) = subplot(ny_GT*4, nx_GT*4, k_subplot);
	    max4X4_pool_tmp = squeeze(Sparse_max4X4_pool{num_Sparse_frames}(1+mod(j_yGT,4), 1+mod(i_xGT,4), :, ceil(j_yGT/4), ceil(i_xGT/4)));
	    ranked_max4X4_pool = max4X4_pool_tmp(Sparse_rank);
	    bar(ranked_max4X4_pool(1:ceil(nf_Sparse/48):nf_Sparse));
	    axis tight;
	    axis off;
	    box off;
	    colormap("default")
	  endfor
	endfor
	set(Sparse_max4X4_pool_fig, "name", ["max4X4_", Sparse_filename_id]);
	saveas(Sparse_max4X4_pool_fig, ...
	       [Sparse_dir, filesep, "max4X4_", Sparse_filename_id], "png");
      endif

      if mean4X4_pool_flag
	Sparse_mean4X4_pool_fig = figure;
	axis off; box off;
	k_subplot = 0;
	Sparse_mean4X4_pool_hndl = zeros(ny_GT*4, nx_GT*4);
	Sparse_mean4X4_pool_axis = zeros(ny_GT*4, nx_GT*4);
	for j_yGT = 1 : ny_GT*4
	  for i_xGT = 1 : nx_GT*4
	    k_subplot = k_subplot + 1;
	    Sparse_mean4X4_pool_axis(j_yGT,i_xGT) = subplot(ny_GT*4, nx_GT*4, k_subplot);
	    mean4X4_pool_tmp = squeeze(Sparse_mean4X4_pool{num_Sparse_frames}(1+mod(j_yGT,4), 1+mod(i_xGT,4), :, ceil(j_yGT/4), ceil(i_xGT/4)));
	    ranked_mean4X4_pool = mean4X4_pool_tmp(Sparse_rank);
	    bar(ranked_mean4X4_pool(1:ceil(nf_Sparse/48):nf_Sparse));
	    axis tight;
	    axis off;
	    box off;
	    colormap("default")
	  endfor
	endfor
	set(Sparse_mean4X4_pool_fig, "name", ["mean4X4_", Sparse_filename_id]);
	saveas(Sparse_mean4X4_pool_fig, ...
	       [Sparse_dir, filesep, "mean4X4_", Sparse_filename_id], "png");
      endif

    endif  %% plot_Sparse_flag
    
    Sparse_hist_pool_array{i_Sparse} = Sparse_hist_pool;
    Sparse_times_array{i_Sparse} = Sparse_times;
    Sparse_max_pool_array{i_Sparse} = Sparse_max_pool;
    Sparse_mean_pool_array{i_Sparse} = Sparse_mean_pool;
    Sparse_max2X2_pool_array{i_Sparse} = Sparse_max2X2_pool;
    Sparse_mean2X2_pool_array{i_Sparse} = Sparse_mean2X2_pool;
    Sparse_max4X4_pool_array{i_Sparse} = Sparse_max4X4_pool;
    Sparse_mean4X4_pool_array{i_Sparse} = Sparse_mean4X4_pool;

    if num_Sparse_list > 10
      close(Sparse_hist_pool_fig);
    endif
    
  endfor  %% i_Sparse
  
endfunction
