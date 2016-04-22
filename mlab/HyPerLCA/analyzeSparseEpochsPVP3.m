function [Sparse_hdr, ...
	  Sparse_hist_rank_array, ...
	  Sparse_times_array, ...
	  Sparse_percent_active_array, ...
	  Sparse_percent_change_array, ...
	  Sparse_std_array, ...
	  Sparse_struct_array, ...
	  Sparse_max_val_array, ...
	  Sparse_min_val_array, ...
	  Sparse_mean_val_array, ...
	  Sparse_std_val_array, ...
	  Sparse_median_val_array] = ...
      analyzeSparseEpochsPVP3(Sparse_list, ...
			     output_dir, ...
			     load_Sparse_flag, ...
			     plot_Sparse_flag, ...
			     fraction_Sparse_frames_read, ...
			     min_Sparse_skip, ...
			     fraction_Sparse_progress, ...
			     Sparse_frames_list, ... % list of frames for which to return sparse activity
			     num_procs, ...
			     num_epochs)
  %% analyze sparse activity pvp file.  
  %% Sparse_hist_rank_array returns the rank order of neurons based on firing frequency, from most
  %% to least active.  
  %% Sparse_times_array gives the simTime for each sparse activity frames included in the analysis
  %% Sparse_percent_active_array gives the percentage of neurons active as a function of time in any given layer
  %% Sparse_percent_change_array gives the percentage change in sparse representation between successive frames,
  %% equal to the number of neurons whose activity changed over the number of neurons active in either frame
  
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
  if ~exist("num_procs") || isempty(num_procs)
    num_procs = 1;
  endif
  if ~exist("num_epochs") || isempty(num_epochs)
    num_epochs = 1;
  endif
  
  Sparse_hdr = [];
  Sparse_hist_rank_array = [];
  Sparse_times_array = [];
  Sparse_percent_active_array = [];
  Sparse_percent_change_array = [];
  Sparse_std_array = [];
  Sparse_max_val_array = [];
  Sparse_min_val_array = [];
  Sparse_mean_val_array = [];
  Sparse_std_val_array = [];
  Sparse_median_val_array = [];
  
  num_Sparse_list = size(Sparse_list,1);
  if num_Sparse_list ==0
    warning(["plotSparsePVP:num_Sparse_list == 0"]);
    return;
  endif
  Sparse_hdr = cell(num_Sparse_list,1);
  Sparse_hist_rank_array = cell(num_Sparse_list,1);
  Sparse_times_array = cell(num_Sparse_list,1);
  Sparse_percent_active_array = cell(num_Sparse_list,1);
  Sparse_percent_change_array = cell(num_Sparse_list,1);
  Sparse_std_array = cell(num_Sparse_list,1);
  Sparse_max_val_array = cell(num_Sparse_list,1);
  Sparse_min_val_array = cell(num_Sparse_list,1);
  Sparse_mean_val_array = cell(num_Sparse_list,1);
  Sparse_std_val_array = cell(num_Sparse_list,1);
  Sparse_median_val_array = cell(num_Sparse_list,1);
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
      Sparse_hist = zeros(nf_Sparse+1,1);
      Sparse_hist_bins = 1:nf_Sparse;
      
      n_Sparse_cell = cell(1);
      n_Sparse_cell{1} = n_Sparse;
      nf_Sparse_cell = cell(1);
      nf_Sparse_cell{1} = nf_Sparse;

      if num_epochs == 1

	[Sparse_struct, Sparse_hdr_tmp] = ...
	    readpvpfile(Sparse_file, progress_step, tot_Sparse_frames, num_Sparse_skip,1);
	num_Sparse_frames = size(Sparse_struct,1);
	Sparse_previous_struct = Sparse_struct(1:num_Sparse_frames-1);
	Sparse_struct = Sparse_struct(2:num_Sparse_frames);
      
	if num_procs == 1
	  [Sparse_times_list, ...
	   Sparse_percent_active_list, ...
	   Sparse_std_list, ...
	   Sparse_hist_frames_list, ...
	   Sparse_percent_change_list, ...
	   Sparse_max_val_list, ...
	   Sparse_min_val_list, ...
	   Sparse_mean_val_list, ...
	   Sparse_std_val_list, ...
	   Sparse_median_val_list] = ...
	      cellfun(@calcSparsePVPArray2, ...
		      Sparse_struct, ...
		      Sparse_previous_struct, ...
		      n_Sparse_cell, ...
		      nf_Sparse_cell, ...
		      "UniformOutput", false);
	elseif num_procs > 1
	  [Sparse_times_list, ...
	   Sparse_percent_active_list, ...
	   Sparse_std_list, ...
	   Sparse_hist_frames_list, ...
	   Sparse_percent_change_list, ...
	   Sparse_max_val_list, ...
	   Sparse_min_val_list, ...
	   Sparse_mean_val_list, ...
	   Sparse_std_val_list, ...
	   Sparse_median_val_list] = ...
	      parcellfun(num_procs, ...
			 @calcSparsePVPArray2, ...
			 Sparse_struct, ...
			 Sparse_previous_struct, ...
			 n_Sparse_cell, ...
			 nf_Sparse_cell, ...
			 "UniformOutput", false);
	endif  %% num_procs

 	Sparse_times = cell2mat(Sparse_times_list);
	Sparse_percent_active = cell2mat(Sparse_percent_active_list);
	Sparse_std = cell2mat(Sparse_std_list);
	if ~isempty(Sparse_hist_frames_list)
	  Sparse_hist_frames = cell2mat(Sparse_hist_frames_list);
	else
	  Sparse_hist_frames = zeros(1, nf_Sparse);
	endif
	Sparse_percent_change = cell2mat(Sparse_percent_change_list);
	Sparse_max_val = cell2mat(Sparse_max_val_list);
	Sparse_min_val = cell2mat(Sparse_min_val_list);
	Sparse_mean_val = cell2mat(Sparse_mean_val_list);
	Sparse_std_val = cell2mat(Sparse_std_val_list);
	Sparse_median_val = cell2mat(Sparse_median_val_list);

     else %% num_epochs

	num_frames_per_epoch = floor((tot_Sparse_frames-1)/num_epochs);
	for i_epoch = 1 : num_epochs

	  [Sparse_struct_epoch, Sparse_hdr_tmp] = ...
	      readpvpfile(Sparse_file, progress_step, i_epoch*num_frames_per_epoch, 1+(i_epoch-1)*num_frames_per_epoch,1);
	  num_Sparse_frames_epoch = size(Sparse_struct_epoch,1);
	  Sparse_previous_struct_epoch = Sparse_struct_epoch(1:num_Sparse_frames_epoch-1);
	  Sparse_struct_epoch = Sparse_struct_epoch(2:num_Sparse_frames_epoch);

 	  if num_procs == 1
	    [Sparse_times_list_epoch, ...
	     Sparse_percent_active_list_epoch, ...
	     Sparse_std_list_epoch, ...
	     Sparse_hist_frames_list_epoch, ...
	     Sparse_percent_change_list_epoch, ...
	     Sparse_max_val_list_epoch, ...
	     Sparse_min_val_list_epoch, ...
	     Sparse_mean_val_list_epoch, ...
	     Sparse_std_val_list_epoch, ...
	     Sparse_median_val_list_epoch] = ...
		cellfun(@calcSparsePVPArray2, ...
			Sparse_struct_epoch, ...
			Sparse_previous_struct_epoch, ...
			n_Sparse_cell, ...
			nf_Sparse_cell, ...
			"UniformOutput", false);
	  else
	    [Sparse_times_list_epoch, ...
	     Sparse_percent_active_list_epoch, ...
	     Sparse_std_list_epoch, ...
	     Sparse_hist_frames_list_epoch, ...
	     Sparse_percent_change_list_epoch, ...
	     Sparse_max_val_list_epoch, ...
	     Sparse_min_val_list_epoc, ...
	     Sparse_mean_val_list_epoch, ...
	     Sparse_std_val_list_epoch, ...
	     Sparse_median_val_list_epoch] = ...
		parcellfun(num_procs, ...
			   @calcSparsePVPArray2, ...
			   Sparse_struct_epoch, ...
			   Sparse_previous_struct_epoch, ...
			   n_Sparse_cell, ...
			   nf_Sparse_cell, ...
			   "UniformOutput", false);
	  endif %% num_procs	
	  if i_epoch == 1
	    Sparse_times = cell2mat(Sparse_times_list_epoch);
	    Sparse_percent_active = cell2mat(Sparse_percent_active_list_epoch);
	    Sparse_std = cell2mat(Sparse_std_list_epoch);
	    if ~isempty(Sparse_hist_frames_list_epoch)
	      Sparse_hist_frames = cell2mat(Sparse_hist_frames_list_epoch); %%reshape(cell2mat(Sparse_hist_frames_list_epoch), [nf_Sparse, length(Sparse_times_list_epoch)]);
	    else
	      Sparse_hist_frames = zeros(length(Sparse_times), nf_Sparse+1);
	    endif
	    Sparse_percent_change = full(cell2mat(Sparse_percent_change_list_epoch));
	    Sparse_max_val = cell2mat(Sparse_max_val_list_epoch);
	    Sparse_min_val = cell2mat(Sparse_min_val_list_epoch);
	    Sparse_mean_val = cell2mat(Sparse_mean_val_list_epoch);
	    Sparse_std_val = cell2mat(Sparse_std_val_list_epoch);
	    Sparse_median_val = cell2mat(Sparse_median_val_list_epoch);
	  else
	    Sparse_times = [Sparse_times; cell2mat(Sparse_times_list_epoch)];
	    Sparse_percent_active = [Sparse_percent_active; cell2mat(Sparse_percent_active_list_epoch)];
	    Sparse_std = [Sparse_std; cell2mat(Sparse_std_list_epoch)];
	    if ~isempty(Sparse_hist_frames_list_epoch)
	      Sparse_hist_frames = [Sparse_hist_frames; cell2mat(Sparse_hist_frames_list_epoch)]; %%reshape(cell2mat(Sparse_hist_frames_list_epoch), [nf_Sparse, length(Sparse_times_list_epoch)])];
	    endif
	    Sparse_percent_change = [Sparse_percent_change; full(cell2mat(Sparse_percent_change_list_epoch))];
	    Sparse_max_val = [Sparse_max_val; cell2mat(Sparse_max_val_list_epoch)];
	    Sparse_min_val = [Sparse_min_val; cell2mat(Sparse_min_val_list_epoch)];
	    Sparse_mean_val = [Sparse_mean_val; cell2mat(Sparse_mean_val_list_epoch)];
	    Sparse_std_val = [Sparse_std_val; cell2mat(Sparse_std_val_list_epoch)];
	    Sparse_median_val = [Sparse_median_val; cell2mat(Sparse_median_val_list_epoch)];
	  endif %% i_epoch == 1
	endfor %% i_epoch
      endif %% num_epochs

      num_Sparse_frames = size(Sparse_times,1);      
      Sparse_hist = sum(Sparse_hist_frames,1);
      Sparse_hist = Sparse_hist(1:nf_Sparse) / ((num_Sparse_frames) * nx_Sparse * ny_Sparse); 
      [Sparse_hist_sorted, Sparse_hist_rank] = sort(Sparse_hist(:), 1, "descend");
      
      Sparse_filename_id = [Sparse_list{i_Sparse,2}, "_", ...
			    num2str(Sparse_times(num_Sparse_frames), "%08d")];

      %% return sepcified sparse activity
      if exist("Sparse_frames_list") && size(Sparse_frames_list,1) >= i_Sparse
	Sparse_frames_times = Sparse_frames_list{i_Sparse};
	num_Sparse_frames_times = length(Sparse_frames_times(:));
	Sparse_intersect = [];
	num_tries = 0;
	max_tries = 1000;
	[Sparse_intersect, Sparse_frames_ndx, Sparse_times_ndx] = ...
	    intersect(Sparse_frames_times, Sparse_times);
	while isempty(Sparse_intersect) && num_tries < max_tries
	  Sparse_frames_times = Sparse_frames_times - 1;
	  num_tries = num_tries + 1;
	  [Sparse_intersect, Sparse_frames_ndx, Sparse_times_ndx] = ...
	      intersect(Sparse_frames_times, Sparse_times);
	endwhile
	if num_epochs == 1
	  Sparse_struct_array{i_Sparse} = Sparse_struct(Sparse_times_ndx);
	else
	  Sparse_struct_array{i_Sparse} = Sparse_struct_epoch(Sparse_times_ndx-((num_epochs-1)*num_frames_per_epoch));
	endif
      endif %% size(Sparse_frames_list,1) >= i_Sparse


      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_hist_bins_", Sparse_filename_id, ".mat"], ...
	   "Sparse_hist_bins");
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_hist_sorted_", Sparse_filename_id, ".mat"], ...
	   "Sparse_hist_sorted");
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_hist_rank_", Sparse_filename_id, ".mat"], ...
	   "Sparse_hist_rank");
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_percent_change_", Sparse_filename_id, ".mat"], ...
	   "Sparse_times", "Sparse_percent_change");    
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_percent_active_", Sparse_filename_id, ".mat"], ...
	   "Sparse_times", "Sparse_percent_active");	 
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_std_", Sparse_filename_id, ".mat"], ...
	   "Sparse_times", "Sparse_std");	 
      save("-mat", ...
	   [Sparse_dir, filesep, "Sparse_vals_", Sparse_filename_id, ".mat"], ...
	   "Sparse_times", "Sparse_max_val", "Sparse_min_val", "Sparse_mean_val", "Sparse_std_val", "Sparse_median_val");	 

    else  %% load Sparse data structures from file


      Sparse_hist_bins_str = ...
	  [Sparse_dir, filesep, "Sparse_hist_bins_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_hist_bins_glob = glob(Sparse_hist_bins_str);
      num_Sparse_hist_bins_glob = length(Sparse_hist_bins_glob);
      if num_Sparse_hist_bins_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_hist_bins_str]);
	break;
      endif
      load("-mat", Sparse_hist_bins_glob{num_Sparse_hist_bins_glob});
      Sparse_hist_sorted_str = ...
	  [Sparse_dir, filesep, "Sparse_hist_sorted_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_hist_sorted_glob = glob(Sparse_hist_sorted_str);
      num_Sparse_hist_sorted_glob = length(Sparse_hist_sorted_glob);
      if num_Sparse_hist_sorted_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_hist_sorted_str]);
	break;
      endif
      load("-mat", Sparse_hist_sorted_glob{num_Sparse_hist_sorted_glob});
      Sparse_hist_rank_str = ...
	  [Sparse_dir, filesep, "Sparse_hist_rank_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_hist_rank_glob = glob(Sparse_hist_rank_str);
      num_Sparse_hist_rank_glob = length(Sparse_hist_rank_glob);
      if num_Sparse_hist_rank_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_hist_rank_str]);
	break;
      endif
      load("-mat", Sparse_hist_rank_glob{num_Sparse_hist_rank_glob});
      Sparse_percent_change_str = ...
	  [Sparse_dir, filesep, "Sparse_percent_change_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_percent_change_glob = glob(Sparse_percent_change_str);
      num_Sparse_percent_change_glob = length(Sparse_percent_change_glob);
      if num_Sparse_percent_change_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_percent_change_str]);
	break;
      endif
      load("-mat", Sparse_percent_change_glob{num_Sparse_percent_change_glob});
      Sparse_percent_active_str = ...
	  [Sparse_dir, filesep, "Sparse_percent_active_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_percent_active_glob = glob(Sparse_percent_active_str);
      num_Sparse_percent_active_glob = length(Sparse_percent_active_glob);
      if num_Sparse_percent_active_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_percent_active_str]);
	break;
      endif
      load("-mat", Sparse_percent_active_glob{num_Sparse_percent_active_glob});
      Sparse_std_str = ...
	  [Sparse_dir, filesep, "Sparse_std_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_std_glob = glob(Sparse_std_str);
      num_Sparse_std_glob = length(Sparse_std_glob);
      if num_Sparse_std_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_std_str]);
	break;
      endif
      load("-mat", Sparse_std_glob{num_Sparse_std_glob});
      Sparse_vals_str = ...
	  [Sparse_dir, filesep, "Sparse_vals_", Sparse_list{i_Sparse,2}, "_", "*", ".mat"];
      Sparse_vals_glob = glob(Sparse_vals_str);
      num_Sparse_vals_glob = length(Sparse_vals_glob);
      if num_Sparse_vals_glob <= 0
	warning(["load_Sparse_flag is true but no files to load in: ", Sparse_vals_str]);
	break;
      endif
      load("-mat", Sparse_vals_glob{num_Sparse_vals_glob});

      Sparse_filename_id = [Sparse_list{i_Sparse,2}, "_", ...
			    num2str(Sparse_times(end), "%08d")];

    endif %% load_Sparse_flag

    num_Sparse_frames = length(Sparse_times);
    if plot_Sparse_flag 
      Sparse_hist_fig = figure;
      Sparse_hist_hndl = bar(Sparse_hist_bins(:), Sparse_hist_sorted(:)); axis tight;
      set(Sparse_hist_fig, "name", ["Hist_", Sparse_filename_id]);
      saveas(Sparse_hist_fig, ...
	     [Sparse_dir, filesep, "Hist_", Sparse_filename_id], "png");
      
      Sparse_percent_change_fig = figure;
      Sparse_percent_change_hndl = plot(Sparse_times, full(Sparse_percent_change)); 
      set(Sparse_percent_change_hndl, "linewidth", 1.5);
      axis([Sparse_times(1) Sparse_times(end) 0 1]); %%axis tight;
      set(Sparse_percent_change_fig, ...
	  "name", ["percent_change_", Sparse_filename_id]);
      saveas(Sparse_percent_change_fig, ...
	     [Sparse_dir, filesep, "percent_change_", Sparse_filename_id], "png");
      
      Sparse_percent_active_fig = figure;
      Sparse_percent_active_hndl = plot(Sparse_times, Sparse_percent_active); axis tight;
      set(Sparse_percent_active_hndl, "linewidth", 1.5);
      set(Sparse_percent_active_fig, "name", ["percent_active_", Sparse_list{i_Sparse,2}, "_", ...
					      num2str(Sparse_times(num_Sparse_frames), "%08d")]);
      saveas(Sparse_percent_active_fig, ...
	     [Sparse_dir, filesep, "percent_active_", Sparse_list{i_Sparse,2}, "_", ...
	      num2str(Sparse_times(num_Sparse_frames), "%08d")], "png");
    endif  %% plot_Sparse_flag 
    
    Sparse_median_active = median(Sparse_percent_active(:));
    disp([Sparse_filename_id, ...
	  " median_active = ", num2str(Sparse_median_active)]);
    
    Sparse_mean_percent_change = mean(Sparse_percent_change(:));
    disp([Sparse_filename_id, ...
	  " mean_percent_change = ", num2str(Sparse_mean_percent_change)]);
    Sparse_hist_rank_array{i_Sparse} = Sparse_hist_rank;
    Sparse_times_array{i_Sparse} = Sparse_times;
    Sparse_percent_active_array{i_Sparse} = Sparse_percent_active;
    Sparse_percent_change_array{i_Sparse} = Sparse_percent_change;
    Sparse_std_array{i_Sparse} = Sparse_std;
    Sparse_max_val_array{i_Sparse} = Sparse_max_val;
    Sparse_min_val_array{i_Sparse} = Sparse_min_val;
    Sparse_mean_val_array{i_Sparse} = Sparse_mean_val;
    Sparse_std_val_array{i_Sparse} = Sparse_std_val;
    Sparse_median_val_array{i_Sparse} = Sparse_median_val;
    disp([Sparse_filename_id, ...
	  " Sparse_max_val = ", num2str(max(Sparse_max_val(:)))]);
    disp([Sparse_filename_id, ...
	  " Sparse_min_val = ", num2str(min(Sparse_min_val(:)))]);
    disp([Sparse_filename_id, ...
	  " Sparse_mean_val = ", num2str(mean(Sparse_mean_val(:)))]);
    disp([Sparse_filename_id, ...
	  " Sparse_std_val = ", num2str(std(Sparse_std_val(:)))]);
    disp([Sparse_filename_id, ...
	  " Sparse_median_val = ", num2str(std(Sparse_median_val(:)))]);

    if num_Sparse_list > 10
      close(Sparse_hist_fig);
      close(Sparse_percent_change_fig);
      close(Sparse_percent_active_fig);
    endif
    
  endfor  %% i_Sparse
  
endfunction
