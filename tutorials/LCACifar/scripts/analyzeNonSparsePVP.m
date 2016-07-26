function [nonSparse_times_array, ...
	  nonSparse_RMS_array, ...
	  nonSparse_norm_RMS_array, ...
	  nonSparse_RMS_fig] = ...
      analyzeNonSparsePVP(nonSparse_list, ...
		       nonSparse_skip, ...
		       nonSparse_norm_list, ...
		       nonSparse_norm_strength, ...
		       Sparse_times_array, ...
		       Sparse_std_array, ...
		       Sparse_std_ndx, ...
		       output_dir, ...
		       plot_nonSparse_flag, ...
		       fraction_nonSparse_frames_read, ...
		       min_nonSparse_skip, ...
		       fraction_nonSparse_progress)
  %% compute the normalized error 
  %% the error is expressed as a std normalized by the std of the target input to the error layer.
  %% Error layers are passed as a list of file names using nonSparse_list.
  %% Layer activity for normalizing the error
  %% are passed using the nonSparse_norm_list, which should have the same
  %% size as nonSparse_list.
  %% if the target input is a sparse file, the sparse activity is passed using Sparse_std_array
  %% with the  index corresponding to the given Error layer indicated by Sparse_std_ndx.
  %% Set Sparse_std_ndx to zero for any layers normalized by non sparse activity stored in the
  %% files listed in nonSparse_norm_list.


  if ~exist("Sparse_std_ndx")
    Sparse_std_ndx = zeros(num_nonSparse_list,1);
  endif
  if ~exist("plot_nonSparse_flag") || isempty(plot_nonSparse_flag)
    plot_nonSparse_flag = false;
  endif
  if ~exist("min_nonSparse_skip") || isempty(min_nonSparse_skip)
    min_nonSparse_skip = 0;  %% 0 -> skip no frames
  endif
  if ~exist("fraction_nonSparse_frames_read") || isempty(fraction_nonSparse_frames_read)
    fraction_nonSparse_frames_read = 1;  %% 1 -> read all frames
  endif
  if ~exist("fraction_nonSparse_progress") || isempty(fraction_nonSparse_progress)
    fraction_nonSparse_progress = 10;
  endif

  nonSparse_times_array = [];
  nonSparse_RMS_array = [];
  nonSparse_norm_RMS_array = [];
  num_nonSparse_list = size(nonSparse_list,1);
  if num_nonSparse_list ==0
    warning(["analyzeNonSparsePVP:num_nonSparse_list == 0"]);
    return;
  endif
  if ~exist("nonSparse_skip") || isempty(nonSparse_skip)
    nonSparse_skip = ones(num_nonSparse_list,1);
  endif
  nonSparse_norm_skip = nonSparse_skip;

  nonSparse_times_array = cell(num_nonSparse_list,1);
  nonSparse_RMS_array = cell(num_nonSparse_list,1);
  nonSparse_norm_RMS_array = cell(num_nonSparse_list,1);

  nonSparse_hdr = cell(num_nonSparse_list,1);
  nonSparse_dir = [output_dir, filesep, "nonSparse"]
  [status, msg, msgid] = mkdir(nonSparse_dir);
  if status ~= 1
    warning(["mkdir(", nonSparse_dir, ")", " msg = ", msg]);
  endif 
  nonSparse_RMS_fig = zeros(num_nonSparse_list,1);
  %%max_nonSparse_RMS = zeros(num_nonSparse_list,1);
  %%min_nonSparse_RMS = zeros(num_nonSparse_list,1);
  for i_nonSparse = 1 : num_nonSparse_list
    nonSparse_file = ...
	[output_dir, filesep, nonSparse_list{i_nonSparse,1}, nonSparse_list{i_nonSparse,2}, ".pvp"]
    if ~exist(nonSparse_file, "file")
      warning(["file does not exist: ", nonSparse_file]);
      continue;
    endif
    nonSparse_fid = fopen(nonSparse_file);
    nonSparse_hdr{i_nonSparse} = readpvpheader(nonSparse_fid);
    fclose(nonSparse_fid);
    tot_nonSparse_frames = nonSparse_hdr{i_nonSparse}.nbands;
  
    %% number of activity frames to analyze, counting backward from last frame, maximum is tot_Sparse_frames
    num_nonSparse_skip = ...
	tot_nonSparse_frames - fix(tot_nonSparse_frames/fraction_nonSparse_frames_read);  
    num_nonSparse_skip = max(num_nonSparse_skip, min_nonSparse_skip);
    progress_step = ceil(tot_nonSparse_frames / fraction_nonSparse_progress);

    [nonSparse_struct, nonSparse_hdr_tmp] = ...
	readpvpfile(nonSparse_file, ...
		    progress_step, ...
		    tot_nonSparse_frames, ...
		    num_nonSparse_skip, ...
		    nonSparse_skip(i_nonSparse));
    num_nonSparse_frames = size(nonSparse_struct,1);
    while ~isstruct(nonSparse_struct{num_nonSparse_frames})
      num_nonSparse_frames = num_nonSparse_frames - 1;
    endwhile

    %% assign defaults for no normalization of error
    nonSparse_norm_file = "";
    num_nonSparse_norm_frames = num_nonSparse_frames;
    nonSparse_norm_struct = [];
    if Sparse_std_ndx(i_nonSparse) == 0 && ...
	  (~isempty(nonSparse_norm_list{i_nonSparse,1}) || ...
	   ~isempty(nonSparse_norm_list{i_nonSparse,2})) %% read normalization from pvp file
      nonSparse_norm_file = ...
	  [output_dir, filesep, nonSparse_norm_list{i_nonSparse,1}, ...
	   nonSparse_norm_list{i_nonSparse,2}, ".pvp"]
      if ~exist(nonSparse_norm_file, "file")
	warning(["AnalyzeNonSparePVP::file does not exist: ", nonSparse_norm_file]);
	continue;
      else 
	nonSparse_norm_fid = fopen(nonSparse_norm_file);
	nonSparse_norm_hdr{i_nonSparse} = readpvpheader(nonSparse_norm_fid);
	fclose(nonSparse_norm_fid);
	tot_nonSparse_norm_frames = nonSparse_norm_hdr{i_nonSparse}.nbands;
	
	fraction_nonSparse_norm_frames_read = fraction_nonSparse_frames_read;
	min_nonSparse_norm_skip = min_nonSparse_skip;
	num_nonSparse_norm_skip = ...
	    tot_nonSparse_norm_frames - fix(tot_nonSparse_norm_frames/fraction_nonSparse_norm_frames_read);  
	num_nonSparse_norm_skip = max(num_nonSparse_norm_skip, min_nonSparse_norm_skip);
	progress_step = ceil(tot_nonSparse_norm_frames / fraction_nonSparse_progress);
	
	[nonSparse_norm_struct, nonSparse_norm_hdr_tmp] = ...
	    readpvpfile(nonSparse_norm_file, ...
			progress_step, ...
			tot_nonSparse_norm_frames, ...
			num_nonSparse_norm_skip, ...
			nonSparse_norm_skip(i_nonSparse));
	
	num_nonSparse_norm_frames = size(nonSparse_norm_struct,1);
	while ~isstruct(nonSparse_norm_struct{num_nonSparse_norm_frames})
	  num_nonSparse_norm_frames = num_nonSparse_norm_frames - 1;
	endwhile
      endif %% ~isempty(nonSparse_norm_list{i_nonSparse,1})
    elseif Sparse_std_ndx(i_nonSparse) > 0 %% get normalization from Sparse activity
      num_nonSparse_norm_frames = length(Sparse_times_array{Sparse_std_ndx(i_nonSparse)});
      nonSparse_norm_struct = cell(num_nonSparse_norm_frames,1);
      for i_frame = 1 : 1 : num_nonSparse_norm_frames
	nonSparse_norm_struct{i_frame}.time = Sparse_times_array{Sparse_std_ndx(i_nonSparse)}(i_frame);
      endfor
    endif

    %% get Error and normalization for each frame
    %% use normalization frame with time stamp closest to time stamp of Error frame
    %% frame_diff stores the maximum difference in time stamps
    frame_diff = 0;
    frame_diff2 = zeros(4,1);
    if  ~isempty(nonSparse_norm_struct) && num_nonSparse_norm_frames > 0
      frame_diff2(1) = ...
	  nonSparse_struct{num_nonSparse_frames-1}.time - ...
	       nonSparse_norm_struct{num_nonSparse_norm_frames-1}.time;
      frame_diff2(2) = ...
	  nonSparse_struct{num_nonSparse_frames}.time - ...
	       nonSparse_norm_struct{num_nonSparse_norm_frames-1}.time;
      frame_diff2(3) = ...
	  nonSparse_struct{num_nonSparse_frames-1}.time - ...
	       nonSparse_norm_struct{num_nonSparse_norm_frames}.time;
      frame_diff2(4) = ...
	  nonSparse_struct{num_nonSparse_frames}.time - ...
	       nonSparse_norm_struct{num_nonSparse_norm_frames}.time;
      [abs_frame_diff, frame_diff_ndx] = min(abs(frame_diff2(:)));
      frame_diff = frame_diff2(frame_diff_ndx);
    endif

    min_nonSparse_frames = min(num_nonSparse_frames,num_nonSparse_norm_frames);
    nonSparse_times = zeros((min_nonSparse_frames),1);
    nonSparse_RMS = zeros((min_nonSparse_frames),1);
    nonSparse_norm_RMS = ones(min_nonSparse_frames,1);
    nonSparse_norm_times = zeros(min_nonSparse_frames,1);
    for i_frame = 1 : min_nonSparse_frames

      if ~isempty(nonSparse_struct{i_frame})
	nonSparse_times(i_frame) = squeeze(nonSparse_struct{i_frame}.time);
	nonSparse_vals = squeeze(nonSparse_struct{i_frame}.values);
	nonSparse_RMS(i_frame) = std(nonSparse_vals(:));

	if ~isempty(nonSparse_norm_struct) 
	  j_frame = min(i_frame, num_nonSparse_norm_frames);
	  nonSparse_norm_time = nonSparse_norm_struct{j_frame}.time;
	  nonSparse_time_shift = 0;
	  while (nonSparse_norm_time - nonSparse_times(i_frame)) > abs_frame_diff && ...
		(i_frame-(nonSparse_time_shift+1)) >= 1
	    nonSparse_time_shift = nonSparse_time_shift + 1;
	    nonSparse_norm_time = nonSparse_norm_struct{i_frame-nonSparse_time_shift}.time;
	  endwhile
	  while (nonSparse_norm_time - nonSparse_times(i_frame)) < -abs_frame_diff && ...
		(i_frame-(nonSparse_time_shift-1)) <= num_nonSparse_norm_frames
	    nonSparse_time_shift = nonSparse_time_shift - 1;
	    nonSparse_norm_time = nonSparse_norm_struct{i_frame-nonSparse_time_shift}.time;
	  endwhile
	  if (i_frame-nonSparse_time_shift) < 1 || (i_frame-nonSparse_time_shift) > num_nonSparse_norm_frames
	    last_frame = i_frame - 1;
	    num_nonSparse_frames = i_frame - 1;
	    nonSparse_times = nonSparse_times(1:num_nonSparse_frames);
	    nonSparse_RMS = nonSparse_RMS(1:num_nonSparse_frames);
	    nonSparse_norm_RMS = nonSparse_norm_RMS(1:num_nonSparse_frames);
	    break;
	  else
	    if Sparse_std_ndx(i_nonSparse) == 0
	      nonSparse_norm_vals = ...
		  squeeze(nonSparse_norm_struct{i_frame-nonSparse_time_shift}.values);
	      nonSparse_norm_RMS(i_frame) = ...
		  sqrt(mean(nonSparse_norm_strength(i_nonSparse).^2 * nonSparse_norm_vals(:).^2));
	    else
	      nonSparse_norm_RMS(i_frame) = ...
		  Sparse_std_array{Sparse_std_ndx(i_nonSparse)}(i_frame-nonSparse_time_shift);
	    endif
	  endif
	endif %% ~isempty(nonSparse_norm_struct)

      else %% isempty(nonSparse_struct{i_frame})
	num_nonSparse_frames = i_frame - 1;
	nonSparse_times = nonSparse_times(1:(num_nonSparse_frames));
	nonSparse_RMS = nonSparse_RMS(1:(num_nonSparse_frames));
	break;
      endif %% ~isempty(nonSparse_struct{i_frame})

    endfor %% i_frame

    num_nonSparse_frames = size(nonSparse_times,1);
    if num_nonSparse_frames <= 0
      disp(["num_nonSparse_frames = ", num2str(num_nonSparse_frames)]);
      continue;
    endif

    normalized_nonSparse_RMS = ...
	nonSparse_RMS(1:num_nonSparse_frames) ./ ...
	(nonSparse_norm_RMS(1:num_nonSparse_frames) + (nonSparse_norm_RMS(1:num_nonSparse_frames)==0));
    nonSparse_RMS_filename = ...
	 ["RMS_", nonSparse_list{i_nonSparse,1}, nonSparse_list{i_nonSparse,2}, ...
	 "_", num2str(nonSparse_times(num_nonSparse_frames), "%08d")];
    nonSparse_RMS_pathname = ... 
	[nonSparse_dir, filesep, nonSparse_RMS_filename];
    save("-mat", ...
	 [nonSparse_RMS_pathname, ".mat"], ...
	 "nonSparse_times", "normalized_nonSparse_RMS");    

    if plot_nonSparse_flag
      nonSparse_RMS_fig(i_nonSparse) = figure;
      nonSparse_RMS_hndl = plot(nonSparse_times, normalized_nonSparse_RMS); 
      set(nonSparse_RMS_hndl, "linewidth", 1.5);
      %%max_nonSparse_RMS(i_nonSparse) = median(normalized_nonSparse_RMS(:)) + 2*std(normalized_nonSparse_RMS(:));
      %%min_nonSparse_RMS(i_nonSparse) = 0.0; %%min(normalized_nonSparse_RMS(:));
      %%axis([nonSparse_times(1) nonSparse_times(num_nonSparse_frames) ...
      %%   min_nonSparse_RMS(i_nonSparse) max_nonSparse_RMS(i_nonSparse)]);
      hold on
      set(nonSparse_RMS_fig(i_nonSparse), "name", ...
	  [nonSparse_RMS_filename]);
      saveas(nonSparse_RMS_fig(i_nonSparse), ...
	     nonSparse_RMS_pathname, "png");
      if num_nonSparse_list > 10
	close(nonSparse_RMS_fig(i_nonSparse))
      endif
    endif
    nonSparse_median_RMS = ...
	median(nonSparse_RMS(1:num_nonSparse_frames) ./ ...
	       (nonSparse_norm_RMS(1:num_nonSparse_frames) + (nonSparse_norm_RMS(1:num_nonSparse_frames)==0)));
    disp([nonSparse_list{i_nonSparse,2}, "_", num2str(nonSparse_times(num_nonSparse_frames), "%i"), ...
	  " median RMS = ", num2str(nonSparse_median_RMS)]);
    nonSparse_times_array{i_nonSparse} = nonSparse_times(1:num_nonSparse_frames);
    nonSparse_RMS_array{i_nonSparse} = nonSparse_RMS(1:num_nonSparse_frames);
    nonSparse_norm_RMS_array{i_nonSparse} = nonSparse_norm_RMS(1:num_nonSparse_frames);
    %%keyboard;

  endfor  %% i_nonSparse

endfunction
