function [ReconError_times_array, ...
	  ReconError_RMS_array, ...
	  ReconError_norm_RMS_array, ...
	  ReconError_RMS_fig] = ...
      analyzeReconErrorPVP(ReconError_list, ...
			   ReconError_skip, ...
			   ReconError_norm_list, ...
			   ReconError_norm_strength, ...
			   ReconError_times_array, ...
			   ReconError_RMS_array, ...
			   ReconError_norm_RMS_array, ...
			   nonSparse_RMS_fig, ...
			   ReconError_RMS_fig_ndx, ...
			   output_dir, ...
			   plot_ReconError_flag, ...
			   fraction_ReconError_frames_read, ...
			   min_ReconError_skip, ...
			   fraction_ReconError_progress)
  %% compute the difference between a reconstructed image and the original image
  %% the difference is reported as a std and normalized by the std of the original image
  %% ReconError_list stores a list of file names for the reconstructed images
  %% ReconError_norm_list stores the list of original images
  %% Results are returned as a cell array, that may be appended to the cell arrays passed in as arguments
  %% Graphical output is produced if plot_ReconError_flag is true.
  %% Plots can be overlaid on the figures whose handles are passed in using nonSparse_RMS_fig depending
  %% on whether ReconError_RMS_fig_ndx > 0.  
  %% ReconError_RMS_fig_ndx(i_ReconError = [1:size(ReconError_list,1)]), if nonzero, determines
  %% on which nonSparse_RMS_fig a given normalized reconstruction error is overlaid.
  

  if ~exist("plot_ReconError_flag") || isempty(plot_ReconError_flag)
    plot_ReconError_flag = false;
  endif
  if ~exist("min_ReconError_skip") || isempty(min_ReconError_skip)
    min_ReconError_skip = 0;  %% 0 -> skip no frames
  endif
  if ~exist("fraction_ReconError_frames_read") || isempty(fraction_ReconError_frames_read)
    fraction_ReconError_frames_read = 1;  %% 1 -> read all frames
  endif
  if ~exist("fraction_ReconError_progress") || isempty(fraction_ReconError_progress)
    fraction_ReconError_progress = 10;
  endif

  fraction_ReconError_norm_frames_read = fraction_ReconError_frames_read;
  ReconError_norm_skip = ReconError_skip;
  fraction_ReconError_norm_progress = fraction_ReconError_progress;      
  min_ReconError_norm_skip = min_ReconError_skip;

  num_ReconError_list = size(ReconError_list,1);
  ReconError_colormap = jet(num_ReconError_list+1);
  if num_ReconError_list ==0
    warning(["analyzeNonSparsePVP:num_ReconError_list == 0"]);
    return;
  endif
  if ~exist("ReconError_skip") || isempty(ReconError_skip) || size(ReconError_skip,1) < num_ReconError_list
    ReconError_skip = ones(num_ReconError_list,1);
  endif

  if ~exist("nonSparse_RMS_fig") || isempty(nonSparse_RMS_fig) 
    ReconError_RMS_fig_ndx = zeros(num_ReconError_list,1);
  endif
  if ~exist("ReconError_RMS_fig_ndx") || isempty(ReconError_RMS_fig_ndx)
    ReconError_RMS_fig_ndx = zeros(num_ReconError_list,1);
  endif
  num_ReconError_RMS_fig_ndx = length(ReconError_RMS_fig_ndx(:));
  for i_ReconError = (num_ReconError_RMS_fig_ndx + 1) : num_ReconError_list
    ReconError_RMS_fig_ndx(i_ReconError) = 0;
  endfor
  for i_ReconError = 1 : num_ReconError_list
    if ReconError_RMS_fig_ndx(i_ReconError) > length(nonSparse_RMS_fig(:))
	warning(["ReconError_RMS_fig_ndx(i_ReconError) > size(ReconError_RMS_fig,1)", "\n", ...
		 "i_ReconError = ", num2str(i_ReconError), "\n", ...
		 "ReconError_RMS_fig_ndx = ", num2str(ReconError_RMS_fig_ndx(i_ReconError)), "\n", ...
		 "size(ReconError_RMS_fig,1) = ", num2str(size(ReconError_RMS_fig,1))]);
      ReconError_RMS_fig_ndx(i_ReconError) = 0;
    endif
  endfor
  disp(["ReconError_RMS_fig_ndx = ", mat2str(ReconError_RMS_fig_ndx)]);
  ReconError_RMS_fig = zeros(num_ReconError_list,1);


  init_ReconError_list = size(ReconError_RMS_array,1);
  if init_ReconError_list == 0 || isempty(ReconError_times_array)
    ReconError_times_array = cell(num_ReconError_list,1);
  else
    ReconError_times_array = [ReconError_times_array; cell(num_ReconError_list,1)];
  endif
  if isempty(ReconError_RMS_array)
    ReconError_RMS_array = cell(num_ReconError_list,1);
  else
    ReconError_RMS_array = [ReconError_RMS_array; cell(num_ReconError_list,1)];
  endif
  if isempty(ReconError_norm_RMS_array)
    ReconError_norm_RMS_array = cell(num_ReconError_list,1);
  else
    ReconError_norm_RMS_array = [ReconError_norm_RMS_array; cell(num_ReconError_list,1)];
  endif

  %% num frames to skip between stored frames, default is 
  if ~exist(output_dir,"dir")
    error(["analyzeReconErrorPVP::output_dir does not exist: ", output_dir]);
  endif
  ReconError_hdr = cell(num_ReconError_list,1);
  ReconError_dir = [output_dir, filesep, "ReconError"]
  [status, msg, msgid] = mkdir(ReconError_dir);
  if status ~= 1
    warning(["mkdir(", ReconError_dir, ")", " msg = ", msg]);
  endif 
  for i_ReconError = 1 : num_ReconError_list
    ReconError_file = [output_dir, filesep, ...
		       ReconError_list{i_ReconError,1}, ReconError_list{i_ReconError,2}, ".pvp"]
    if ~exist(ReconError_file, "file")
      warning(["file does not exist: ", ReconError_file]);
      continue;
    endif
    ReconError_fid = fopen(ReconError_file);
    ReconError_hdr{i_ReconError} = readpvpheader(ReconError_fid);
    fclose(ReconError_fid);
    tot_ReconError_frames = ReconError_hdr{i_ReconError}.nbands;

    %% number of activity frames to analyze, counting backward from last frame, maximum is tot_Sparse_frames
    num_ReconError_skip = ...
	tot_ReconError_frames - fix(tot_ReconError_frames/fraction_ReconError_frames_read);  
    num_ReconError_skip = max(num_ReconError_skip, min_ReconError_skip);
    progress_step = ceil(tot_ReconError_frames / fraction_ReconError_progress);

    [ReconError_struct, ReconError_hdr_tmp] = ...
	readpvpfile(ReconError_file, ...
		    progress_step, ...
		    tot_ReconError_frames, ...
		    num_ReconError_skip, ...
		    ReconError_skip(i_ReconError));
    num_ReconError_frames = size(ReconError_struct,1);
    while ~isstruct(ReconError_struct{num_ReconError_frames})
      num_ReconError_frames = num_ReconError_frames - 1;
    endwhile

    %% assign defaults for no normalization of error
    ReconError_norm_file = "";
    num_ReconError_norm_frames = num_ReconError_frames;
    ReconError_norm_RMS = ones(num_ReconError_frames,1);
    ReconError_norm_times = zeros(num_ReconError_frames,1);
    ReconError_norm_struct = [];
    if ~isempty(ReconError_norm_list{i_ReconError,1}) || ...
	  ~isempty(ReconError_norm_list{i_ReconError,2})
      ReconError_norm_file = ...
	  [output_dir, filesep, ...
	   ReconError_norm_list{i_ReconError,1}, ...
	   ReconError_norm_list{i_ReconError,2}, ".pvp"]
      if ~exist(ReconError_norm_file, "file")
	warning(["AnalyzeReconErrorPVP::file does not exist: ", ReconError_norm_file]);
	continue;
      endif
      ReconError_norm_fid = fopen(ReconError_norm_file);
      ReconError_norm_hdr{i_ReconError} = readpvpheader(ReconError_norm_fid);
      fclose(ReconError_norm_fid);
      tot_ReconError_norm_frames = ReconError_norm_hdr{i_ReconError}.nbands;

      num_ReconError_norm_skip = ...
	  tot_ReconError_norm_frames - fix(tot_ReconError_norm_frames/fraction_ReconError_norm_frames_read);  
      num_ReconError_norm_skip = max(num_ReconError_norm_skip, min_ReconError_norm_skip);
      progress_step = ceil(tot_ReconError_norm_frames / fraction_ReconError_norm_progress);
      [ReconError_norm_struct, ReconError_norm_hdr_tmp] = ...
	  readpvpfile(ReconError_norm_file, ...
		      progress_step, ...
		      tot_ReconError_norm_frames, ...
		      num_ReconError_norm_skip, ...
		      ReconError_norm_skip(i_ReconError));      
      num_ReconError_norm_frames = size(ReconError_norm_struct,1);
      while ~isstruct(ReconError_norm_struct{num_ReconError_norm_frames})
	num_ReconError_norm_frames = num_ReconError_norm_frames - 1;
      endwhile
    else
      ReconError_norm_vals = zeros(num_ReconError_frames,1);
      ReconError_norm_struct = [];
    endif

    %%keyboard;
    %% get Error and normalization for each frame
    %% use normalization frame with time stamp closest to time stamp of Error frame
    %% frame_diff stores the maximum difference in time stamps
    frame_diff = 0;
    frame_diff2 = zeros(4,1);
    if  ~isempty(ReconError_norm_struct) && num_ReconError_norm_frames > 0
      frame_diff2(1) = ...
	  ReconError_struct{num_ReconError_frames-1}.time - ...
	       ReconError_norm_struct{num_ReconError_norm_frames-1}.time;
      frame_diff2(2) = ...
	  ReconError_struct{num_ReconError_frames}.time - ...
	       ReconError_norm_struct{num_ReconError_norm_frames-1}.time;
      frame_diff2(3) = ...
	  ReconError_struct{num_ReconError_frames-1}.time - ...
	       ReconError_norm_struct{num_ReconError_norm_frames}.time;
      frame_diff2(4) = ...
	  ReconError_struct{num_ReconError_frames}.time - ...
	       ReconError_norm_struct{num_ReconError_norm_frames}.time;
      [abs_frame_diff, frame_diff_ndx] = min(abs(frame_diff2(:)));
      frame_diff = frame_diff2(frame_diff_ndx);
    endif

    %%keyboard;
    min_ReconError_frames = min(num_ReconError_frames,num_ReconError_norm_frames);
    ReconError_times = zeros((min_ReconError_frames),1);
    ReconError_RMS = zeros((min_ReconError_frames),1);
    for i_frame = 1 : 1 : min_ReconError_frames

      if ~isempty(ReconError_struct{i_frame})
	ReconError_times(i_frame) = squeeze(ReconError_struct{i_frame}.time);
	ReconError_vals = squeeze(ReconError_struct{i_frame}.values);

	if ~isempty(ReconError_norm_struct) 
	  j_frame = min(i_frame, num_ReconError_norm_frames);
	  ReconError_norm_time = ReconError_norm_struct{j_frame}.time;
	  ReconError_time_shift = 0;
	  while (ReconError_norm_time - ReconError_times(i_frame)) > abs_frame_diff && ...
		(i_frame-(ReconError_time_shift+1)) >= 1
	    ReconError_time_shift = ReconError_time_shift + 1;
	    ReconError_norm_time = ReconError_norm_struct{i_frame-ReconError_time_shift}.time;
	  endwhile
	  while (ReconError_norm_time - ReconError_times(i_frame)) < -abs_frame_diff && ...
		(i_frame-(ReconError_time_shift-1)) <= num_ReconError_norm_frames
	    ReconError_time_shift = ReconError_time_shift - 1;
	    ReconError_norm_time = ReconError_norm_struct{i_frame-ReconError_time_shift}.time;
	  endwhile

	  ReconError_norm_vals = squeeze(ReconError_norm_struct{i_frame-ReconError_time_shift}.values);
	  if (i_frame-ReconError_time_shift) < 1 || (i_frame-ReconError_time_shift) > num_ReconError_norm_frames
	    last_frame = i_frame - 1;
	    num_ReconError_frames = i_frame - 1;
	    ReconError_times = ReconError_times(1:num_ReconError_frames);
	    ReconError_RMS = ReconError_RMS(1:num_ReconError_frames);
	    ReconError_norm_RMS = ReconError_norm_RMS(1:num_ReconError_frames);
	    break;
	  else
	    downsample_factor_row = ceil(size(ReconError_norm_vals,1)/size(ReconError_vals,1));
	    downsample_factor_col = ceil(size(ReconError_norm_vals,2)/size(ReconError_vals,2));
	    downsample_factor_feature = ceil(size(ReconError_norm_vals,3)/size(ReconError_vals,3));
	    ReconError_norm_vals_downsampled = ReconError_norm_vals(1:downsample_factor_row:end,1:downsample_factor_col:end,:);
	    ReconError_RMS(i_frame) = ...
		std(ReconError_vals(:) - ReconError_norm_strength(i_ReconError) * ReconError_norm_vals_downsampled(:)) / ...
		(std(ReconError_norm_strength(i_ReconError) * ...
			  (ReconError_norm_vals(:) + all(ReconError_norm_vals(:)==0))));
%5		sqrt(mean(ReconError_norm_strength(i_ReconError).^2 * ...
%5			  (ReconError_norm_vals(:).^2 + (ReconError_norm_vals(:)==0).^2)));
	  endif
	endif %% ~isempty(ReconError_norm_struct)
	
      else %% isempty(ReconError_struct{i_frame})
	num_ReconError_frames = i_frame - 1;
	ReconError_times = ReconError_times(1:(num_ReconError_frames));
	ReconError_RMS = ReconError_RMS(1:(num_ReconError_frames));
	break;
      endif %% ~isempty(ReconError_struct{i_frame})
      
    endfor %% i_frame
    
    num_ReconError_frames = size(ReconError_times,1);
    if num_ReconError_frames <= 0
      disp(["num_ReconError_frames = ", num2str(num_ReconError_frames)]);
      continue;
    endif

    ReconError_RMS = ReconError_RMS(1:num_ReconError_frames);
    ReconError_times = ReconError_times(1:num_ReconError_frames);

    ReconError_RMS_filename = ...
	["RMS_", ReconError_list{i_ReconError,1}, ReconError_list{i_ReconError,2}, ...
	 "_", num2str(ReconError_times(num_ReconError_frames), "%08d")];
    ReconError_RMS_pathname = ... 
	[ReconError_dir, filesep, ReconError_RMS_filename];
    save("-mat", ...
	 [ReconError_RMS_pathname, ".mat"], ...
	 "ReconError_times", "ReconError_RMS");    

    if i_ReconError > num_ReconError_list
      keyboard;
    endif
    original_name = "";
    if plot_ReconError_flag
      if ReconError_RMS_fig_ndx(i_ReconError) > 0 && ...
	    ReconError_RMS_fig_ndx(i_ReconError) <= length(nonSparse_RMS_fig(:))
	original_name = get(nonSparse_RMS_fig(ReconError_RMS_fig_ndx(i_ReconError)),"name");
	ReconError_RMS_filename = [original_name, "_", ReconError_RMS_filename];
	ReconError_RMS_fig(i_ReconError) = nonSparse_RMS_fig(ReconError_RMS_fig_ndx(i_ReconError));
      else 
	ReconError_RMS_fig(i_ReconError) = figure;
	axis tight;
      endif
      figure(ReconError_RMS_fig(i_ReconError));
      gca;
      hold on
      ReconError_RMS_hndl = plot(ReconError_times, ReconError_RMS); 
      set(ReconError_RMS_hndl, "linewidth", 1.5);
      set(ReconError_RMS_hndl, "color", ReconError_colormap(i_ReconError+1,:));
      set(ReconError_RMS_fig(i_ReconError), "name", ReconError_RMS_filename);
      ReconError_RMS_pathname2 = ... 
	  [ReconError_dir, filesep, ReconError_RMS_filename];
      local_pwd = pwd;
      chdir(ReconError_dir);
%%      saveas(ReconError_RMS_fig(i_ReconError), ...
%%	     ReconError_RMS_pathname2, "png");
      saveas(ReconError_RMS_fig(i_ReconError), ...
	     ReconError_RMS_filename, "png");
      chdir(local_pwd);
    endif
    ReconError_median_RMS = median(ReconError_RMS(:));
    disp([ReconError_RMS_filename, ...
	  " median RMS = ", num2str(ReconError_median_RMS)]);

    ReconError_times_array{init_ReconError_list+i_ReconError} = ReconError_times;
    ReconError_RMS_array{init_ReconError_list+i_ReconError} = ReconError_RMS;
    %%ReconError_norm_RMS_array{init_ReconError_list+i_ReconError} = ReconError_norm_strength(i_ReconError)*ones(size(ReconError_RMS));
    ReconError_norm_RMS_array{init_ReconError_list+i_ReconError} = ones(size(ReconError_RMS));
  endfor  %% i_ReconError
  %%keyboard;

endfunction