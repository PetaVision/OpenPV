function [Recon_hdr, ...
	  Recon_fig, ...
	  Recon_fig_name, ...
	  Recon_vals, ...
	  Recon_time, ...
	  Recon_mean, ...
	  Recon_std] = ...
      analyzeUnwhitenedReconPVP(Recon_list, ...
		      num_Recon_frames_per_layer, ...
		      output_dir, ...
		      plot_Recon_flag, ...
		      Recon_sum_list, ...
		      Recon_LIFO_flag)

  if exist(output_dir,"dir") ~= 7
    error(["analyzeReconPVP::output_dir does not exist: ", output_dir]);
  endif
  if ~exist("num_Recon_frames_per_layer") || isempty(num_Recon_frames_per_layer)
    num_Recon_frames_per_layer = 1;
  endif
  if ~exist("plot_Recon_flag", "var") || isempty(plot_Recon_flag)
    plot_Recon_flag = true;
  endif
  if ~exist("Recon_LIFO_flag", "var") || isempty(Recon_LIFO_flag)
    Recon_LIFO_flag = true;
  endif

  num_Recon_list = size(Recon_list,1);
  num_Recon_frames = repmat(num_Recon_frames_per_layer, 1, num_Recon_list);
  
  %%keyboard;
  Recon_dir = [output_dir, filesep, "Recon"]
  [status, msg, msgid] = mkdir(Recon_dir);
  if status ~= 1
    warning(["mkdir(", Recon_dir, ")", " msg = ", msg]);
  endif 

  Recon_hdr = cell(num_Recon_list, 1);
  Recon_fid = zeros(num_Recon_list, 1);
  Recon_fig = zeros(num_Recon_list, 1);
  Recon_fig_name = cell(num_Recon_list, 1);
  Recon_vals = cell(num_Recon_list, 1);
  Recon_time = cell(num_Recon_list, 1);

  tot_Recon_frames = zeros(num_Recon_list,1);
  Recon_mean = zeros(num_Recon_list, 1);
  Recon_std = zeros(num_Recon_list, 1);
  for i_Recon = 1 : num_Recon_list
    Recon_file = [output_dir, filesep, Recon_list{i_Recon,1}, Recon_list{i_Recon,2}, ".pvp"]
    if ~exist(Recon_file, "file")
      warning(["file does not exist: ", Recon_file]);
      continue;
    endif
    Recon_fid(i_Recon) = fopen(Recon_file);
    Recon_hdr{i_Recon} = readpvpheader(Recon_fid(i_Recon));
    fclose(Recon_fid(i_Recon));
  endfor %% i_Recon
  for i_Recon = 1 : num_Recon_list
    Recon_file = [output_dir, filesep, Recon_list{i_Recon,1}, Recon_list{i_Recon,2}, ".pvp"]
    if ~exist(Recon_file, "file")
      warning(["file does not exist: ", Recon_file]);
      continue;
    endif
    tot_Recon_frames(i_Recon) = Recon_hdr{i_Recon}.nbands;
    progress_step = ceil( tot_Recon_frames(i_Recon)/ 10);
    if Recon_LIFO_flag
      [Recon_struct, Recon_hdr_tmp] = ...
	  readpvpfile(Recon_file, ...
		      progress_step, ...
		      tot_Recon_frames(i_Recon), ... 
		      tot_Recon_frames(i_Recon)-num_Recon_frames(i_Recon)+1); 
    else
      [Recon_struct, Recon_hdr_tmp] = ...
	  readpvpfile(Recon_file, ...
		      progress_step, ...
		      num_Recon_frames(i_Recon), ... 
		      1); 
    endif
    %%if plot_Recon_flag
    %%  Recon_fig(i_Recon, i_frame) = figure;
    %%endif
    num_Recon_colors = Recon_hdr{i_Recon}.nf;
    Recon_vals{i_Recon} = cell(num_Recon_frames(i_Recon),1);
    Recon_time{i_Recon} = zeros(num_Recon_frames(i_Recon),1);

    for i_frame = 1 : num_Recon_frames(i_Recon)
      Recon_time{i_Recon}(i_frame) = Recon_struct{i_frame}.time;
      Recon_vals{i_Recon}{i_frame} = Recon_struct{i_frame}.values;
      if plot_Recon_flag
	Recon_fig(i_Recon, i_frame) = figure;
      endif
      Recon_fig_name{i_Recon} = Recon_list{i_Recon,2};
      num_Recon_sum_list = length(Recon_sum_list{i_Recon});
      for i_sum = 1 : num_Recon_sum_list
	sum_ndx = Recon_sum_list{i_Recon}(i_sum);
	%% if simulation still running, current layer might reflect later times
	%% however, trigger layers may have a slightly later timestamp than non-trigger layers
	%% assume simTime difference is no larger than 2 (hack)
	j_frame = i_frame;
	while abs(Recon_time{i_Recon}(i_frame) - Recon_time{sum_ndx}(j_frame)) > 2
	  j_frame = j_frame + 1;
	  if j_frame > num_Recon_frames(i_Recon)
	    break;
	  endif
	endwhile
	if j_frame > num_Recon_frames(i_Recon)
	  continue;
	endif
	Recon_vals{i_Recon}{i_frame} = Recon_vals{i_Recon}{i_frame} + ...
	    Recon_vals{sum_ndx}{j_frame};
	Recon_fig_name{i_Recon} = [Recon_fig_name{i_Recon}, "_", Recon_list{sum_ndx,2}];
      endfor %% i_sum
      mean_Recon_tmp = mean(Recon_vals{i_Recon}{i_frame}(:));
      std_Recon_tmp = std(Recon_vals{i_Recon}{i_frame}(:));
      Recon_mean(i_Recon) = Recon_mean(i_Recon) + mean_Recon_tmp;
      Recon_std(i_Recon) = Recon_std(i_Recon) + std_Recon_tmp;
      Recon_fig_name{i_Recon} = [Recon_fig_name{i_Recon}, "_", num2str(Recon_time{i_Recon}(i_frame), "%08d")];
      Recon_vals_tmp = ...
	  permute(Recon_vals{i_Recon}{i_frame},[2,1,3]);
      Recon_vals_tmp = ...
	  (Recon_vals_tmp - min(Recon_vals_tmp(:))) / ...
	  ((max(Recon_vals_tmp(:))-min(Recon_vals_tmp(:))) + ...
	   ((max(Recon_vals_tmp(:))-min(Recon_vals_tmp(:)))==0));
      Recon_vals_tmp = uint8(255*squeeze(Recon_vals_tmp));
      if plot_Recon_flag
	set(Recon_fig(i_Recon, i_frame), "name", Recon_fig_name{i_Recon});
	imagesc(Recon_vals_tmp); %%imagesc(flipdim(Recon_vals_tmp,1)); 
	if num_Recon_colors == 1
	  colormap(gray); 
	endif
	box off; axis off; axis image;
	saveas(Recon_fig(i_Recon, i_frame), ...
	       [Recon_dir, filesep, Recon_fig_name{i_Recon}, ".png"], "png");
      endif
      imwrite(Recon_vals_tmp, [Recon_dir, filesep, Recon_fig_name{i_Recon}, "_image", ".png"], "png");



    endfor   %% i_frame
    Recon_mean(i_Recon) = Recon_mean(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    Recon_std(i_Recon) = Recon_std(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    disp([ Recon_fig_name{i_Recon}, 
	  "_Recon_mean = ", num2str(Recon_mean(i_Recon)), " +/- ", num2str(Recon_std(i_Recon))]);
    
  endfor %% i_Recon

endfunction %% analyzeReconPVP
