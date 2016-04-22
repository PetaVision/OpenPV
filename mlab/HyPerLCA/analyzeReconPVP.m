function [Recon_hdr, ...
	  Recon_fig, ...
	  Recon_fig_name, ...
	  Recon_vals, ...
	  Recon_time, ...
	  Recon_mean, ...
	  Recon_std, ...
	  unwhitened_Recon_fig, ...
	  unwhitened_Recon_vals] = ...
      analyzeReconPVP(Recon_list, ...
		      num_Recon_frames_per_layer, ...
		      output_dir, ...
		      plot_Recon_flag, ...
		      Recon_sum_list, ...
		      DoG_weights, ...
		      Recon_unwhiten_list, ...
		      Recon_normalize_list, ...
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

  %% unwhiten using DoG weights, if present
  if ~exist("DoG_weights", "var") || isempty(DoG_weights)
    unwhiten_flag = false;
  endif
  unwhitened_Recon_fig = [];
  unwhitened_Recon_DoG = [];
  if unwhiten_flag
    unwhitened_Recon_fig = zeros(num_Recon_list, 1);
    unwhitened_Recon_DoG = cell(num_Recon_list, 1);
    mean_unwhitened_Recon = cell(num_Recon_list, 1);
    std_unwhitened_Recon = cell(num_Recon_list, 1);
    max_unwhitened_Recon = cell(num_Recon_list, 1);
    min_unwhitened_Recon = cell(num_Recon_list, 1);
  else
    Recon_unwhiten_list = zeros(num_Recon_list,1);
  endif


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

    %% initialize unwhitening block
    if Recon_unwhiten_list(i_Recon)
      if plot_Recon_flag
	unwhitened_Recon_fig(i_Recon) = figure;
      endif
      unwhitened_Recon_vals{i_Recon} = cell(num_Recon_frames(i_Recon),1);
      mean_unwhitened_Recon{i_Recon,1} = zeros(num_Recon_colors,num_Recon_frames(i_Recon));
      std_unwhitened_Recon{i_Recon, 1} = ones(num_Recon_colors, num_Recon_frames(i_Recon));
      max_unwhitened_Recon{i_Recon, 1} = ones(num_Recon_colors, num_Recon_frames(i_Recon));
      min_unwhitened_Recon{i_Recon, 1} = zeros(num_Recon_colors,num_Recon_frames(i_Recon));
    endif

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
	downsample_factor_row = size(Recon_vals{sum_ndx}{j_frame},1) / size(Recon_vals{i_Recon}{i_frame},1);
	downsample_factor_col = size(Recon_vals{sum_ndx}{j_frame},2) / size(Recon_vals{i_Recon}{i_frame},2);
	if downsample_factor_row > 1 || downsample_factor_col > 1
	  Recon_vals_downsampled = reshape(Recon_vals{sum_ndx}{j_frame}, [size(Recon_vals{i_Recon}{i_frame},1), downsample_factor_row, size(Recon_vals{i_Recon}{i_frame},2), downsample_factor_col, size(Recon_vals{i_Recon}{i_frame},3)]);
	  Recon_vals_downsampled = squeeze(mean(mean(Recon_vals_downsampled,4), 2));
	else
	  Recon_vals_downsampled = Recon_vals{sum_ndx}{j_frame};
	endif
	Recon_vals{i_Recon}{i_frame} = ...
	    ((Recon_vals{i_Recon}{i_frame} -min(Recon_vals{i_Recon}{i_frame}(:))) / ...
	     (((max(Recon_vals{i_Recon}{i_frame}(:)) - min(Recon_vals{i_Recon}{i_frame}(:)))) + ...
	      (max(Recon_vals{i_Recon}{i_frame}(:)) == min(Recon_vals{i_Recon}{i_frame}(:))))) + ...
	    ((Recon_vals_downsampled - min(Recon_vals_downsampled(:))) / ...
	     ((max(Recon_vals_downsampled(:)) - min(Recon_vals_downsampled(:))) + ...
	      (max(Recon_vals_downsampled(:)) == min(Recon_vals_downsampled(:)))));
	Recon_fig_name{i_Recon} = [Recon_fig_name{i_Recon}, "_", Recon_list{sum_ndx,2}];
      endfor %% i_sum
      mean_Recon_tmp = mean(Recon_vals{i_Recon}{i_frame}(:));
      std_Recon_tmp = std(Recon_vals{i_Recon}{i_frame}(:));
      Recon_mean(i_Recon) = Recon_mean(i_Recon) + mean_Recon_tmp;
      Recon_std(i_Recon) = Recon_std(i_Recon) + std_Recon_tmp;
      Recon_fig_name{i_Recon} = [Recon_fig_name{i_Recon}, "_", num2str(Recon_time{i_Recon}(i_frame), "%08d")];
      Recon_vals_tmp = ...
      permute(Recon_vals{i_Recon}{i_frame},[2,1,3]);
      if num_Recon_colors > 3
	Recon_colormap = prism(num_Recon_colors+1);
	Recon_vals_tmp2 = zeros([size(Recon_vals_tmp,1),size(Recon_vals_tmp,2),3]);
	for Recon_color_ndx = 1 : num_Recon_colors
	  Recon_color = Recon_colormap(Recon_color_ndx,:);
	  Recon_vals_tmp2(:,:,1) = squeeze(Recon_vals_tmp2(:,:,1)) + squeeze(Recon_vals_tmp(:,:,Recon_color_ndx) .* Recon_color(1));
	  Recon_vals_tmp2(:,:,2) = squeeze(Recon_vals_tmp2(:,:,2)) + squeeze(Recon_vals_tmp(:,:,Recon_color_ndx) .* Recon_color(2));
	  Recon_vals_tmp2(:,:,3) = squeeze(Recon_vals_tmp2(:,:,3)) + squeeze(Recon_vals_tmp(:,:,Recon_color_ndx) .* Recon_color(3));
	endfor
	Recon_vals_tmp = Recon_vals_tmp2;
      endif

      if num_Recon_colors > 3
	Recon_colormap = prism(num_Recon_colors+1);
	Recon_vals_tmp2 = zeros([size(Recon_vals_tmp,1),size(Recon_vals_tmp,2),3]);
	for Recon_color_ndx = 1 : num_Recon_colors
	  Recon_color = Recon_colormap(Recon_color_ndx,:);
	  Recon_vals_tmp2(:,:,1) = squeeze(Recon_vals_tmp2(:,:,1)) + squeeze(Recon_vals_tmp(:,:,Recon_color_ndx) .* Recon_color(1));
	  Recon_vals_tmp2(:,:,2) = squeeze(Recon_vals_tmp2(:,:,2)) + squeeze(Recon_vals_tmp(:,:,Recon_color_ndx) .* Recon_color(2));
	  Recon_vals_tmp2(:,:,3) = squeeze(Recon_vals_tmp2(:,:,3)) + squeeze(Recon_vals_tmp(:,:,Recon_color_ndx) .* Recon_color(3));
	endfor
	Recon_vals_tmp = Recon_vals_tmp2;
      endif

      Recon_vals_tmp = ...
	  (Recon_vals_tmp - min(Recon_vals_tmp(:))) / ...
	  ((max(Recon_vals_tmp(:))-min(Recon_vals_tmp(:))) + ...
	   ((max(Recon_vals_tmp(:))-min(Recon_vals_tmp(:)))==0));
      Recon_vals_tmp = uint8(255*squeeze(Recon_vals_tmp));
      if plot_Recon_flag
	set(Recon_fig(i_Recon, i_frame), "name", Recon_fig_name{i_Recon});
	imagesc(Recon_vals_tmp); 
	if num_Recon_colors == 1
	  colormap(gray); 
	endif
	box off; axis off; axis image;
	saveas(Recon_fig(i_Recon, i_frame), ...
	       [Recon_dir, filesep, Recon_fig_name{i_Recon}, ".png"], "png");
	if num_Recon_list > 20
	  close(Recon_fig(i_Recon, i_frame));
	endif
      else
	%%imwrite(Recon_vals_tmp, [Recon_dir, filesep, Recon_fig_name{i_Recon}, ".png"], "png");
      endif
      imwrite(Recon_vals_tmp, [Recon_dir, filesep, Recon_fig_name{i_Recon}, ".png"], "png");


      %%  unwhitening block
      if Recon_unwhiten_list(i_Recon)
	%%keyboard;
	for i_color = 1 : num_Recon_colors
	  tmp_Recon = ...
	      squeeze(Recon_vals{i_Recon}{i_frame}(:,:,i_color));
	  mean_unwhitened_Recon{i_Recon}(i_color, i_frame) = mean(tmp_Recon(:));
 	  std_unwhitened_Recon{i_Recon}(i_color, i_frame) = std(tmp_Recon(:));
	  max_unwhitened_Recon{i_Recon}(i_color, i_frame) = max(tmp_Recon(:));
	  min_unwhitened_Recon{i_Recon}(i_color, i_frame) = min(tmp_Recon(:));
	endfor
	unwhitened_Recon_vals{i_Recon}{i_frame} = ...
	    zeros(size(permute(Recon_vals{i_Recon}{i_frame},[2,1,3])));
	for i_color = 1 : num_Recon_colors
	  tmp_Recon = ...
	      deconvolvemirrorbc(squeeze(Recon_vals{i_Recon}{i_frame}(:,:,i_color))', DoG_weights);
	  j_frame = i_frame;
	  while (Recon_time{i_Recon}(i_frame) > Recon_time{Recon_normalize_list(i_Recon)}(j_frame))
	    j_frame = j_frame + 1;
	    if j_frame > num_Recon_frames(i_Recon)
	      j_frame = i_frame;
	      break;
	    endif
	  endwhile
	  if j_frame > num_Recon_frames(i_Recon)
	    j_frame = i_frame;
	  endif
	  if (Recon_time{i_Recon}(i_frame) ~= Recon_time{Recon_normalize_list(i_Recon)}(j_frame))
	    warning(["i_Recon = ", num2str(i_Recon), ", i_frame = ", num2str(i_frame), ...
		     ", Recon_time{i_Recon}(i_frame) = ", ...
		     num2str(Recon_time{i_Recon}(i_frame)), ...
		     " ~= ", ...
		     "Recon_normalize_list(i_Recon) = ", ...
		     num2str(Recon_normalize_list(i_Recon)), ", j_frame = ", num2str(j_frame), ...
		     ", Recon_time{Recon_normalize_list(i_Recon)}(j_frame) = ", ...
		     num2str(Recon_time{Recon_normalize_list(i_Recon)}(j_frame))]);
	  endif
	  %%j_frame = ceil(i_frame * tot_Recon_frames(Recon_normalize_list(i_Recon)) / tot_Recon_frames(i_Recon));
	  if i_Recon ~= Recon_normalize_list(i_Recon)
	    tmp_Recon = ...
		(tmp_Recon - mean_unwhitened_Recon{i_Recon}(i_color, j_frame)) * ...
		(std_unwhitened_Recon{Recon_normalize_list(i_Recon)}(i_color, j_frame) / ...
		 (std_unwhitened_Recon{i_Recon}(i_color, j_frame) + ...
		  (std_unwhitened_Recon{i_Recon}(i_color, j_frame)==0))) + ...
		mean_unwhitened_Recon{Recon_normalize_list(i_Recon)}(i_color, j_frame); 
	  endif
	  unwhitened_Recon_vals{i_Recon}{i_frame}(:,:,i_color) = tmp_Recon;
	endfor %% i_color
	if plot_Recon_flag
	  figure(unwhitened_Recon_fig(i_Recon));
	endif
	for i_sum = 1 : num_Recon_sum_list
	  sum_ndx = Recon_sum_list{i_Recon}(i_sum);
	  unwhitened_Recon_vals{i_Recon}{i_frame} = unwhitened_Recon_vals{i_Recon}{i_frame} + ...
	      unwhitened_Recon_vals{sum_ndx}{i_frame};
	endfor %% i_sum
	unwhitened_Recon_vals_tmp = ...
	    permute(unwhitened_Recon_vals{i_Recon}{i_frame},[2,1,3]);
	unwhitened_Recon_vals_tmp = ...
	    (unwhitened_Recon_vals_tmp - min(unwhitened_Recon_vals_tmp(:))) / ...
	    ((max(unwhitened_Recon_vals_tmp(:))-min(unwhitened_Recon_vals_tmp(:))) + ...
	     ((max(unwhitened_Recon_vals_tmp(:))-min(unwhitened_Recon_vals_tmp(:)))==0));
	unwhitened_Recon_vals_tmp = uint8(255*squeeze(unwhitened_Recon_vals_tmp));
	if plot_Recon_flag
	  set(unwhitened_Recon_fig(i_Recon), "name", ["unwhitened_", Recon_fig_name{i_Recon}]); 
	  imagesc(unwhitened_Recon_vals_tmp); 
	  if num_Recon_colors == 1
	    colormap(gray); 
	  endif
	  box off; axis off; axis image;
	  saveas(unwhitened_Recon_fig(i_Recon), ...
		 [Recon_dir, filesep, "unwhitened_", Recon_fig_name{i_Recon}, ".png"], "png");
	  drawnow
	else
	  imwrite(unwhitened_Recon_vals_tmp, ...
		  [Recon_dir, filesep, "unwhitened_", Recon_fig_name{i_Recon}, ".png"], "png");
	endif
      endif %% Recon_unwhiten_list


    endfor   %% i_frame
    Recon_mean(i_Recon) = Recon_mean(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    Recon_std(i_Recon) = Recon_std(i_Recon) / (num_Recon_frames(i_Recon) + (num_Recon_frames(i_Recon) == 0));
    disp([ Recon_fig_name{i_Recon}, 
	  "_Recon_mean = ", num2str(Recon_mean(i_Recon)), " +/- ", num2str(Recon_std(i_Recon))]);
    
  endfor %% i_Recon

endfunction %% analyzeReconPVP
