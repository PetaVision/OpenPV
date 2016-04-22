function [DoG_weights] = get_DoG_weights(DoG_center_path, DoG_surround_path)
  global plot_flag
  DoG_weights = [];
  if ~exist("DoG_center_path") || isempty(DoG_center_path)
    return;
  endif
  if ~exist("DoG_surround_path") || isempty(DoG_surround_path)
    return;
  endif
  i_frame = 1;
  i_arbor = 1;
  [DoG_center_struct, DoG_center_hdr] = readpvpfile(DoG_center_path,1);
  DoG_center_weights = (DoG_center_struct{i_frame}.values{i_arbor});
  size_DoG_center_weights = size(DoG_center_weights);
  [DoG_surround_struct, DoG_surround_hdr] = readpvpfile(DoG_surround_path,1);
  DoG_surround_weights = (DoG_surround_struct{i_frame}.values{i_arbor});
  size_DoG_weights = size(DoG_surround_weights);
  DoG_pad = (size_DoG_weights(1:2) - size_DoG_center_weights(1:2)) / 2;
  num_dims_DoG = length(size_DoG_weights);
  if num_dims_DoG > 2
    num_pre_colors = size_DoG_weights(3);
  else
    num_pre_colors = 1;
  endif
  if num_dims_DoG > 3
    num_post_colors = size_DoG_weights(4);
  else
    num_post_colors = 1;
  endif
  DoG_center_padded = zeros(size_DoG_weights(1:2));
  DoG_row_start = DoG_pad(1)+1;
  DoG_row_stop = size_DoG_weights(1)-DoG_pad(1);
  DoG_col_start = DoG_pad(2)+1;
  DoG_col_stop = size_DoG_weights(2)-DoG_pad(2);
  if plot_flag
    DoG_fig = figure;
    set(DoG_fig, "name", "DoG Weights");
  endif
  for i_pre_color = 1 : num_pre_colors
    for i_post_color = 1 : num_post_colors
      DoG_center_padded(DoG_row_start:DoG_row_stop, DoG_col_start:DoG_col_stop) = ...
	  DoG_center_weights(:,:, i_pre_color, i_post_color);
      DoG_weights = ...
	  DoG_center_padded - DoG_surround_weights(:, :, i_pre_color, i_post_color);
      if ~plot_flag 
	continue
      endif
      subplot(num_pre_colors, num_post_colors, (i_pre_color - 1) * num_post_colors + i_post_color);
      patch_tmp = DoG_weights;
      patch_tmp2 = patch_tmp; %% imresize(patch_tmp, 12);
      min_patch = min(patch_tmp2(:));
      max_patch = max(patch_tmp2(:));
      patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch-min_patch)==0));
      patch_tmp2 = uint8(patch_tmp2);
      imagesc(patch_tmp2); colormap(gray);
      box off
      axis off
    endfor
  endfor
  drawnow;
  %%saveas(DoG_fig, [DoG_surround_path, filesep, "DoG_weights.png"]);
endfunction