function [blur_weights] = get_Blur_weights(blur_center_path)
  global plot_flag
  blur_weights = [];
  if ~exist("blur_center_path") || isempty(blur_center_path)
    return;
  endif
  i_frame = 1;
  i_arbor = 1;
  [blur_center_struct, blur_center_hdr] = readpvpfile(blur_center_path,1);
  blur_weights = (blur_center_struct{i_frame}.values{i_arbor});
  if plot_flag
    blur_fig = figure;
    set(blur_fig, "name", "blur Weights");
    size_blur_weights = size(blur_weights);
    num_dims_blur = length(size_blur_weights);
    if num_dims_blur > 2
      num_pre_colors = size_blur_weights(3);
    else
      num_pre_colors = 1;
    endif
    if num_dims_blur > 3
      num_post_colors = size_blur_weights(4);
    else
      num_post_colors = 1;
    endif
    for i_pre_color = 1 : num_pre_colors
      for i_post_color = 1 : num_post_colors
	subplot(num_pre_colors, num_post_colors, (i_pre_color - 1) * num_post_colors + i_post_color);
	patch_tmp = blur_weights(:,:,i_pre_color, i_post_color);
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
    %%saveas(blur_fig, [blur_center_path, filesep, "blur_weights.png"]);
    drawnow;
  endif
endfunction %% get_Blur_weights