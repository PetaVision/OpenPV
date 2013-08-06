%%function drawpvp_weights

function weights = drawpvp_weights(pvpfilename, recon_dir)
  i_frame = 1;
  i_arbor = 1;
  i_patch = 1;

  ndx = strfind(pvpfilename,"/");
  fname = pvpfilename(ndx(end):end-4);

  [wstruct, hdr] = readpvpfile(pvpfilename,1);
  weights = wstruct{1}.values{1};
  h = figure;
  set(h,"name",fname);
  size_weights = size(weights);
  num_dims = length(size_weights);
  if num_dims > 2
    num_pre_colors = size_weights(3);
  else
    num_pre_colors = 1;
  endif
  if num_dims > 3
    num_post_colors = size_weights(4);
  else
    num_post_colors = 1;
  endif
  for i_pre_color = 1 : num_pre_colors
    for i_post_color = 1 : num_post_colors
      subplot(num_pre_colors, num_post_colors, (i_pre_color - 1) * num_post_colors + i_post_color);
      patch_tmp = weights(:,:,i_pre_color, i_post_color);
      patch_tmp2 = patch_tmp;
      min_patch = min(patch_tmp2(:));
      max_patch = max(patch_tmp2(:));
      patch_tmp2 = (patch_tmp2 - min_patch) * 255 / (max_patch - min_patch + ((max_patch-min_patch)==0));
      patch_tmp2 = uint8(patch_tmp2);
      imagesc(patch_tmp2); colormap(gray);
      box off
      axis off
    endfor
  endfor
  saveas(h, [recon_dir, filesep, fname, ".png"]);
  drawnow;
endfunction
