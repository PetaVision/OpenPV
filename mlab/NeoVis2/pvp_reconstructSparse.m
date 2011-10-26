
function [pvp_image, pvp_num_active] = ...
      pvp_reconstructSparse(frame_ID, pvp_time, pvp_activity)
  
  global NFEATURES NCOLS NROWS N
  %%keyboard;

  pvp_image = imread(frame_ID);
  pvp_image = rgb2gray(pvp_image);
  if any( size(pvp_image) ~= [NROWS, NCOLS] );
    pvp_image = imresize(pvp_image, [NROWS, NCOLS]);
  endif
  %%figure; imagesc(pvp_image); colormap(gray);
  pvp_image = repmat(pvp_image, [1, 1, 3]);
  pvp_active_ndx = find(pvp_activity);
  pvp_active_ndx = pvp_active_ndx + 1;
  pvp_num_active = length(pvp_active_ndx);
  [pvp_active_features, pvp_active_cols, pvp_active_rows] = ...
      ind2sub([NFEATURES NCOLS NROWS], pvp_active_ndx);
  for i_active = 1 : pvp_num_active
    pvp_image(pvp_active_rows(i_active), pvp_active_cols(i_active), 1) = uint8(255);
  endfor

