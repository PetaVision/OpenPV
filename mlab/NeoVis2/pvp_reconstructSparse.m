
function [pvp_image] = ...
      pvp_reconstructSparse(frame_pathname, pvp_time, pvp_activity)
  
  global NFEATURES NCOLS NROWS N
  %%keyboard;

  global pvp_overlay_original
  if pvp_overlay_original
    pvp_image = imread(frame_pathname);
    pvp_image = rgb2gray(pvp_image);
    if any( size(pvp_image) ~= [NROWS, NCOLS] );
      pvp_image = imresize(pvp_image, [NROWS, NCOLS]);
    endif
    %%figure; imagesc(pvp_image); colormap(gray);
    pvp_image = repmat(pvp_image, [1, 1, 3]);
  else
    pvp_image = zeros(NROWS, NCOLS, 3);
    pvp_image = uint8(pvp_image);
  endif

  pvp_active_ndx = find(pvp_activity);
  pvp_num_active = length(pvp_active_ndx);
  pvp_active_ndx = pvp_active_ndx + 1;
  [pvp_active_features, pvp_active_cols, pvp_active_rows] = ...
      ind2sub([NFEATURES NCOLS NROWS], pvp_active_ndx);
  dilate_size = 1;
  for i_active = 1 : pvp_num_active
    pvp_image(pvp_active_rows(i_active), pvp_active_cols(i_active), 1) = uint8(255);
    pvp_image(pvp_active_rows(i_active), pvp_active_cols(i_active), 2) = uint8(0);
    pvp_image(pvp_active_rows(i_active), pvp_active_cols(i_active), 3) = uint8(0);
    %% add dilation
    for i_dilate_row = pvp_active_rows(i_active)-dilate_size: pvp_active_rows(i_active)+dilate_size
      if i_dilate_row < 1 || i_dilate_row > size(pvp_image, 1)
	continue;
      endif
      for i_dilate_col = pvp_active_cols(i_active)-dilate_size: pvp_active_cols(i_active)+dilate_size
	if i_dilate_col < 1 || i_dilate_col > size(pvp_image, 2)
	  continue;
	endif
	pvp_image(i_dilate_row, i_dilate_col, 1) = uint8(255);
	pvp_image(i_dilate_row, i_dilate_col, 2) = uint8(0);
	pvp_image(i_dilate_row, i_dilate_col, 3) = uint8(0);
      endfor
    endfor
  endfor

