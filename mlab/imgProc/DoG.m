function [image_DoG, zero_DoG] = ...
      DoG(image_original, ...
	  amp_center, ...
	  sigma_center, ...
	  amp_surround, ...
	  sigma_surround)

  DEBUG_FLAG = 0;
  ABSOLUTE_SCALE_FLAG = 1;
  
  image_size = size(image_original);
  if ABSOLUTE_SCALE_FLAG == 1
    max_original = 255;
    min_original = 0;
  else
    max_original = double( max(image_original(:)) );
    min_original = double( min(image_original(:)) );
  endif
  
  radius_center = ceil(3.0 * sigma_center);
  gauss_center = -radius_center : radius_center;
  gauss_center = exp( -0.5 * gauss_center.^2 / sigma_center.^2 );
  gauss_center2D = gauss_center' * gauss_center;
  norm_center2D = sum( gauss_center2D(:) );
  gauss_center2D = ...
      amp_center * ...
      gauss_center2D / ...
      ( norm_center2D + (norm_center2D == 0) );
  image_center = conv2(image_original, gauss_center2D, 'same');

  radius_surround = ceil(3.0 * sigma_surround);
  gauss_surround = -radius_surround : radius_surround;
  gauss_surround = exp( -0.5 * gauss_surround.^2 / sigma_surround.^2 );
  gauss_surround2D = gauss_surround' * gauss_surround;
  norm_surround2D = sum( gauss_surround2D(:) );
  gauss_surround2D = ...
      amp_surround * ...
      gauss_surround2D / ...
      ( norm_surround2D + (norm_surround2D == 0) );
  image_surround = conv2(image_original, gauss_surround2D, 'same');

  image_DoG = (image_center - image_surround);
  max_DoG = double( max(image_DoG(:)) );
  min_DoG = double( min(image_DoG(:)) );
  slope_DoG = ... 
      ( max_original - min_original ) / ...
      ( max_DoG - min_DoG  + ((max_DoG-min_DoG)==0) );
  image_DoG = min_original + ...
      ( image_DoG - min_DoG ) .* slope_DoG;
  zero_DoG = min_original + ...
      ( 0.0 - min_DoG ) .* slope_DoG;
  if DEBUG_FLAG
    disp(['slope_DoG = ', num2str(slope_DoG)]);
  endif