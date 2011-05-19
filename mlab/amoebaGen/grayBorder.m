function [output_image] = grayBorder(input_image, margin_width, gray_val)
  image_size = size(input_image);
  if nargin < 2 || ~exist("margin_width") || isempty(margin_width)
    margin_width = 1;
  endif
  if nargin < 3 || ~exist("gray_val") || isempty(gray_val)
    gray_val = 0;
  endif
  output_image = input_image;
  output_image(1:1+margin_width, :, :) = gray_val;
  output_image(end-margin_width:end, :, :) = gray_val;
  output_image(:, 1:1+margin_width, :) = gray_val;
  output_image(:, end-margin_width:end, :) = gray_val;

  

  