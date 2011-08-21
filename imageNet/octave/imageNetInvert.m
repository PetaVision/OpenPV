function [invert_image] = imageNetInvert(original_image)

  %% produces inverse of input image
  invert_image(:,1:3) = uint8( 255 - original_image(:,1:3) );

endfunction %% imageNetInvert