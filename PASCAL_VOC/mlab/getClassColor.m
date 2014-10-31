function [class_color] = getClassColor(class_id)
  class_id = floor(class_id);
  red_val = floor(class_id / (2^16));
  class_id2 = class_id - red_val*(2^16);
  green_val = floor(class_id2/  (2^8));
  class_id3 = class_id2 - green_val*uint32(2^8);
  blue_val = floor(class_id3) ;
  class_color = [uint8(red_val), uint8(green_val), uint8(blue_val)];
endfunction
