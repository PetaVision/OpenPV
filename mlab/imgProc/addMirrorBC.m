function [extended_image] = addMirrorBC(original_image, pad)

  num_bands = size(original_image,3);
  original_size = [size(original_image,1), size(original_image,2)];
  extended_size(1) = original_size(1) + 2 * pad;
  extended_size(2) = original_size(2) + 2 * pad;
  
  extended_image = zeros(extended_size(1), extended_size(2));

  %% interior
  extended_image(pad+1:end-pad, pad+1:end-pad) = ...
      original_image;
  %% NW
  extended_image(1:pad, 1:pad) = ...
      rotdim(original_image(1:pad, 1:pad), 2);
  %% N
  extended_image(1:pad, pad+1:end-pad) = ...
      flipud(original_image(1:pad, :));
  %% NE
  extended_image(1:pad, end-pad:end) = ...
      rotdim(original_image(1:pad, end-pad:end), 2);
  %% W
  extended_image(pad+1:end-pad, 1:pad) = ...
      fliplr(original_image(:,1 :pad));
  %% E
  extended_image(pad+1:end-pad, end-pad:end) = ...
      fliplr(original_image(:, end-pad:end));
  %% SW
  extended_image(end-pad:end, 1:pad) = ...
      rotdim(original_image(end-pad:end, 1:pad), 2);
  %% S
  extended_image(end-pad:end, pad+1:end-pad) = ...
      flipud(original_image(end-pad:end, :));
  %% SE
  extended_image(end-pad:end, end-pad:end) = ...
      rotdim(original_image(end-pad:end, end-pad:end), 2);
