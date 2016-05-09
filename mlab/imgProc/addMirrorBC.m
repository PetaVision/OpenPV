function [pad_image] = addMirrorBC(original_image, pad)

  num_bands = size(original_image,3);
  original_size = [size(original_image,1), size(original_image,2)];

  pad_max = min( [pad, original_size(1), original_size(2)] );
  pad_image = original_image;
  pad_cum = 0;
  while pad_cum < pad

    pad_size = [size(pad_image,1), size(pad_image,2)];

    extended_size(1) = pad_size(1) + 2 * pad_max;
    extended_size(2) = pad_size(2) + 2 * pad_max;
  
    extended_image = zeros(extended_size(1), extended_size(2));

    %% interior
    extended_image(pad_max+1:end-pad_max, pad_max+1:end-pad_max) = ...
      pad_image;
    %% NW
    extended_image(1:pad_max, 1:pad_max) = ...
	rotdim(pad_image(1:pad_max, 1:pad_max), 2);
    %% N
    extended_image(1:pad_max, pad_max+1:end-pad_max) = ...
	flipud(pad_image(1:pad_max, :));
    %% NE
    extended_image(1:pad_max, end-pad_max+1:end) = ...
	rotdim(pad_image(1:pad_max, end-pad_max+1:end), 2);
    %% W
    extended_image(pad_max+1:end-pad_max, 1:pad_max) = ...
	fliplr(pad_image(:,1 :pad_max));
    %% E
    extended_image(pad_max+1:end-pad_max, end-pad_max+1:end) = ...
	fliplr(pad_image(:, end-pad_max+1:end));
    %% SW
    extended_image(end-pad_max+1:end, 1:pad_max) = ...
	rotdim(pad_image(end-pad_max+1:end, 1:pad_max), 2);
    %% S
    extended_image(end-pad_max+1:end, pad_max+1:end-pad_max) = ...
	flipud(pad_image(end-pad_max+1:end, :));
    %% SE
    extended_image(end-pad_max+1:end, end-pad_max+1:end) = ...
	rotdim(pad_image(end-pad_max+1:end, end-pad_max+1:end), 2);

    pad_image = extended_image;
    pad_max = min( [pad - pad_cum, extended_size(1), extended_size(2)] );
    pad_max = max(1, pad_max);
    pad_cum = pad_cum + pad_max;


endwhile%%pad_mas