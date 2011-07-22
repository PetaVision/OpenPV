
function [pad_image] = imageNetPad(original_image, ...
				   original_info, ...
				   image_resize)
  

  original_height = original_info.Height;
  original_width = original_info.Width;
  max_dim = 0;
  min_dim = 0;
  rescale_factor = 1.0;
  if original_height > original_width
    max_dim = 1;
    min_dim = 2;
    rescale_factor = image_resize(1) / original_height;
  elseif original_height < original_width
    max_dim = 2;
    min_dim = 1;
    rescale_factor = image_resize(2) / original_width;
  endif
  if rescale_factor > 1 && 0
    original_info
    keyboard
  endif
  standard_image = imresize(original_image, rescale_factor);
  if max_dim > 0 && size(standard_image,max_dim) ~= image_resize(max_dim)
    keyboard
  endif
  if max_dim == 0  %% no need to pad
    pad_image = standard_image;
    return;
  endif
  pad_size = size(standard_image,max_dim) - size(standard_image,min_dim);
  pad_size1 = floor(pad_size/2);
  pad_size2 = ceil(pad_size/2);
  pad_size1 = min(pad_size1,  size(standard_image,min_dim));
  pad_size2 = min(pad_size2,  size(standard_image,min_dim));
  pad_image = ...
      zeros([size(standard_image,max_dim),...
	     size(standard_image,max_dim),...
	     size(standard_image,3)]);
  pad_image = uint8(pad_image);	
  flip_image = flipdim(standard_image,min_dim);
  if max_dim == 1
    pad_image(:,pad_size1+1:end-pad_size2,:) = standard_image;
    pad_image(:,1:pad_size1,:) = ...
	flip_image(:,end-pad_size1+1:end,:);
    pad_image(:,end-pad_size2+1:end,:) = ...
	flip_image(:,1:pad_size2,:);
  else
    pad_image(pad_size1+1:end-pad_size2,:,:) = standard_image;
    pad_image(1:pad_size1,:,:) = ...
	flip_image(end-pad_size1+1:end,:,:);
    pad_image(end-pad_size2+1:end,:,:) = ...
	flip_image(1:pad_size2,:,:);
  endif

endfunction  %% imageNetPad