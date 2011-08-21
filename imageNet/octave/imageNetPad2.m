
function [pad_image] = imageNetPad2(original_image, ...
				   original_info, ...
				   image_resize)
  

  pad_image = [];
  original_height = original_info.Height;
  original_width = original_info.Width;
  original_size = [original_height, original_width];
  max_dim = 0;
  min_dim = 0;
  rescale_factor = 1.0;
  original_ratio = original_height / original_width;
  pad_ratio = image_resize(1) / image_resize(2);
  rescale_height = image_resize(1) / original_height;
  rescale_width = image_resize(2) / original_width;
  rescale_factor = min(rescale_height, rescale_width);
  if rescale_height == 1 && rescale_width == 1 %% no need to pad
    pad_image = original_image;
    return;
  endif
  try
    standard_image = imresize(original_image, rescale_factor);
    standard_size = size(standard_image);
    if standard_size(1)  < image_resize(1)
      max_dim = 2;
      min_dim = 1;
    elseif standard_size(2) < image_resize(2)
      max_dim = 1;
      min_dim = 2;
    elseif all(standard_size(1:2) == image_resize(:)')
      pad_image = standard_image;
      return;
    elseif any(standard_size(1:2) > image_resize(:)')
      error(["any(standard_size(1:2) = ", ...
	     num2str(standard_size(1:2)), ...
	     " > image_resize(:) = ", ...
	     num2str(image_resize(:)), ...
	     " is true: "])
    endif
  catch
    disp(["failed imageNetPad2::imresize"]);
    return;
    %%keyboard;  %% 
  end
  pad_size = image_resize(min_dim) - standard_size(min_dim);
  pad_size1 = floor(pad_size/2);
  pad_size2 = ceil(pad_size/2);
  if pad_size2 >= size(standard_image,min_dim)
    pad_image = [];
    return;
  endif
  pad_image = ...
      zeros(image_resize(1),...
	    image_resize(2),...
	    size(standard_image,3));
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