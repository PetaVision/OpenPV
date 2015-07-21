function [amoeba_image_x, amoeba_image_y, amoeba_struct] = ...
      MNISTSegments(amoeba_struct, train_flag, target_id, target_type)

  digit_id = amoeba_struct.target_id;
  resize_factor = amoeba_struct.min_resize + rand() * ...
      (amoeba_struct.max_resize - amoeba_struct.min_resize);
  [MNIST_images] = ...
      getMNIST(digit_id, ...
	       train_flag, ...
	       resize_factor);
  num_images = size(MNIST_images,2);
  %%ydisp(["num_images = " , num2str(num_images)]);
  amoeba_image_x = cell(num_images, 1);
  amoeba_image_y = cell(num_images, 1);
  amoeba_image_val = cell(num_images, 1);
  for i_image = 1 : num_images
    [tmp_x, ...
     tmp_y, ...
     tmp_val] = ...
	find(MNIST_images{2,i_image});
    [MNIST_size_x, MNIST_size_y] = ...
	size( MNIST_images{2,i_image} );
    MNIST_center_x = (1 + MNIST_size_x ) / 2;
    MNIST_center_y = (1 + MNIST_size_y ) / 2;
    max_val = max(tmp_val);
    min_val = min(tmp_val);
    mid_val = min_val + (max_val - min_val) / 2;
    low_mean = mean(tmp_val(tmp_val<mid_val));
    low_std = std(tmp_val(tmp_val<mid_val));
    high_mean = mean(tmp_val(tmp_val>=mid_val));
    high_std = std(tmp_val(tmp_val>=mid_val));
    mid_val = high_mean - 0 * high_std;
    ON_ndx = find(tmp_val > mid_val);
    %%OFF_ndx = find(tmp_val <= mid_val);
    tmp_x = tmp_x(ON_ndx);
    tmp_y = tmp_y(ON_ndx);
    tmp_x = tmp_x - MNIST_center_x;
    tmp_y = tmp_y - MNIST_center_y;
    amoeba_struct.outer_diameter = 20 * resize_factor;
    amoeba_struct.inner_diameter = 0;
    [tmp_x, tmp_y, amoeba_struct] = ...
	offsetAmoebaSegments(amoeba_struct, ...
			     {tmp_x}, ...
			     {tmp_y});
    amoeba_image_x{i_image, 1} = tmp_x{1}(:);
    amoeba_image_y{i_image, 1} = tmp_y{1}(:);
  endfor %% i_image

  