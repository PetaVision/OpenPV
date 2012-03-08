
train_flag = 1;
resize_factor = 1;
nominal_ndx = 1:10;
MNIST_path = "~/Pictures/MNIST/";
for digit_id = 6
  [MNIST_images] = ...
      getMNIST(digit_id, ...
	       train_flag, ...
	       resize_factor, ...
	       nominal_ndx, ...
	       MNIST_path);
  out_dir = [MNIST_path, num2str(digit_id), filesep];
  mkdir(out_dir);
  num_images = size(MNIST_images, 2);
  for i_image = 1 : num_images
    MNIST_filename = [out_dir, "MNIST_", num2str(i_image, "%4.4i"), ".png"];
    image_tmp = uint8(MNIST_images{2,i_image});
    %%disp(["size(MNIST_image) = ", num2str(size(image_tmp))]);
    imwrite(image_tmp, MNIST_filename, "png");
  endfor
endfor
