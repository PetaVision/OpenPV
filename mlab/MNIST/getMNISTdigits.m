function [MNIST_images] = ...
      getMNISTdigits(digit_id, ...
		     MNIST_labels, ...
		     MNIST_ndx, ...
		     resize_factor, ...
		     nominal_ndx, ...
		     train_flag, ...
		     MNIST_digits_filename, ...
		     MNIST_labels_filename)

  MNIST_path = "/Users/gkenyon/MATLAB/MNIST/";
  
  if nargin < 1 || ~exist("digit_id") || isempty(digit_id)
    digit_id = 0;
  endif
  if nargin < 4 || ~exist("resize_factor") || isempty(resize_factor)
    resize_factor = 1;
  endif
  if nargin < 5 || ~exist("nominal_ndx") || isempty(nominal_ndx)
    nominal_ndx = -1;  %% denotes that 1 random digit_image will be returned
  endif
  if nargin < 6 || ~exist("train_flag") || isempty(train_flag)
    train_flag = 0;
  endif
  if nargin < 7 || ~exist("MNIST_digits_filename") || isempty(MNIST_digits_filename)
    if train_flag == 1
      MNIST_digits_filename = [MNIST_path, 'train-images.idx3-ubyte'];
    else
      MNIST_digits_filename = [MNIST_path, 't10K-images.idx3-ubyte'];
    endif
  endif
  if nargin < 8 || ~exist("MNIST_labels_filename") || isempty(MNIST_labels_filename)
    if train_flag == 1
      MNIST_labels_filename = [MNIST_path, 'train-labels.idx1-ubyte'];
    else
      MNIST_labels_filename = [MNIST_path, 't10K-labels.idx1-ubyte'];
    endif
  endif
  if nargin < 3 || ~exist("MNIST_ndx") || isempty(MNIST_ndx) || ...
	~exist("MNIST_labels") || isempty(MNIST_labels)
    [MNIST_labels, MNIST_ndx] = ...
      getMNISTlabels(train_flag, MNIST_labels_filename);
  endif

  if train_flag == 1
    num_items = 60000;
  else
    num_items = 10000;
  endif

  MNIST_mode = "r+b";
  MNIST_arch = "ieee-be";

  %% get digits
  [fid_MNIST_digits, MNIST_digits_msg] = ...
      fopen(MNIST_digits_filename, MNIST_mode, MNIST_arch);
  if fid_MNIST_digits == -1
    warning(["fid_MNIST_digits == -1 for fopen of file ", ...
	     MNIST_digits_filename]);
    error(MNIST_digits_msg);
  endif

  %% check/read header
  MNIST_size = 1;
  MNIST_precision = "int32";
  MNIST_skip = 0;
  [MNIST_magic, MNIST_count] = ...
      fread(fid_MNIST_digits, MNIST_size, MNIST_precision, MNIST_skip, MNIST_arch);
  if MNIST_magic != 2051
    error(["MNIST_magic != 2051 in file ", MNIST_digits_filename]);
  endif
  [MNIST_num_digits, MNIST_count] = ...
      fread(fid_MNIST_digits, MNIST_size, MNIST_precision, MNIST_skip, MNIST_arch);
  if MNIST_num_digits != num_items
    disp(num2str(MNIST_num_digits(:)));
    error(["MNIST_num_digits = ", ...
	   " != ", num2str(num_items), ...
	   " in file ", MNIST_digits_filename]);
  endif
  [MNIST_num_rows, MNIST_count] = ...
      fread(fid_MNIST_digits, MNIST_size, MNIST_precision, MNIST_skip, MNIST_arch);
  [MNIST_num_cols, MNIST_count] = ...
      fread(fid_MNIST_digits, MNIST_size, MNIST_precision, MNIST_skip, MNIST_arch);

  %% determine which images to extract
  digit_ndx = digit_id;
  if digit_ndx == 0
    digit_ndx = 10;
  endif
  MNIST_num_images = length(nominal_ndx);
  MNIST_images = cell(2, MNIST_num_images);
  max_nominal_ndx = MNIST_ndx{1,digit_ndx};
  for image_count = 1 : MNIST_num_images
    i_ndx = nominal_ndx(image_count);
    if i_ndx <= 0
      nominal_ndx(image_count) = ceil(rand * max_nominal_ndx);
    endif
  endfor
  nominal_ndx = mod( nominal_ndx, max_nominal_ndx ) + 1;
  
  %% read images
  MNIST_image_size = MNIST_num_rows * MNIST_num_cols;
  MNIST_precision = "uchar";
  MNIST_offset0 = 16;
  for image_count = 1 : MNIST_num_images
    tmp_MNIST_ndx = MNIST_ndx{2, digit_ndx}(nominal_ndx(image_count));
    MNIST_images{1, image_count} = tmp_MNIST_ndx;
    MNIST_offset = (tmp_MNIST_ndx-1) * MNIST_image_size;
    MNIST_offset = MNIST_offset + MNIST_offset0;
    fseek(fid_MNIST_digits, MNIST_offset, "bof");
    [MNIST_images{2,image_count}, MNIST_count] = ...
	fread(fid_MNIST_digits, ...
	      MNIST_image_size, MNIST_precision, MNIST_skip, MNIST_arch);
  endfor  
  fclose(fid_MNIST_digits);

  %% resize images
  for image_count = 1 : MNIST_num_images
    digit_image = ...
	reshape(MNIST_images{2,image_count}, [MNIST_num_rows, MNIST_num_cols])';
    digit_image = imresize (digit_image, resize_factor);
    MNIST_images{2,image_count} = digit_image;
  endfor

  MNIST_plot_flag = 0;
  if MNIST_plot_flag 
    for image_count = 1 : MNIST_num_images
      figure;
      image(MNIST_images{2,image_count});
      colormap('gray');
      axis "off"
    endfor
  endif
