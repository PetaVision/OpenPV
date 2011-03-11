function [MNIST_labels, MNIST_ndx] = ...
      getMNISTlabels(train_flag, MNIST_labels_filename)

  if nargin < 1 || ~exist("train_flag") || isempty(train_flag)
    train_flag = 0;
  endif
  if nargin < 2 || ~exist("MNIST_labels_filename") || isempty(MNIST_labels_filename)
    MNIST_path = "/Users/gkenyon/MATLAB/MNIST/";
    if train_flag == 1
      MNIST_labels_filename = [MNIST_path, 'train-labels.idx1-ubyte'];
    else
      MNIST_labels_filename = [MNIST_path, 't10K-labels.idx1-ubyte'];
    endif
  endif

  if train_flag == 1
    num_items = 60000;
  else
    num_items = 10000;
  endif

  MNIST_mode = "r+b";
  MNIST_arch = "ieee-be";

  %% get labels
  [fid_MNIST_labels, MNIST_labels_msg] = ...
      fopen(MNIST_labels_filename, MNIST_mode, MNIST_arch);
  if fid_MNIST_labels == -1
    warning(["fid_MNIST_labels == -1 for fopen of file ", ...
	     MNIST_labels_filename]);
    error(MNIST_labels_msg);
  endif

  %% check header
  MNIST_size = 1;
  MNIST_precision = "int32";
  MNIST_skip = 0;
  [MNIST_magic, MNIST_count] = ...
      fread(fid_MNIST_labels, MNIST_size, MNIST_precision, MNIST_skip, MNIST_arch);
  if MNIST_magic != 2049
    error(["MNIST_magic != 2049 in file ", MNIST_labels_filename]);
  endif
  [MNIST_num_labels, MNIST_count] = ...
      fread(fid_MNIST_labels, MNIST_size, MNIST_precision, MNIST_skip, MNIST_arch);
  if MNIST_num_labels != num_items
    disp(num2str(MNIST_num_labels(:)));
    error(["MNIST_num_labels = ", ...
	   " != ", num2str(num_items), ...
	   " in file ", MNIST_labels_filename]);
  endif
  
  %% read labels
  MNIST_size = MNIST_num_labels;
  MNIST_precision = "uchar";
  [MNIST_labels, MNIST_count] = ...
      fread(fid_MNIST_labels, MNIST_size, MNIST_precision, MNIST_skip, MNIST_arch);
  if MNIST_count != MNIST_num_labels
    error(["MNIST_count != MNIST_num_labels in file ", MNIST_labels_filename]);
  endif
  fclose(fid_MNIST_labels);

  MNIST_ndx = cell(2, 10);
  for digit_id = 0 : 9
    digit_ndx = digit_id;
    if digit_ndx == 0
      digit_ndx = 10;
    endif
    MNIST_ndx{2, digit_ndx} = ...
	find(MNIST_labels == digit_id);
    MNIST_ndx{1,digit_ndx} = length(MNIST_ndx{2,digit_ndx});
  endfor

  %% test labels
  for digit_id = 0 : 9
    digit_ndx = digit_id;
    if digit_ndx == 0
      digit_ndx = 10;
    endif
    badlabel_flag = ...
	any( MNIST_labels( MNIST_ndx{2,digit_ndx}(:) ) ~= digit_id );
    if badlabel_flag
      badlabel_ndx = find(MNIST_labels( MNIST_ndx{2,digit_ndx}(:) ) ~= digit_id);
      error(["badlabel_flag = ", num2str(badlabel_flag), ...
	     ", digit_id = ", num2str(digit_id), ...
	     ", badlabel_ndx(1) = ", num2str(badlabel_ndx(1))]);
    endif
  endfor

  %% test ndx
  test_MNIST_ndx = 0;
  if test_MNIST_ndx == 0
    return;
  endif
  MNIST_label_size = 1;
  MNIST_precision = "uchar";
  MNIST_offset0 = 8-1;
  [fid_MNIST_labels, MNIST_labels_msg] = ...
      fopen(MNIST_labels_filename, MNIST_mode, MNIST_arch);
  [tmp_MNIST_label, MNIST_count] = ...
      fread(fid_MNIST_labels, ...
	    MNIST_label_size*16, MNIST_precision, MNIST_skip, MNIST_arch);
  disp(num2str(tmp_MNIST_label(:)));
  for digit_id = 0 : 9
    digit_ndx = digit_id;
    if digit_ndx == 0
      digit_ndx = 10;
    endif
    MNIST_num_images = MNIST_ndx{1,digit_ndx};
    for image_count = 1 : MNIST_num_images
      tmp_MNIST_ndx = MNIST_ndx{2, digit_ndx}(image_count);
      MNIST_offset = tmp_MNIST_ndx * MNIST_label_size;
      MNIST_offset = MNIST_offset + MNIST_offset0;
      fseek(fid_MNIST_labels, MNIST_offset, "bof");
      [tmp_MNIST_label, MNIST_count] = ...
	  fread(fid_MNIST_labels, ...
		MNIST_label_size, MNIST_precision, MNIST_skip, MNIST_arch);
      goodlabel_flag = ...
	  (tmp_MNIST_label == digit_id);
      if (goodlabel_flag == 0)
	badlabel_ndx = tmp_MNIST_ndx;
	error(["badlabel_flag = ", num2str(badlabel_flag), ...
	       ", digit_id = ", num2str(digit_id), ...
	       ", badlabel_ndx = ", num2str(tmp_MNIST_ndx), ...
	       ", MNIST_label = ", num2str(tmp_MNIST_label)]);
      endif
    endfor
  endfor
  fclose(fid_MNIST_labels);

  
  

  
