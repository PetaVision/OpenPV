function [pad_image] = padGray(gray_chip, pad_size, gray_val)

  [num_rows_chip, num_cols_chip] = size(gray_chip(:,:,1));
  pad_image = repmat(gray_val, pad_size);
  center_row_chip = fix(num_rows_chip / 2);
  center_col_chip = fix(num_cols_chip / 2);
  row_start = fix(pad_size(1)/2)-floor(num_rows_chip/2);
  row_end = fix(pad_size(1)/2)+ceil(num_rows_chip/2)-1;
  col_start = fix(pad_size(2)/2)-floor(num_cols_chip/2);
  col_end = fix(pad_size(2)/2)+ceil(num_cols_chip/2)-1;
  chip_row_start = 1;
  if row_start < 1
    chip_row_start = -row_start + 2;
    row_start = 1;
  endif
  chip_row_end = num_rows_chip;
  if row_end > pad_size(1)
    chip_row_end = num_rows_chip - (row_end - pad_size(1));
    row_end = pad_size(1);
  endif
  chip_col_start = 1;
  if col_start < 1
    chip_col_start = -col_start + 2;
    col_start = 1;
  endif
  chip_col_end = num_cols_chip;
  if col_end > pad_size(2)
    chip_col_end = num_cols_chip - (col_end - pad_size(2));
    col_end = pad_size(2);
  endif
  pad_image(row_start:row_end, ...
	    col_start:col_end) = ...
      gray_chip(chip_row_start:chip_row_end, ...
		chip_col_start:chip_col_end);
  