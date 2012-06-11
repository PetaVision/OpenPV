function pvp_writeKernel(weights, weights_size, filename, resize_weights)

  global SPIKE_PATH 
  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer 
  global COMPRESSED_FLAG

  NFEATURES = size(weights, 2);
  NFP = weights_size(1);
  NXP = weights_size(2);
  NYP = weights_size(3);

  resize_weights_flag = 0;
  if nargin > 3 && ~isempty(resize_weights)
    resize_weights_flag = 1;
  endif

  max_weight = -100000;
  min_weight = 100000;
  for i_feature = 1 : NFEATURES
    weights_tmp = weights{i_feature}(:);
    max_weight = max( [weights_tmp(:)', max_weight] );
    min_weight = min( [weights_tmp(:)', min_weight] );
  endfor

  disp(['min_weight = ', num2str(min_weight)]);
  disp(['max_weight = ', num2str(max_weight)]);
  min_weight = 0.0;

  log2margin = 0;
  marginWidth = 0;  % default margin width is 0
  while ( marginWidth < fix(weights_size(2)/2) && ...
         marginWidth < fix(weights_size(3)/2) && ...
         marginWidth < NROWS / 2 && ...
         marginWidth < NCOLS / 2 )
    log2margin = log2margin + 1;
    marginWidth = 2 ^ log2margin; % - 1
  endwhile


  NUM_BIN_PARAMS = 18 + 2;
  NUM_WGT_EXTRA_PARAMS = 6;
  NUM_WGT_PARAMS = NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS;
  num_params = NUM_WGT_PARAMS;
  int32_size = 4;  %sizeof(int)
  PVP_WGT_FILE_TYPE  = 5;  %% KERNEL_FILE_TYPE
  NX_PROCS = 2;
  NY_PROCS = 2;
  NX = NCOLS / NX_PROCS;
  NY = NROWS / NY_PROCS;
  NF = NFEATURES;
  NUM_RECORDS = NFEATURES;
  RECORD_SIZE = prod(weights_size);
  if COMPRESSED_FLAG
    DATA_SIZE = 1;  % char
    DATA_TYPE = 1; % char
  else
    DATA_SIZE = 4;  % float
    DATA_TYPE = 3; % float
  endif
  NX_GLOBAL = NCOLS;
  NY_GLOBAL = NROWS;
  KX0 = 0;
  KY0 = 0;
  NPAD = marginWidth;
  NBANDS = NFEATURES;
  time_step = 0.0;
  WGT_NXP = weights_size(2);
  WGT_NYP = weights_size(3);
  WGT_NFP = weights_size(1);
  WGT_MIN = min_weight;
  WGT_MAX = max_weight;
  WGT_NUMPATCHES = NX_PROCS * NY_PROCS * NFEATURES;
  row_index_first = 1;
  row_index_last = weights_size(3);
  col_index_first = 1;
  col_index_last = weights_size(2);
  if resize_weights_flag
    row_middle = ( weights_size(1) / 2 ) + 0.5;
    NYP_half = ( resize_weights(3) / 2 );
    row_index_first = ceil( row_middle - NYP_half );
    row_index_last = floor( row_middle + NYP_half );
    
    col_middle = ( weights_size(2) / 2 ) + 0.5;
    NXP_half = ( resize_weights(2) / 2 );
    col_index_first = ceil( col_middle - NXP_half );
    col_index_last = floor( col_middle + NXP_half );
    
    WGT_NXP = length( col_index_first : col_index_last );
    WGT_NYP = length( row_index_first : row_index_last );
  endif


  kernel_filename = ...
      [SPIKE_PATH, filename, "_", num2str(NX_PROCS), "x", num2str(NY_PROCS), '.pvp'];

  disp(['WGT_NXP = ', num2str(WGT_NXP)]);
  disp(['WGT_NYP = ', num2str(WGT_NYP)]);
  disp(['WGT_NFP = ', num2str(WGT_NFP)]);
  disp(['NPAD = ', num2str(NPAD)]);


  fid = fopen(kernel_filename, 'w');

  fwrite(fid, num_params * int32_size, 'int32');
  fwrite(fid, num_params, 'int32');
  fwrite(fid, PVP_WGT_FILE_TYPE, 'int32'); % type
  fwrite(fid, NX, 'int32'); % nx
  fwrite(fid, NY, 'int32'); % ny
  fwrite(fid, NF, 'int32'); % nf
  fwrite(fid, NUM_RECORDS, 'int32');
  fwrite(fid, RECORD_SIZE, 'int32');
  fwrite(fid, DATA_SIZE, 'int32');
  fwrite(fid, DATA_TYPE, 'int32');
  fwrite(fid, NX_PROCS, 'int32'); % MPI config
  fwrite(fid, NY_PROCS, 'int32'); % MPI config
  fwrite(fid, NX_GLOBAL, 'int32'); % MPI config
  fwrite(fid, NY_GLOBAL, 'int32'); % MPI config
  fwrite(fid, KX0, 'int32'); % MPI config
  fwrite(fid, KY0, 'int32'); % MPI config
  fwrite(fid, NPAD, 'int32'); % MPI config
  fwrite(fid, NBANDS, 'int32');
  fwrite(fid, time_step, 'double');
  fwrite(fid, WGT_NXP, 'int32'); % nxp
  fwrite(fid, WGT_NYP, 'int32'); % nyp
  fwrite(fid, WGT_NFP, 'int32'); % nfp
  fwrite(fid, min_weight, 'float32'); % minVal
  fwrite(fid, max_weight, 'float32'); % maxVal
  fwrite(fid, WGT_NUMPATCHES, 'int32'); % numPatches
  for kx_proc = 1 : NX_PROCS
    for ky_proc = 1 : NY_PROCS
      for i_feature = 1 : NFEATURES
	if COMPRESSED_FLAG 
	  weights_tmp = ...
	      weights{i_feature} .* ...
	      ( weights{i_feature} >= min_weight ) .*  ...
	      ( weights{i_feature} <= max_weight );
	  weights_tmp = ...
	      floor( (2^8) * ( weights_tmp - min_weight ) / ...
		    ( (max_weight - min_weight) + ( (max_weight - min_weight) == 0 ) ) );
	else
	  weights_tmp = weights{i_feature} .* ...
	      ( weights{i_feature} >= min_weight ) + ...
	      min_weight .* ...
	      ( weights{i_feature} < min_weight );
	endif
	weights_tmp = ...
	    reshape( weights_tmp(:), weights_size );
	weights_tmp = ...
	    weights_tmp( :, ...
			col_index_first : col_index_last, ...
			row_index_first : row_index_last );
	fwrite(fid, uint16(WGT_NXP), 'uint16');
	fwrite(fid, uint16(WGT_NYP), 'uint16');
	if COMPRESSED_FLAG 
	  fwrite(fid, uint8(weights_tmp), 'uint8');
	else
	  fwrite(fid, weights_tmp, 'float32');	  
	endif
       endfor
    endfor
  endfor
  fclose(fid);

endfunction %% pvp_writeKernel