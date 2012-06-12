function [weights, nxp, nyp, offset] = ...
      pvp_readWeights(weights_filename, pvp_header)
  
  global SPIKE_PATH 
  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer
  global NO NK % for the current layer
  global NUM_ARBORS
  global MAX_ARBORS
  
  global NUM_BIN_PARAMS 
  global NUM_WGT_PARAMS
  
  
  global N_CONNECTIONS
  global NXP NYP NFP
  global COMPRESSED_FLAG
    
  if exist("SPIKE_PATH", "dir")
    weights_filename = [SPIKE_PATH,weights_filename];   
  endif
  if ~exist(weights_filename,'file')
    error(['~exist(weights_filename,''file'') in pvp file: ', weights_filename]);
  endif  
  fprintf(["reading connection weights from %s\n"],weights_filename);
	   
  %%default return arguments
  pvp_kernel = [];
  ave_weight = 0;
	 
  if ~exist(pvp_header) || isempty(pvp_header)
    [pvp_header, pvp_index] = pvp_readWeightHeader(weights_filename);
  endif
  if isempty(pvp_header)
    disp(['isempty(pvp_header) in pvp file: ', weights_filename]);
    return;
  endif

  
  num_pvp_params = pvp_header(pvp_index.NUM_PARAMS);
  if ( num_pvp_params ~= 26 )
    disp(['num_pvp_params ~= 26 in pvp file: ', weights_filename]);
  endif
  
  NXP = pvp_header(pvp_index.WGT_NXP);
  NYP = pvp_header(pvp_index.WGT_NYP);
  NFP = pvp_header(pvp_index.WGT_NFP);
  weight_min = pvp_header(pvp_index.WGT_MIN);
  weight_max = pvp_header(pvp_index.WGT_MAX);
  num_patches = pvp_header(pvp_index.WGT_NUMPATCHES);
  NX_PROCS = pvp_header(pvp_index.NX_PROCS);
  NY_PROCS = pvp_header(pvp_index.NY_PROCS);
  num_patches = num_patches; %% / (NX_PROCS * NY_PROCS);  %% latest PV fileio writes mpi independent kernels
  
  
  NCOLS = pvp_header(pvp_index.WGT_NXP);
  NROWS = pvp_header(pvp_index.WGT_NYP);
  NFEATURES = pvp_header(pvp_index.WGT_NFP);
  N = NROWS * NCOLS * NFEATURES;

  NUM_ARBORS = pvp_header(pvp_index.NUM_ARBORS);
  if MAX_ARBORS > NUM_ARBORS + 1
    MAX_ARBORS = NUM_ARBORS + 1;
  endif
  
  disp(['NXP = ', num2str(NXP)]);
  disp(['NYP = ', num2str(NYP)]);
  disp(['NFP = ', num2str(NFP)]);
  disp(['num_patches = ', num2str(num_patches)]);
  disp(['weight_min = ', num2str(weight_min)]);
  disp(['weight_max = ', num2str(weight_max)]);
  
  fid = fopen(weights_filename, 'r');
  
  data_size = pvp_header(pvp_index.DATA_SIZE);
  if data_size > 1
    COMPRESSED_FLAG = 0;
  elseif data_size == 1
    COMPRESSED_FLAG = 1;
  endif

  nxp = repmat(NXP, num_patches,1); %%MAX_ARBORS);
  nyp = repmat(NYP, num_patches,1); %%MAX_ARBORS);
  offset = repmat(NYP, num_patches,1); %%MAX_ARBORS);
  weights = cell(num_patches,MAX_ARBORS);
   
  for i_arbor = 1 : MAX_ARBORS %% min(NUM_ARBORS, MAX_ARBORS)

    if i_arbor < MAX_ARBORS %% min(NUM_ARBORS, MAX_ARBORS)
      arbor_list = i_arbor-1;
    elseif i_arbor == MAX_ARBORS %% min(NUM_ARBORS, MAX_ARBORS)
      arbor_list = [0:NUM_ARBORS-1];
    endif

    for i_patch = 1 : num_patches
      weights{i_patch,i_arbor} = zeros(NXP * NYP * NFP,1);
    endfor

    arbor_count = 0
    for j_arbor = arbor_list

      arbor_count = arbor_count + 1;
      arbor_offset = j_arbor * NXP * NYP * NFP * 4;
      pvp_status = fseek(fid, pvp_header(pvp_index.HEADER_SIZE) + arbor_offset, 'bof');
      if ( pvp_status == -1 )
	disp(['fseek(fid, pvp_header(pvp_index.HEADER_SIZE), ''bof'') == -1 in pvp file: ', weights_filename]);
	return;
      endif
    
      for i_patch = 1 : num_patches
	nxp(i_patch,1) = fread(fid, 1, 'uint16');
	nyp(i_patch,1) = fread(fid, 1, 'uint16');
	offset(i_patch,1) = fread(fid, 1, 'uint32');
	if COMPRESSED_FLAG 
	  weights_tmp = fread(fid, NXP * NYP * NFP, 'uint8');
	  weights{i_patch, i_arbor} = weights{i_patch, i_arbor} + ...
	      weight_min + weights_tmp * ...
	      ( weight_max -  weight_min ) / 255;
	else
	  weights_tmp = fread(fid, NXP * NYP * NFP, 'float32');
	  weights{i_patch,i_arbor} = weights{i_patch, i_arbor} + weights_tmp;
	endif
      endfor %% i_patch
  
    endfor %% j_arbor
endfor %% i_arbor
fclose(fid);
