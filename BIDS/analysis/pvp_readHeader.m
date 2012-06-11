function [pvp_header, pvp_index] = pvp_readHeader(filename)

  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer
  global NO NK % for the current layer

  pvp_fileTypes;

  pvp_index = struct;
  pvp_index.HEADER_SIZE  = 0+1;
  pvp_index.NUM_PARAMS   = 1+1;
  pvp_index.FILE_TYPE    = 2+1;
  pvp_index.NX           = 3+1;
  pvp_index.NY           = 4+1;
  pvp_index.NF           = 5+1;
  pvp_index.NUM_RECORDS  = 6+1;
  pvp_index.RECORD_SIZE  = 7+1;
  pvp_index.DATA_SIZE    = 8+1;
  pvp_index.DATA_TYPE    = 9+1;
  pvp_index.NX_PROCS    = 10+1;
  pvp_index.NY_PROCS    = 11+1;
  pvp_index.NX_GLOBAL   = 12+1;
  pvp_index.NY_GLOBAL   = 13+1;
  pvp_index.KX0         = 14+1;
  pvp_index.KY0         = 15+1;
  pvp_index.NPAD        = 16+1;
  pvp_index.NUM_ARBORS  = 17+1;
  pvp_index.TIME        = 18+1;

  if ~exist(filename,'file')
    error(['~exist(filename,''file'') in pvp file: ', filename]);
    return;
  endif

  fid = fopen(filename, 'r');
  if fid == -1
    pvp_header = [];
    return;
  endif
  pvp_header = fread(fid, NUM_BIN_PARAMS-2, 'int32');
  if isempty(pvp_header)
    error(['isempty(pvp_header) in pvp file: ', filename]);
    return;
  endif
  pvp_time = fread(fid, 1, 'double');
  fclose(fid);
  pvp_header(pvp_index.TIME) = pvp_time;

  file_type = pvp_header(pvp_index.FILE_TYPE);
  %disp(['file_type = ', num2str(file_type)]);
  %disp(['PVP_WGT_FILE_TYPE = ', num2str(PVP_WGT_FILE_TYPE)]);
  if ( ( file_type ~= PVP_NONSPIKING_ACT_FILE_TYPE ) && ...
  	 ( file_type ~= PVP_ACT_FILE_TYPE ) && ...
  	 ( file_type ~= PVP_WGT_FILE_TYPE ) && ...
  	 ( file_type ~= PVP_KERNEL_FILE_TYPE ) )
    error(['file_type = ', num2str(file_type), ' ~= PVP_NONSPIKING_ACT_FILE_TYPE in pvp file: ', filename]);
    return;
  endif


  num_pvp_params = pvp_header(pvp_index.NUM_PARAMS);
  if ( ( num_pvp_params ~= 20 ) &&  ( num_pvp_params ~= 26 ) )
    error(['num_pvp_params = ', num2str(num_pvp_params), ' ~= 20 in pvp file: ', filename]);
    return;
  endif

  NCOLS = pvp_header(pvp_index.NX_GLOBAL);
  NROWS = pvp_header(pvp_index.NY_GLOBAL);
  NFEATURES = pvp_header(pvp_index.NF);
  N = NFEATURES * NCOLS * NROWS;
  NO = floor( NFEATURES / NK );
