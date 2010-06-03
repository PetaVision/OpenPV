function [weights, nxp, nyp, pvp_header, pvp_index] = pvp_readWeights(i_conn)

  global output_path 
  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer
  global NO NK % for the current layer


  global NUM_BIN_PARAMS 
  global NUM_WGT_PARAMS

  
  global N_CONNECTIONS
  global NXP NYP NFP
  
  if nargin < 1
    i_conn = 0;
  end
  
				% PetaVision always names spike files aN.pvp, where
				% N == layer index (starting at 0)
  filename = ['w', num2str(i_conn-1),'_last.pvp'];
  filename = [output_path, filename];
  
  fprintf(['read connection weights from %s\n'],filename);
  
				%default return arguments
  pvp_kernel = [];
  ave_weight = 0;
	 
  if ~exist(filename,'file')
    disp(['~exist(filename,''file'') in pvp file: ', filename]);
    return;
  end
  
  [pvp_header, pvp_index] = pvp_readWeightHeader(filename);
  if isempty(pvp_header)
    disp(['isempty(pvp_header) in pvp file: ', filename]);
    return;
  end
  
  num_pvp_params = pvp_header(pvp_index.NUM_PARAMS);
  if ( num_pvp_params ~= 26 )
    disp(['num_pvp_params ~= 26 in pvp file: ', filename]);
  end
  
				% presynaptic dimensions
				% TODO: change pvp header extension for weights to add postsynaptic info
  NXP = pvp_header(pvp_index.WGT_NXP);
  NYP = pvp_header(pvp_index.WGT_NYP);
  NFP = pvp_header(pvp_index.WGT_NFP);
  weight_min = pvp_header(pvp_index.WGT_MIN);
  weight_max = pvp_header(pvp_index.WGT_MAX);
  num_patches = pvp_header(pvp_index.WGT_NUMPATCHES);

%  if ( weight_max < -1 ) && ( weight_min == 0 )
%    weight_max = - 1 / weight_max;
%  end%%if

  
  NCOLS = pvp_header(pvp_index.WGT_NXP);
  NROWS = pvp_header(pvp_index.WGT_NYP);
  NFEATURES = pvp_header(pvp_index.WGT_NFP);
  N = NROWS * NCOLS * NFEATURES;

  disp(['NXP = ', num2str(NXP)]);
  disp(['NYP = ', num2str(NYP)]);
  disp(['NFP = ', num2str(NFP)]);
  disp(['num_patches = ', num2str(num_patches)]);
  disp(['weight_min = ', num2str(weight_min)]);
  disp(['weight_max = ', num2str(weight_max)]);

  fid = fopen(filename, 'r');
  pvp_status = fseek(fid, pvp_header(pvp_index.HEADER_SIZE), 'bof');
  if ( pvp_status == -1 )
    disp(['fseek(fid, pvp_header(pvp_index.HEADER_SIZE), ''bof'') == -1 in pvp file: ', filename]);
    Return;
  end

  nxp = repmat(NXP, num_patches,1);
  nyp = repmat(NYP, num_patches,1);
  weights = cell(num_patches,1);
  for i_patch = 1 : num_patches
    nxp(i_patch) = fread(fid, 1, 'uint16');
    nyp(i_patch) = fread(fid, 1, 'uint16');
    weights_tmp = fread(fid, nxp(i_patch) * nyp(i_patch) * NFP, 'uint8');
    weights{i_patch} = ...
	weight_min + weights_tmp * ...
	( weight_max -  weight_min ) / 255;
  end
  fclose(fid);

	  