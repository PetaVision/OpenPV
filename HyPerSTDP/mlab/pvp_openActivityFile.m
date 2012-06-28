function [pvp_fid, ...
	  pvp_header, ...
	  pvp_index ] = ...
      pvp_openActivityFile(pvp_path, pvp_layer)


  filename = ["a", num2str(pvp_layer-1),".pvp"];
  pvp_filename = [pvp_path, filename];

  if ~exist(pvp_filename,"file")
    disp(["~exist(pvp_filename,""file"") in pvp file: ", pvp_filename]);
    pvp_fid = -1;
    return;
  endif

  [pvp_fid, pvp_header, pvp_index] = pvp_readActivityHeader(pvp_filename);

  num_pvp_params = pvp_header(pvp_index.NUM_PARAMS);
  if ( num_pvp_params ~= 20 )
    disp(["num_pvp_params ~= 20 in pvp file: ", filename]);
  endif

