function [pvp_fid, ...
	  pvp_header, ...
	  pvp_index ] = ...
      pvp_openActivityFile(pvp_path, pvp_layer)


  pvp_filename = ["a", num2str(layer-1),".pvp"];
  pvp_filename = [pvp_path, filename];

  if ~exist(pvp_filename,"file")
    disp(["~exist(pvp_filename,""file"") in pvp file: ", pvp_filename]);
    return;
  endif

  [pvp_fid, pvp_header, pvp_index] = pvp_readActivityHeader(pvp_filename);

  file_type = pvp_header(pvp_index.FILE_TYPE);
  if ( file_type ~= PVP_NONSPIKING_ACT_FILE_TYPE )
    disp(["file_type ~= PVP_NONSPIKING_ACT_FILE_TYPE in pvp file: ", filename]);
  endif

  num_pvp_params = pvp_header(pvp_index.NUM_PARAMS);
  if ( num_pvp_params ~= 20 )
    disp(["num_pvp_params ~= 20 in pvp file: ", filename]);
  endif



  



