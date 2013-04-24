function [pvp_time,...
	  pvp_activity, ...
	  pvp_offset] = ...
      pvp_readSparseLayerActivity(pvp_fid, pvp_frame, pvp_header, pvp_index, pvp_offset)

  global VERBOSE_FLAG

  NCOLS = pvp_header(pvp_index.NX_GLOBAL);  
  NROWS = pvp_header(pvp_index.NY_GLOBAL);
  NFEATURES = pvp_header(pvp_index.NF);
  N = NFEATURES * NCOLS * NROWS; 
  NUM_BIN_PARAMS = 20;
  %% use offset if available or else just set pvp_fid to end of header
  pvp_time = 0.0;
  pvp_activity = [];
  if pvp_offset == 0    
    pvp_status = fseek(pvp_fid, 4*(NUM_BIN_PARAMS), "bof");  % set pvp_fid of end of header
    if ( pvp_status == -1 )
      disp(["fseek(fid, pvp_frame_offset, ""cof"") == -1 in pvp_frame: ", num2str(pvp_frame)]);
      return;
    endif
  else
    pvp_status = fseek(pvp_fid, pvp_offset, "bof");
    if ( pvp_status == -1 )
      error(["fseek(pvp_fid, exclude_offset, ""bof"") == -1 in pvp file: ", filename]);
      return;
    endif
  endif

  while pvp_time < pvp_frame
    [pvp_time, count] = fread(pvp_fid,1,"float64");
    if feof(pvp_fid) 
      break;
    endif
    if VERBOSE_FLAG
      disp(["pvp_time = ", num2str(pvp_time)]);
    endif
    if count == 0
      error(["count == 0: ", "pvp_time = ", num2st(pvp_time)]);
    endif
    [num_spikes, count] = fread(pvp_fid, 1, "int32");
    if VERBOSE_FLAG
      disp(["num_spikes = ", num2str(num_spikes)]);
    endif
    if count == 0
      error(["count == 0: ", "num_spikes = ", num2st(num_spikes)]);
    endif
    [spike_id, count] = fread(pvp_fid, num_spikes, "int32"); 
    if count ~= num_spikes
      error(["count ~= num_spikes, ", "count = ", num2st(count), ", num_spikes = ", num2st(num_spikes)]);      
    endif
  endwhile
  if feof(pvp_fid)
    pvp_offset = -1;
    pvp_time = -1;
    return;
  endif
  pvp_offset = ftell(pvp_fid);
  pvp_activity = sparse(spike_id+1, 1, 1, N, 1, num_spikes); %%%
%%%compensate for PetaVision starting at zero

endfunction %% readSparseSpikes












