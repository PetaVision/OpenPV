function [fid, ...
	  pvp_header, ...
	  pvp_index, ...
	  total_spikes, ...
	  total_steps,...
	  exclude_spikes,...
	  exclude_steps, ...
	  exclude_offset ] = ...
      pvp_openSparseSpikes(layer)

  global SPIKE_PATH 
				%  global N NROWS NCOLS % for the current layer
				%  global NFEATURES  % for the current layer
				%  global NO NK % for the current layer
  global BIN_STEP_SIZE DELTA_T
  global BEGIN_TIME END_TIME 
				%  global pvp_header pvp_index


				% PetaVision always names spike files aN.pvp, where
				% N == layer index (starting at 0)
  filename = ['a', num2str(layer-1),'.pvp'];
  filename = [SPIKE_PATH, filename];

  disp([ 'read tot spikes from ',filename ]);
  
  [pvp_header, pvp_index] = pvp_readHeader(filename);
  if isempty(pvp_header)
    error(['isempty(pvp_header) in pvp file: ', filename]);
    return;
  endif

  pvp_BEGIN_TIME = pvp_header(pvp_index.TIME);
  if ( pvp_BEGIN_TIME > 1 )
    disp(['pvp_BEGIN_TIME = ', num2str(pvp_BEGIN_TIME), ' in pvp file: ', filename]);
  endif
  DELTA_T = 1.0; % TODO: should be read in from header

				% adjust begin_step relative to pvp_BEGIN_TIME
  if BEGIN_TIME > pvp_BEGIN_TIME
    BEGIN_TIME = BEGIN_TIME - pvp_BEGIN_TIME;
  else
    BEGIN_TIME = pvp_BEGIN_TIME;
  endif

  fid = fopen(filename, 'r', 'native');

  pvp_status = fseek(fid, pvp_header(pvp_index.HEADER_SIZE), 'bof');
  if ( pvp_status == -1 )
    error(['fseek(fid, pvp_header(pvp_index.HEADER_SIZE), ''bof'') == -1 in pvp file: ', filename]);
    return;
  endif

				% TODO: Should get total_steps from NUM_RECORDS, but not yet implemented
				% in PetaVision: i.e.
				% total_steps = pvp_header(pvp_index.NUM_RECORDS)
				% so instead get total_steps by counting lines in pvp file
  total_steps = 0;
  total_spikes = 0;
  exclude_steps = 0;
  exclude_spikes = 0;
  exclude_offset = ftell(fid);
				%spike_time = fread(fid,1,'float64');
  spike_time = pvp_BEGIN_TIME;

  while ~feof(fid)
    prev_spike_time = spike_time;
    [spike_time, count] = fread(fid,1,'float64');
    if count == 0
      break;
    endif
    [num_spikes, count] = fread(fid, 1, 'int32');
    spike_id_tmp = fread(fid, num_spikes, 'int32'); 
    if ( spike_time < BEGIN_TIME)
      exclude_steps = exclude_steps + 1;
      exclude_spikes = exclude_spikes + num_spikes;
      exclude_offset = ftell(fid);
      continue;
    endif
    total_spikes = total_spikes + num_spikes;
    total_steps = total_steps + 1;
    if ( ~isempty(spike_time) && spike_time > END_TIME)
      break;
    endif
  endwhile

  END_TIME = prev_spike_time;








