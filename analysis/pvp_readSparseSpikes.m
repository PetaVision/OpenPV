function [spike_array, ave_rate] = pvp_readSparseSpikes(layer, pvp_order)

  global output_path 
  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer
  global NO NK % for the current layer
  global n_time_steps begin_step end_step time_steps tot_steps
  global stim_begin_step stim_end_step stim_steps 
  global bin_size dt
  global begin_time end_time stim_begin_time stim_end_time 
  global pvp_header pvp_index

  if nargin < 2
    pvp_order = 1;
  endif

				% PetaVision always names spike files aN.pvp, where
				% N == layer index (starting at 0)
  filename = ['a', num2str(layer-1),'.pvp'];
  filename = [output_path, filename];

  disp([ 'read spikes from ',filename ]);
  
				%default return arguments
  spike_array = [];
  ave_rate = 0;

  if ~exist(filename,'file')
    disp(['~exist(filename,''file'') in pvp file: ', filename]);
    return;
  endif

  [pvp_header, pvp_index] = pvp_readHeader(filename);
  if isempty(pvp_header)
    disp(['isempty(pvp_header) in pvp file: ', filename]);
    return;
  endif

  pvp_begin_time = pvp_header(pvp_index.TIME);
  if ( pvp_begin_time > 1 )
    disp(['pvp_begin_time = ', num2str(pvp_begin_time), ' in pvp file: ', filename]);
  endif
  dt = 1.0; % TODO: should be read in from header

				% adjust begin_step relative to pvp_begin_time
  if begin_time > pvp_begin_time
    begin_time = begin_time - pvp_begin_time;
  else
    begin_time = pvp_begin_time;
  endif
  begin_step = max(1, ceil( begin_time / dt ) );

  if ( stim_begin_time < begin_time )
    stim_begin_time = begin_time;
  endif
  stim_begin_step = ( stim_begin_time - pvp_begin_time ) / dt + 1;

  num_pvp_params = pvp_header(pvp_index.NUM_PARAMS);
  if ( num_pvp_params ~= 20 )
    disp(['num_pvp_params ~= 20 in pvp file: ', filename]);
  endif

  NCOLS = pvp_header(pvp_index.NX);
  NROWS = pvp_header(pvp_index.NY);
  NFEATURES = pvp_header(pvp_index.NF);
  N = NROWS * NCOLS * NFEATURES;
  NO = floor( NFEATURES / NK );

  fid = fopen(filename, 'r', 'native');

  pvp_status = fseek(fid, pvp_header(pvp_index.HEADER_SIZE), 'bof');
  if ( pvp_status == -1 )
    disp(['fseek(fid, pvp_header(pvp_index.HEADER_SIZE), ''bof'') == -1 in pvp file: ', filename]);
    Return;
  endif

				% TODO: Should get n_time_steps from NUM_RECORDS, but not yet implemented
				% in PetaVision: i.e.
				% n_time_steps = pvp_header(pvp_index.NUM_RECORDS)
				% so instead get n_time_steps by counting lines in pvp file
  n_time_steps = 0;
  total_spikes = 0;
  spike_time = fread(fid,1,'float64');
  
  while ~feof(fid)
    n_time_steps = n_time_steps + 1;
    num_spikes = fread(fid, 1, 'int32');
    pvp_offset = num_spikes * sizeof2(int32(0));
    pvp_status = fseek(fid, pvp_offset, 'cof');
    if pvp_status == -1
      disp(['pvp_status = fseek(fid, pvp_offset, ''cof'') == -1 in pvp file: ', filename]);
      return;
    endif
    old_spike_time = spike_time;
    spike_time = fread(fid,1,'float64');
    if ( old_spike_time < begin_time)
      continue;
    endif 
    total_spikes = total_spikes + num_spikes;
    if ( ~isempty(spike_time) && spike_time > end_time)
      break;
    endif  
  endwhile  
  end_time = old_spike_time;
  end_step = n_time_steps - begin_step + 1;
  time_steps = begin_step:end_step;
  time_steps = time_steps - begin_step + 1;
  tot_steps = length(time_steps);

  if ( stim_end_time > end_time )
    stim_end_time = end_time;
  endif
  stim_end_step = ( stim_end_time - pvp_begin_time ) / dt + 1;
  stim_steps = stim_begin_step : stim_end_step;

  pvp_status = fseek(fid, pvp_header(pvp_index.HEADER_SIZE), 'bof');
  if ( pvp_status == -1 )
    disp(['fseek(fid, pvp_header(pvp_index.HEADER_SIZE), ''bof'') == -1 in pvp file: ', filename]);
    return;
  endif

  spike_id = zeros(total_spikes,1);
  spike_step = zeros(total_spikes,1);
  total_spikes = 0;
  for i_step = 1 : n_time_steps
    spike_time = fread(fid,1,'float64');
    num_spikes = fread(fid, 1, 'int32');
    spike_id_tmp = fread(fid, num_spikes, 'int32'); % spike_id_tmp is a column vector
    if ( spike_time < begin_time)
      continue;
    endif   
    if ( spike_time ~= (begin_step + i_step - 1) * dt ) &&  ...
	  ( spike_time ~= (begin_step + i_step - 2) )
      disp(['spike_time ~= (begin_step + i_step) * dt in pvp file: ', filename]);
    endif
    spike_id(total_spikes+1:total_spikes+num_spikes) = spike_id_tmp+1;
    spike_step(total_spikes+1:total_spikes+num_spikes) = repmat(i_step - begin_step + 1, num_spikes, 1);
    total_spikes = total_spikes + num_spikes;
  endfor
  fclose(fid);
  ave_rate = 1000 * total_spikes / ( N * ( end_time - begin_time + dt ) );
  if ~pvp_order
    [ f_spike_id, col_spike_id, row_spike_id ] = ind2sub( [NFEATURES, NCOLS, NROWS], spike_id );
    spike_id = sub2ind( [NROWS, NCOLS, NFEATURES], f_spike_id, col_spike_id, row_spike_id );
  endif
  spike_array = sparse(spike_step, spike_id, 1, n_time_steps - begin_step + 1, N, total_spikes);






