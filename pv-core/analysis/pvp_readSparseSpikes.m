function [spike_array] = ...
      pvp_readSparseSpikes(layer, ...
			   i_epoch, ...
			   epoch_struct, ...
			   layer_struct, ...
			   pvp_order)

  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer
  global NO NK % for the current layer
  global BIN_STEP_SIZE DELTA_T
  global BEGIN_TIME END_TIME 
  %%  global spike_array

  if nargin <= 4 || isempty(pvp_order)
    pvp_order = 1;
  endif

  %% init BEGIN/END times for each epoch
  BEGIN_TIME = epoch_struct.begin_time(i_epoch, layer) - DELTA_T;
  END_TIME = BEGIN_TIME;  % read 1 line
  
  %% get fid, only reads 1 time step
  [fid] = ...
      pvp_openSparseSpikes(layer);
  
  BEGIN_TIME = epoch_struct.begin_time(i_epoch, layer);
  END_TIME = epoch_struct.end_time(i_epoch, layer);
  exclude_offset = epoch_struct.exclude_offset(i_epoch, layer);
  total_spikes = epoch_struct.total_spikes(i_epoch, layer);
  total_steps = epoch_struct.total_steps(i_epoch, layer);        
  
  pvp_status = fseek(fid, exclude_offset, 'bof');
  if ( pvp_status == -1 )
    error(['fseek(fid, exclude_offset, ''bof'') == -1 in pvp file: ', filename]);
    return;
  endif

  spike_id = zeros(total_spikes,1);
  spike_step = zeros(total_spikes,1);
  spike_count = 0;
  step_count = 0;
  while ~feof(fid)
    spike_time = fread(fid,1,'float64');
    num_spikes = fread(fid, 1, 'int32');
    spike_id_tmp = fread(fid, num_spikes, 'int32'); % spike_id_tmp is a column vector
    if ( spike_time < epoch_struct.begin_time(i_epoch, layer))
      continue;
    endif
    if ~pvp_order
      [ f_spike_id, col_spike_id, row_spike_id ] = ...
	  ind2sub( [NFEATURES, NCOLS, NROWS], spike_id_tmp );
      spike_id_tmp = sub2ind( [NROWS, NCOLS, NFEATURES], f_spike_id, col_spike_id, row_spike_id );
    endif
    step_count = step_count + 1;
				%    spike_array(sparse_step, spike_id_tmp+1) = 1; %int8(1);
    if length( spike_id ) == spike_count
      spike_id = [spike_id; zeros(num_spikes,1)];
    elseif length( spike_id ) < (spike_count + num_spikes)
      disp(['length( spike_id ) = ', ...
	    num2str(length( spike_id )), ...
	    ' >= (spike_count + num_spikes)', ...
	    num2str(spike_count + num_spikes)]);
      error('length( spike_id ) >= (spike_count + num_spikes)');
    endif
    spike_id(spike_count+1:spike_count+num_spikes) = ...
	spike_id_tmp+1;

    if length( spike_step ) == spike_count
      spike_step = [spike_step; zeros(num_spikes,1)];
    elseif length( spike_step ) < (spike_count + num_spikes)
      disp(['length( spike_step ) = ', ...
	    num2str(length( spike_step )), ...
	    ' >= (spike_count + num_spikes)', ...
	    num2str(spike_count + num_spikes)]);
      error('length( spike_step ) >= (spike_count + num_spikes)');
    endif
    spike_step(spike_count+1:spike_count+num_spikes) = ...
	repmat(step_count, num_spikes, 1);

    spike_count = spike_count + num_spikes;
    if ( spike_time > epoch_struct.end_time(i_epoch, layer) ) || ...
	  ( spike_count >= total_spikes ) || ...
	  ( step_count >= total_steps )
      break;
    endif
  endwhile

  if spike_count ~= total_spikes
    disp(['spike_count = ', num2str(spike_count)]);
    disp(['total_spikes = ', num2str(total_spikes)]);
    error('spike_count ~= total_spikes');
  endif
  if step_count > total_steps %% total_spikes could be read before end
    %% of epoch
    disp(['step_count = ', num2str(step_count)]);
    disp(['total_steps = ', num2str(total_steps)]);
    error('step_count ~= total_steps');
  endif

  if ~pvp_order
    [ f_spike_id, col_spike_id, row_spike_id ] = ...
	ind2sub( [NFEATURES, NCOLS, NROWS], spike_id );
    spike_id = ...
	sub2ind( [NROWS, NCOLS, NFEATURES], f_spike_id, col_spike_id, row_spike_id );
  endif
  spike_array = sparse(spike_step, spike_id, 1, total_steps, N, total_spikes);
  spike_array = spones(spike_array);
endfunction %% readSparseSpikes





