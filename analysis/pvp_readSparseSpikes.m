function [spike_count, ...
	  step_count] = ...
      pvp_readSparseSpikes(fid, ...
			   exclude_offset, ...
			   total_spikes, ...
			   total_steps, ...
			   pvp_order)

  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer
  global NO NK % for the current layer
  global BIN_STEP_SIZE DELTA_T
  global BEGIN_TIME END_TIME 
  global pvp_header pvp_index
  global NUM_TRIALS i_trial

  global SPIKE_ARRAY

  if nargin < 2|| exclude_offset <= 0 || isempty(exclude_offset)
    exclude_offset = ...
	 pvp_header(pvp_index.HEADER_SIZE);
  end%%if
  if nargin < 3|| total_spikes <= 0 || isempty(total_spikes)
    total_spikes = inf;
  end%%if
  if nargin < 4|| total_steps <= 0 || isempty(total_steps)
    total_stepss = inf;
  end%%if
  if nargin < 5 || isempty(pvp_order)
    pvp_order = 1;
  end%%if

pvp_status = fseek(fid, exclude_offset, 'bof');
if ( pvp_status == -1 )
  error(['fseek(fid, exclude_offset, ''bof'') == -1 in pvp file: ', filename]);
  return;
end%%if

%SPIKE_ARRAY = ...
%    sparse([], [], [], num_sparse_steps, N, total_spikes);
spike_id = zeros(total_spikes,1);
spike_step = zeros(total_spikes,1);
spike_count = 0;
step_count = 0;
while ~feof(fid)
  spike_time = fread(fid,1,'float64');
  num_spikes = fread(fid, 1, 'int32');
  spike_id_tmp = fread(fid, num_spikes, 'int32'); % spike_id_tmp is a column vector
  if ( spike_time < BEGIN_TIME)
    continue;
  end%%if
  if ~pvp_order
    [ f_spike_id, col_spike_id, row_spike_id ] = ...
	ind2sub( [NFEATURES, NCOLS, NROWS], spike_id_tmp );
    spike_id_tmp = sub2ind( [NROWS, NCOLS, NFEATURES], f_spike_id, col_spike_id, row_spike_id );
  end%%if
  step_count = step_count + 1;
%    SPIKE_ARRAY(sparse_step, spike_id_tmp+1) = 1; %int8(1);
  spike_id(spike_count+1:spike_count+num_spikes) = ...
      spike_id_tmp+1;
  spike_step(spike_count+1:spike_count+num_spikes) = ...
      repmat(step_count, num_spikes, 1);
  spike_count = spike_count + num_spikes;
  if ( spike_time > END_TIME ) || ...
	( spike_count >= total_spikes ) || ...
	( step_count >= total_steps )
    break;
  end%%if
end%%for
%total_spikes = spike_count;
%total_steps = step_count;
%fclose(fid);
%ave_rate = 1000 * total_spikes / ( N * ( END_TIME - BEGIN_TIME + DELTA_T ) );
if ~pvp_order
    [ f_spike_id, col_spike_id, row_spike_id ] = ind2sub( [NFEATURES, NCOLS, NROWS], spike_id );
    spike_id = sub2ind( [NROWS, NCOLS, NFEATURES], f_spike_id, col_spike_id, row_spike_id );
end%%if
SPIKE_ARRAY = sparse(spike_step, spike_id, 1, total_steps, N, total_spikes);
clear spike_step spike_id
SPIKE_ARRAY = spones(SPIKE_ARRAY);






