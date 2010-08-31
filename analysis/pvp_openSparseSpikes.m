function [fid, ...
	  total_spikes, ...
	  total_steps,...
	  exclude_spikes,...
	  exclude_steps, ...
	  exclude_offset ] = ...
      pvp_openSparseSpikes(layer)

  global OUTPUT_PATH 
  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer
  global NO NK % for the current layer
  global BIN_STEP_SIZE DELTA_T
  global BEGIN_TIME END_TIME 
  global pvp_header pvp_index

  pvp_fileTypes;

				% PetaVision always names spike files aN.pvp, where
				% N == layer index (starting at 0)
  filename = ['a', num2str(layer-1),'.pvp'];
  filename = [OUTPUT_PATH, filename];

  disp([ 'read tot spikes from ',filename ]);
  
				%default return arguments
if ~exist(filename,'file')
    error(['~exist(filename,''file'') in pvp file: ', filename]);
    return;
end%%if

[pvp_header, pvp_index] = pvp_readHeader(filename);
if isempty(pvp_header)
    error(['isempty(pvp_header) in pvp file: ', filename]);
    return;
end%%if

file_type = pvp_header(pvp_index.FILE_TYPE);
if ( file_type ~= PVP_ACT_FILE_TYPE )
    error(['file_type ~= PVP_ACT_FILE_TYPE in pvp file: ', filename]);
    return;
end%%if

pvp_BEGIN_TIME = pvp_header(pvp_index.TIME);
if ( pvp_BEGIN_TIME > 1 )
    disp(['pvp_BEGIN_TIME = ', num2str(pvp_BEGIN_TIME), ' in pvp file: ', filename]);
end%%if
DELTA_T = 1.0; % TODO: should be read in from header

% adjust begin_step relative to pvp_BEGIN_TIME
if BEGIN_TIME > pvp_BEGIN_TIME
    BEGIN_TIME = BEGIN_TIME - pvp_BEGIN_TIME;
else
    BEGIN_TIME = pvp_BEGIN_TIME;
end%%if

num_pvp_params = pvp_header(pvp_index.NUM_PARAMS);
if ( num_pvp_params ~= 20 )
    error(['num_pvp_params ~= 20 in pvp file: ', filename]);
    return;
end%%if

NCOLS = pvp_header(pvp_index.NX);
NROWS = pvp_header(pvp_index.NY);
NFEATURES = pvp_header(pvp_index.NF);
N = NROWS * NCOLS * NFEATURES;
NO = floor( NFEATURES / NK );

fid = fopen(filename, 'r', 'native');

pvp_status = fseek(fid, pvp_header(pvp_index.HEADER_SIZE), 'bof');
if ( pvp_status == -1 )
    error(['fseek(fid, pvp_header(pvp_index.HEADER_SIZE), ''bof'') == -1 in pvp file: ', filename]);
    return;
end%%if

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
  end%%if
  [num_spikes, count] = fread(fid, 1, 'int32');
  spike_id_tmp = fread(fid, num_spikes, 'int32'); 
  if ( spike_time < BEGIN_TIME)
    exclude_steps = exclude_steps + 1;
    exclude_spikes = exclude_spikes + num_spikes;
    exclude_offset = ftell(fid);
    continue;
  end%%if
  total_spikes = total_spikes + num_spikes;
  total_steps = total_steps + 1;
  if ( ~isempty(spike_time) && spike_time > END_TIME)
    break;
  end%%if
end%%while

END_TIME = prev_spike_time;








