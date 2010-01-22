function [spike_array, ave_rate] = pvp_readSparseSpikes(layer)

global input_path 
global N NX NY % for the current layer
global NF  % for the current layer
global n_time_steps begin_step end_step time_steps
global stim_begin_step stim_end_step stim_steps 
global bin_size dt
global begin_time stim_begin_time stim_end_time 

% PetaVision always names spike files aN.pvp, where
% N == layer index (starting at 0)
filename = ['a', num2str(layer-1),'.pvp'];
filename = [input_path, filename];

fprintf('read spikes from %s\n',filename);

%default return arguments
spike_array = [];
ave_rate = 0;

if ~exist(filename,'file')
    disp(['~exist(filename,''file'') in pvp file: ', filename]);
    return;
end

[pvp_header, pvp_index] = pvp_readPVPHeader(filename);
if isempty(pvp_header)
    disp(['isempty(pvp_header) in pvp file: ', filename]);
    return;
end

pvp_begin_time = pvp_header(pvp_index.TIME);
if ( pvp_begin_time > 0 )
    disp(['pvp_begin_time > 0 in pvp file: ', filename]);
end
dt = 1.0; % TODO: should be read in from header

% adjust begin_step relative to pvp_begin_time
if begin_time > pvp_begin_time
    begin_time = begin_time - pvp_begin_time;
else
    begin_time = pvp_begin_time;
end
begin_step = 1 + begin_time / dt;
stim_begin_step = ( stim_begin_time - pvp_begin_time ) / dt + 1;
stim_end_step = ( stim_end_time - pvp_begin_time ) / dt + 1;
stim_steps = stim_begin_step : stim_end_step;


num_pvp_params = pvp_header(pvpindex.NUM_PARAMS);
if ( num_pvp_params ~= 20 )
    disp(['num_pvp_params ~= 20 in pvp file: ', filename]);
end

NX = pvp_header(pvp_index.NX);
NY = pvp_header(pvp_index.NY);
NF = pvp_header(pvp_index.NF);
N = NX * NY * NF;

fid = fopen(filename, 'r', 'native');

pvp_status = fseek(fid, pvp_header(pvp_index.HEADER_SIZE), 'bof');
if ( pvp_status == -1 )
    disp(['fseek(fid, pvp_header(pvp_index.HEADER_SIZE), ''bof'') == -1 in pvp file: ', filename]);
    Return;
end

% TODO: Should get n_time_steps from NUM_RECORDS, but not yet implemented
% in PetaVision: i.e.
% n_time_steps = pvp_header(pvp_index.NUM_RECORDS)
% so instead get n_time_steps by counting lines in pvp file
n_time_steps = 0;
total_spikes = 0;
while ~feof(fid)
    n_time_steps = n_time_steps + 1;
    spike_time = fread(fid,1,'float64');
    num_spikes = fread(fid, 1, 'int32');
    pvp_offset = num_spikes * sizeof(int32(0));
    pvp_status = fseek(fid, pvp_offset, 'cof');
    if pvp_status == -1
       disp(['pvp_status = fseek(fid, pvp_offset, ''cof'') == -1 in pvp file: ', filename]);
       return;
    end
    if ( spike_time < begin_time)
        continue;
    end    
    total_spikes = total_spikes + num_spikes;
end
pvp_end_time = spike_time;
end_step = n_time_steps - begin_step + 1;
time_steps = begin_step:end_step;

pvp_status = fseek(fid, pvp_header(pvp_index.HEADER_SIZE), 'bof');
if ( pvp_status == -1 )
    disp(['fseek(fid, pvp_header(pvp_index.HEADER_SIZE), ''bof'') == -1 in pvp file: ', filename]);
    return;
end

spike_id = zeros(total_spikes,1);
spike_step = zeros(total_spikes,1);
total_spikes = 0;
for i_step = 1 : n_time_steps
    spike_time = fread(fid,1,'float64');
    num_spikes = fread(fid, 1, 'int32');
    spike_id_tmp = fread(fid, num_spikes, 'int32'); % spike_id_tmp is a column vector
    if ( spike_time < begin_time)
        continue;
    end    
    if ( spike_time ~= (begin_step + i_step) * dt )
       disp(['spike_time ~= (begin_step + i_step) * dt in pvp file: ', filename]);
    end
    spike_id(total_spikes+1:total_spikes+num_spikes) = spike_id_tmp+1;
    spike_step(total_spikes+1:total_spikes+num_spikes) = repmat(i_step - begin_step + 1, num_spikes, 1);
    total_spikes = total_spikes + num_spikes;
end
fclose(fid);
ave_rate = 1000 * total_spikes / ( N * ( pvp_end_time - begin_time + dt ) );
spike_array = sparse(spike_step, spike_id, 1, n_time_steps - begin_step + 1, N, total_spikes);




