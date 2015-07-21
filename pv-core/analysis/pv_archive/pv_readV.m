function [v_array] = pv_readV(fname, neuron_ndx)

global input_dir n_time_steps begin_step
global NK NO NX NY

%read membrane potentials
filename = fname;
filename = [input_dir, filename];
if begin_step > n_time_steps
    begin_step = 1;
end
if exist(filename, 'file')
    fid = fopen(filename, 'r', 'native');
    num_params = fread(fid, 1, 'int');
    params = zeros(num_params,1);
    NVmem = 1;
    for i_param = 1 : num_params
        params(i_param) = fread(fid, 1, 'int');
        NVmem = NVmem * params(i_param);
    end
    if isempty(neuron_ndx)
        Vmem_ndx = ceil(NVmem/2);
    else
        Vmem_ndx = pv_convertNdx(neuron_ndx, params, NX, NY, NK);
    end
    v_array = zeros( n_time_steps - begin_step + 1, numel(Vmem_ndx) );
    fread(fid, NVmem * ( begin_step - 1 ), 'float');
    for i_step = begin_step : n_time_steps
        [v_step, Ntmp] = fread(fid, NVmem, 'float');
        v_array(i_step - begin_step + 1,:) = ...
            v_step( Vmem_ndx );
    end
    fclose(fid);
    %disp(['ave_rate = ', num2str(ave_rate)]);
else
    disp(['Skipping, could not open ', filename]);
end


