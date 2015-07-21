function [W_array] = stdp_compAverageWeightsEvol(fname)

global input_dir N n_time_steps begin_step patch_size write_step

% can read patch_size from the header

%read membrane potentials
filename = fname;
filename = [input_dir, filename];
if begin_step > n_time_steps
    begin_step = 1;
end
%W_array = zeros( (n_time_steps - begin_step + 1)/write_step, N*patch_size );
if exist(filename, 'file')
    fid = fopen(filename, 'r', 'native');
    num_params = fread(fid, 1, 'int')
    pause
    for i_param = 1 : num_params
        tmp = fread(fid, 1, 'int')
        pause
    end
    pause
    fread(fid, N * patch_size * ( begin_step - 1 ), 'float');
    for i_step = begin_step : write_step: n_time_steps
        [w_step, Ntmp] = fread(fid, N*patch_size, 'float');
        W_array(i_step - begin_step + 1,:) = w_step;
    end
    fclose(fid);
    
else
    disp(['Skipping, could not open ', filename]);
end


