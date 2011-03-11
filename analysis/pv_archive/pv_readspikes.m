function [spikes, ave_rate] = pv_readspikes(fname)

global output_path N NX NY NO n_time_steps begin_step

%read membrane potentials
filename = fname;
filename = [output_path, filename];
ave_rate = 0;
if begin_step > n_time_steps
    begin_step = 1;
end
%spikes = sparse([],[],[],n_time_steps - begin_step + 1, N, floor(N*(n_time_steps - begin_step + 1)*0.1) );
spikes = sparse(n_time_steps - begin_step + 1, N );

if exist(filename,'file')
    fid = fopen(filename, 'r', 'native');
    fread(fid, N * ( begin_step - 1 ), 'float');
    pack;
    for i_step = begin_step : n_time_steps
        [spikes_step, Ntmp] = fread(fid, N, 'float');
        spikes( i_step - begin_step + 1, spikes_step > 0 ) = 1;
        if mod(i_step - begin_step + 1, 10) == 0
            disp(['i_step = ', num2str(i_step)]);
        end
    end
    fclose(fid);
    ave_rate = 1000 * sum(spikes(:)) / ( N * ( n_time_steps - begin_step + 1 ) );
    %disp(['ave_rate = ', num2str(ave_rate)]);
else
    disp(['Skipping, could not open ', filename]);
end


