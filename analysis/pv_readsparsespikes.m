function [spikes, ave_rate] = pv_readsparsespikes(fname)

global output_path N n_time_steps begin_step

filename = fname;
filename = [output_path, filename];

ave_rate = 0;
if begin_step > n_time_steps
    begin_step = 1;
end

if exist(filename,'file')
    total_spikes = 0;
    fid = fopen(filename, 'r', 'native');
    for i_step = 1 : n_time_steps
        num_spikes = fread(fid, 1, 'float');
        eofstat = feof(fid);
%         fprintf('eofstat = %d\n', eofstat);
        if (feof(fid))
            n_time_steps = i_step - 1;
            fprintf('feof reached');
            break;
        end
%         if num_spikes == []
%             fprintf('end of file\n');
%         else
%             fprintf('%d: number of spikes = %f\n', i_step, num_spikes);
%         end

        fread(fid, num_spikes, 'float');
        if i_step < begin_step
            continue
        end
        total_spikes = total_spikes + num_spikes;
    end
    fclose(fid);
%     pack;
    fid = fopen(filename, 'r', 'native');
    spike_id = [];
    spike_step = [];
    for i_step = 1 : n_time_steps
        num_spikes = fread(fid, 1, 'float');
%         if num_spikes == []
%             fprintf('end of file\n');
%         else
%             fprintf('number of spikes = %f\n', num_spikes);
%         end
        
        spike_id_tmp = fread(fid, num_spikes, 'float');
        if i_step < begin_step
            continue
        end
        spike_id = [spike_id; spike_id_tmp+1];
        spike_step = [spike_step; repmat(i_step - begin_step + 1, num_spikes, 1)];
%         if mod(i_step - begin_step + 1, 10) == 0
%             disp(['i_step = ', num2str(i_step)]);
%         end
    end
    fclose(fid);
    spikes = sparse(spike_step, spike_id, 1, n_time_steps - begin_step + 1, N, total_spikes);
    ave_rate = 1000 * sum(spikes(:)) / ( N * ( n_time_steps - begin_step + 1 ) );
    %disp(['ave_rate = ', num2str(ave_rate)]);
else
    disp(['Skipping, could not open ', filename]);
    spikes = sparse([], [], [], n_time_steps - begin_step + 1, N, 0);
    ave_rate = 0; 
end


