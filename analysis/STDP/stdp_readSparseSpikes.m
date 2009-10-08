function [spikes, ave_rate] = stdp_readSparseSpikes(fname)

global input_dir N n_time_steps begin_step NX NY NO NK

filename = fname;
filename = [input_dir, filename];

ave_rate = 0;
if begin_step > n_time_steps
    begin_step = 1;
end

if exist(filename,'file')
    total_spikes = 0;
    fid = fopen(filename, 'r', 'native');
    %      header
    %      params[0] = nParams;
    %      params[1] = nx;
    %      params[2] = ny;
    %      params[3] = nf;
    num_params = fread(fid, 1, 'int');
    NX = fread(fid, 1, 'int');
    NY = fread(fid, 1, 'int');
    NO = fread(fid, 1, 'int');
    fprintf('num_params = %d NX = %d NY = %d NO = %d \n',num_params,NX,NY,NO);
    %pause
    NK = 1;
    N = NX * NY * NO;
    for i_step = 1 : n_time_steps
        num_spikes = fread(fid, 1, 'float');
        eofstat = feof(fid);
%         fprintf('eofstat = %d\n', eofstat);
        if (feof(fid))
            n_time_steps = i_step - 1;
            fprintf('feof reached: n_time_steps = %d\n',n_time_steps);
            break;
        end
         
        %fprintf('%d: number of spikes = %f\n', i_step, num_spikes);

        fread(fid, num_spikes, 'float');
        if i_step < begin_step
            continue
        end
        total_spikes = total_spikes + num_spikes;
    end
    fclose(fid);
%     pack;
    fid = fopen(filename, 'r', 'native');
    for i_params = 1 : num_params + 1 % correct!! to include 
                                      % reading of num_params
        tmp = fread(fid, 1, 'int');
    end
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
         if mod(i_step - begin_step + 1, 1000) == 0
             disp(['i_step = ', num2str(i_step)]);
         end
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


