function [spike_times, ave_rate] = stdp_readSpikes(fname, k_index, begin_step, end_step)

global input_dir n_time_steps 

filename = fname;
filename = [input_dir, filename];
fprintf('read spike times from %s\n',filename);

debug = 0;  % if 1 prints spiking neurons

ave_rate = 0;


if exist(filename,'file')
    total_spikes = 0;
    fid = fopen(filename, 'r', 'native');
    [time,numParams,NX,NY,NF] = readHeader(fid);
    fprintf('NX = %d NY = %d NF = %d \n',NX,NY,NF);
    %pause
    
    N = NX * NY * NF;
    
    % intialize array of spike times
    
    
    spike_times = []; 
    
    
    for i_step = 1 : n_time_steps

        time = fread(fid,1,'float64');
        num_spikes = fread(fid, 1, 'int');
        S = fread(fid, num_spikes, 'int');

        if debug
            fprintf('%d: %f number of spikes = %d: ', ...
                i_step, time, num_spikes);
            for i=1:length(S)
                fprintf('%d ',S(i));
            end
            fprintf('\n');
            %pause
        end

        if i_step < begin_step
            continue
        end
        if i_step > end_step
            break
        end
        
        for k=1:num_spikes
            if S(k) == k_index
                spike_times(end+1) = time;
                ave_rate = ave_rate + 1;
                if debug
                    fprintf('%d %f\n',k_index, time)
                    pause
                end
            end
        end
                
        
        %pause
        if mod(i_step - begin_step, 10000) == 0
            disp(['i_step = ', num2str(i_step)]);
        end        

    end
    fclose(fid);
    ave_rate = (ave_rate * 1000.0)/(end_step - begin_step+1);
    %fprintf('ave_rate= %f\n',ave_rate);
    %spike_times
    %pause
    
else
    disp(['Skipping, could not open ', filename]);
    spike_times = [];
    ave_rate = 0; 
end

%
% end primary function
    
    
function [time,numParams,NX,NY,NF] = readHeader(fid)
% NOTE: see analysis/python/PVReadWeights.py for reading params
% We call this function first because it rewinds the input file

    head = fread(fid,3,'int');
    if head(3) ~= 2  % PV_INT_TYPE
       disp('incorrect file type')
       return
    end
    numParams = head(2);
    fseek(fid,0,'bof'); % rewind file
    params = fread(fid, numParams-2, 'int'); 
    %pause
    NX         = params(4);
    NY         = params(5);
    NF         = params(6);
    fprintf('numParams = %d ',numParams);
    fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
    %pause
    % read time - last two params
    time = fread(fid,1,'float64');
    fprintf('time = %f\n',time);
    %pause
    
% End subfunction 
%    
