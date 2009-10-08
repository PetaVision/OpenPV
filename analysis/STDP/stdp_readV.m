function [v_array] = stdp_readV(fname,  neuron_ndx)

global input_dir N n_time_steps begin_step

%read membrane potentials
filename = fname;
filename = [input_dir, filename]

fprintf('n_time_steps = %d begin_step = %d\n',n_time_steps, begin_step);

if begin_step > n_time_steps
    begin_step = 1;
end
%v_array = zeros( n_time_steps - begin_step + 1, numel(neuron_ndx));
% numel(A) returns the number of elements in array or subscripted array 
% expression
v_array = [];

if exist(filename, 'file')
    fid = fopen(filename, 'r', 'native');
    num_params = fread(fid, 1, 'int');
    fprintf('num_params = %d: ',num_params);
    
    for i_param = 1 : num_params
        params = fread(fid, 1, 'int');
        fprintf(' %d ',params)
    end
    fprintf('\n');
    %pause
    
    fread(fid, N * ( begin_step - 1 ), 'float');
    
    
    i_step = begin_step;
    
    while (~feof(fid))
        
        [v_step, Ntmp] = fread(fid, N, 'float');
        
        if(~isempty(v_step))
            
            %v_array(i_step - begin_step + 1,:) = ...
            %    v_step( neuron_ndx );
            v_array = [v_array; v_step( neuron_ndx )' ];
            
        end
        
        if ~feof(fid)
            i_step = i_step + 1;
            %fprintf('%d\n',i_step);
        end
    end
    
    fclose(fid);
    %disp(['ave_rate = ', num2str(ave_rate)]);
else
    disp(['Skipping, could not open ', filename]);
end


