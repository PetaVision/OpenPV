function [sumW, T] = stdp_compAverageWeightsEvol(fname)
% for each neuron computes the sum of its synaptic (patch) weights vs time
% fname is the binary post-synaptic weights file w$_Post.bin where
% $ is the connection number.

global input_dir N NX NY patch_size write_step

% can read patch_size from the header

filename = fname;
filename = [input_dir, filename];

%W_array = zeros( (n_time_steps - begin_step + 1)/write_step, N*patch_size );
if exist(filename, 'file')
    % get the number of recorded weights snap shots
    fid = fopen(filename, 'r', 'native');
    num_params = fread(fid, 1, 'int')
    NXP = fread(fid, 1, 'int');
    NYP = fread(fid, 1, 'int');
    NFP = fread(fid, 1, 'int');
    minVal = fread(fid, 1, 'int');
    maxVal = fread(fid, 1, 'int');
    numPatches = fread(fid, 1, 'int');
    fprintf('num_params = %d NXP = %d NYP = %d NFP = %d ',...
        num_params,NXP,NYP,NFP);
    fprintf('minVal = %f maxVal = %d numPatches = %d\n',...
        minVal,maxVal,numPatches);
    %pause
    
    if patch_size ~= NXP*NYP
        disp('wring patch parameters')
        return
    end
    
    T = 0; % time step index
    
    while (~feof(fid))
        T=T+1;
        for j=1:NY
            for i=1:NX
                nxp = fread(fid, 1, 'uint16'); % unsigned short
                nyp = fread(fid, 1, 'uint16'); % unsigned short
                w = fread(fid, patch_size+3, 'uchar'); % unsigned char
                %w = fread(fid, patch_size, 'uchar');
            end
        end
    end
    
    fclose(fid);
    fprintf('feof reached: the number of weights snapshots is %d\n',T);
    
    % read and populate the weights array
    fid = fopen(filename, 'r', 'native');
    num_params = fread(fid, 1, 'int');
    for i_params = 1 : num_params     
        tmp = fread(fid, 1, 'int');
    end
    
    
    sumW = zeros(T, NX*NY); 
    
    
    T = 0; % time step index
    while (~feof(fid))
        T=T+1;
        for j=1:NY
            for i=1:NX
                n = (j-1)*NX + i; % neuron index
                nxp = fread(fid, 1, 'uint16'); % unsigned short
                nyp = fread(fid, 1, 'uint16'); % unsigned short
                w = fread(fid, patch_size+3, 'uchar'); % unsigned char
                %w = fread(fid, patch_size, 'uchar');
                % scale weights: they are quantized before are written
                w = minVal + (maxVal - minVal) * ( (w * 1.0)/ 255.0);
                if(~isempty(w))
                    sumW(T,n) = sum(w(1:patch_size));
                end
            end
        end
    end
    fclose(fid);
    fprintf('feof reached: the number of weights snapshots is %d\n',T);
    
    figure('Name','Weights sum evolution')
    for n= (3*NX + 10):(3*NX+20)
        plot(sumW(:,n),'-k');hold on
        pause
    end
    
else
    disp(['Skipping, could not open ', filename]);
end


