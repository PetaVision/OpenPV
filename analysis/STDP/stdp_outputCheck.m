function stdp_outputCheck(fname, begin_step, end_step)

global input_dir n_time_steps dT

filename = fname;
filename = [input_dir, filename];
fprintf('read spikes from %s\n',filename);

debug = 0;  % if 1 prints spiking neurons

ave_rate = 0;
if begin_step > n_time_steps
    begin_step = 1;
end

if exist(filename,'file')
    total_spikes = 0;
    fid = fopen(filename, 'r', 'native');
    [time,numParams,NX,NY,NF] = readHeader(fid);
    fprintf('NX = %d NY = %d NF = %d \n',NX,NY,NF);
    %pause
    
    N = NX * NY * NF;
    minInd = N+1;
    maxInd = -1;
   
    spike_array = zeros(N,1);
    spike_fig = figure('Name','Spike Activity');
    frame_fig = figure('Name','Movie Frame');
    
    for i_step = 1 : n_time_steps
        
        eofstat = feof(fid);
        if (eofstat)
            n_time_steps = i_step - 1;
            fprintf('feof reached: n_time_steps = %d eof = %d\n',...
                n_time_steps,eofstat);
            break;
        else
            eofstat = feof(fid);
            time = fread(fid,1,'float64');
            if isempty(time)
                break
            else
                %fprintf('time = %f\n',time);
            end
            %fprintf('time = %f\n',time);
            num_spikes = fread(fid, 1, 'int');
            %fprintf('time = %f num_spikes = %d ',time, num_spikes);
            %fprintf('eofstat = %d\n', eofstat);
            %pause
        end
        
        S =fread(fid, num_spikes, 'int'); % S is a column vector
        
        if debug 
            fprintf('%d: %f number of spikes = %d: ', ...
                i_step, time, num_spikes);
            for i=1:length(S)
                fprintf('%d ',S(i));
            end
            fprintf('\n');
            %pause
        else
            %fprintf('%d: %f number of spikes = %d eof = %d\n', ...
            %    i_step, time, num_spikes,eofstat);
        end
        
        maxInd = max([maxInd S']);
        minInd = min([minInd S']);
        
        if i_step < begin_step
            continue
        elseif i_step > end_step
             break
        else
            spike_array(:) = 0;
            spike_array(S+1) = 1;
            
            % plot "spike" activity
            figure(spike_fig);
            A = reshape(spike_array, NX, NY);
            imagesc(A')
            
            % load movie frame and plot it
            T = (i_step-1)* dT;
            moviefile = [input_dir 'Movie_' num2str(T,'%10.6f') '.tif'];
            fprintf('%s\n', ['Movie_' num2str(T,'%10.6f') '.tif'] ); 
            figure(frame_fig);
            imshow(moviefile);
            
            pause
        end
        
        
    end
    fclose(fid);
    
else
    disp(['Skipping, could not open ', filename]);
    spikes = sparse([], [], [], end_step - begin_step + 1, N, 0);
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
