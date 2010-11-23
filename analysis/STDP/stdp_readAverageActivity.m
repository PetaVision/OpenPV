function [average_array, ave_rate] = stdp_readAverageActivity(fname, begin_step, end_step)

global input_dir n_time_steps 

filename = fname;
filename = [input_dir, filename];
fprintf('read spikes from %s\n',filename);
fprintf('begin_step = %d end_step = %d\n',begin_step, end_step);
%pause

debug = 1;  % if 1 prints spiking neurons

ave_rate = 0;

if exist(filename,'file')
    total_spikes = 0;
    fid = fopen(filename, 'r', 'native');
    [time,numParams,NX,NY,NF] = readHeader(fid);
    if debug
        fprintf('NX = %d NY = %d NF = %d \n',NX,NY,NF);
        pause
    end
    
    N = NX * NY * NF;
    minInd = N+1;
    maxInd = -1;
   
    average_array = zeros(1,N);
    
    
    for i_step = 1 : n_time_steps
        
        
        if (feof(fid))
            n_time_steps = i_step - 1;
            eofstat = feof(fid);
            fprintf('feof reached: n_time_steps = %d eof = %d\n',...
                n_time_steps,eofstat);
            break;
        else
            time = fread(fid,1,'float64');
            %fprintf('time = %f\n',time);
            num_spikes = fread(fid, 1, 'int');
            eofstat = feof(fid);
            %fprintf('eofstat = %d\n', eofstat);
        end
        
        S =fread(fid, num_spikes, 'int'); % S is a column vector
        
        if debug 
            fprintf('%d: %f number of spikes = %d: ', ...
                i_step, time, num_spikes);
            for i=1:length(S)
                fprintf('%d ',S(i));
            end
            fprintf('\n');
            pause
        else
            %fprintf('%d: %f number of spikes = %d eof = %d\n', ...
            %    i_step, time, num_spikes,eofstat);
        end
        
        maxInd = max([maxInd S']);
        minInd = min([minInd S']);
        
        
        if i_step < begin_step
            continue
        elseif (i_step > end_step)
            break
        end
        
        % we are between begin and end_step
        % update activity!
        %fprintf('%d %f\n',i_step, time);
        average_array(S+1) = average_array(S+1) + 1;
        total_spikes = total_spikes + num_spikes;
        
        %pause
        
    end % loop over sim steps
    fclose(fid);
    ave_rate = 2 * 1000 * total_spikes / ( N * ( end_step - begin_step + 1 ) );
    average_array =  average_array * ...
        ( 2 * 1000.0 / ( end_step - begin_step + 1 ) );
    
    fprintf('i_step = %d minInd = %d maxInd = %d aveRate = %f\n',...
        i_step,minInd,maxInd,ave_rate);
    
else
    disp(['Skipping, could not open ', filename]);
    average_array = [];
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
    %fprintf('numParams = %d ',numParams);
    %fprintf('NX = %d NY = %d NF = %d ',NX,NY,NF);
    %pause
    % read time - last two params
    time = fread(fid,1,'float64');
    %fprintf('time = %f\n',time);
    %pause
    
% End subfunction 
%    
