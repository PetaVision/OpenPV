

input_dir = '/Users/manghel/Documents/workspace/marian/output/';
global input_dir N n_time_steps begin_step NX NY 

n_time_steps = 5000; % the argument of -n; even when dt = 0.5 
patch_size = 9;  % nxp * nyp
write_step = 100; % set in input/params.stdp

%% simulation parameters
begin_step = 1;  % where we start the analysis
stim_begin = 1;  % generally not true, but I read spikes
                      % starting from begin_step
stim_end = 5000;
stim_length = stim_end - stim_begin + 1;
stim_begin = stim_begin - begin_step + 1;
stim_end = stim_end - begin_step + 1;
stim_steps = stim_begin : stim_end;

NX=32;
NY=32;
N=NX*NY;

%% analysis parameters
read_data = 1;
print_data = 0;
plot_data = 0;
num_layers = 2;% retina + V1

spike_analysis = 1;
read_spikes = 1;
simple_movie = 0;
plot_raster = 1;
plot_spike_activity=1;
        
% presynaptic neuron location

%% read probe file

x=9;
y=24;
patch_size = 9;

filename = ['r',num2str(x),'-',num2str(y),'.probe'];
filename = [input_dir, filename];


if read_data
    
    disp('read M,P,W data')
if exist(filename,'file')
    
    fid = fopen(filename, 'r');
 
    t = 0;
    while t <= 10000
        
        t=t+1;
        %fprintf('%d\n',t);
       
        C = textscan(fid,'%*s',1);

        %read M
        C = textscan(fid,'%*s',1);
        C = textscan(fid,'%f',patch_size);
        M(t,:)=C{1}';
        if print_data
        for i=1:patch_size
           fprintf('%f ', M(t,i));
        end
        fprintf('\n');
        end
        
        % read P
        C = textscan(fid,'%*s',1);
        C = textscan(fid,'%f',patch_size);
        P(t,:)=C{1}'; 
        if print_data
        for i=1:patch_size
           fprintf('%f ', P(t,i));
        end
        fprintf('\n');
        end
        
        % read W
        C = textscan(fid,'%*s',1);
        C = textscan(fid,'%f',patch_size);
        W(t,:)=C{1}';  
        if print_data
        for i=1:patch_size
           fprintf('%f ', W(t,i));
        end
        fprintf('\n');
        %pause
        end
        
        %eofstat = feof(fid);
%       fprintf('eofstat = %d\n', eofstat);
        if (feof(fid))
            fprintf('feof reached: t = %d\n',t);
            break;
        end
         
    
    end
    fclose(fid);
%  
else
    disp(['probe file could not be open ', filename]);
    
end

end % read_data


%% plot M,P, and W evolution

if plot_data
    disp('plot data')
    cmap = colormap(hsv(128));

    figure('Name','M values')
    for i=1:patch_size
        plot(M(:,i),'-','Color',cmap(i*10,:));hold on
        %pause
    end

    figure('Name','P values')
    %for i=1:patch_size  % should all be the same
    for i=1:1
        plot(P(:,i),'-','Color',cmap(i*10,:));hold on
        %pause
    end

    figure('Name','W values')
    for i=1:patch_size
        plot(W(:,i),'-','Color',cmap(i*10,:));hold on
        pause
    end
end

%% read activity (spiking) information

if spike_analysis
    
for layer = 1:num_layers

    % Read parameters from file which pv created: LAYER
    [f_file, v_file, w_file] = stdp_globals( layer-1 );

    % Read spike events
    
    
    if read_spikes
        disp('read spikes')
        [spike_array{layer}, ave_rate] = stdp_readSparseSpikes(f_file);
        disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate)]);
        tot_steps = size( spike_array{layer}, 1 );
           
    end
    
    
    if simple_movie
        disp('simple movie')
        for t=1:tot_steps
            if max(spike_array{layer}(t,:)) > 0
                fprintf('%d\n',t);
                A = reshape(spike_array{layer}(t,:),32,32);
                imagesc(A')
                pause(0.1)
            end
        end
        disp('pause')
        pause
    end
    
    
    % raster plot
    
    if plot_raster
        disp('plot_raster')
        if ~isempty(spike_array{layer})
            plot_title = ['Raster for layer = ',int2str(layer)];
            figure('Name',plot_title);
            axis([0 tot_steps 0 N]);
            hold on
            box on
            
            [spike_time, spike_id] = find((spike_array{layer}));
            lh = plot(spike_time, spike_id, '.g');
            %set(lh,'Color',my_gray);    

        end
        pause
    end
    
    
    if plot_spike_activity 
        disp('compute rate array and spike activity array')
        rate_array{layer} = 1000 * full( mean(spike_array{layer}(stim_steps,:),1) );
        % this is a 1xN array where N=NX*NY
        disp('plot_rate_reconstruction')
        stdp_reconstruct(rate_array{layer}, ['Rate reconstruction for layer  ', int2str(layer)]);
        pause        
        
        
        % plot spikes for selected indices
        % stdp_plotSpikes(spike_array{layer},[]);
        
    end
end

end % spike_analysis
