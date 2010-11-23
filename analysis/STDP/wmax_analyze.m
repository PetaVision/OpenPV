% analysis of adaptive wMax models

% Make the global parameters available at the command-line for convenience.
global NX NY 
global input_dir output_dir n_time_steps

input_dir = '/Users/manghel/Documents/workspace/soren/output/';
%input_dir = '/Users/manghel/Documents/STDP-sim/adaptiveWmax3/';
output_dir = '/Users/manghel/Documents/workspace/soren/output/';

num_layers = 4;
n_time_steps = 10000;

NX         = 16;
NY         = 16;% 8;               % use xScale and yScale to get layer values
averageR   = 2;

plot_wMax             = 1;
comp_scores           = 1;
plot_rate             = 0;
read_spikes           = 0;  



% layer 1 is the image layer
for layer = 2:num_layers; % layer 1 here is layer 0 in PV
% Read relevant file names and scale parameters
    [f_file, v_file, r_file, w_file, w_last, l_name, xScale, yScale] = wmax_globals( layer );
    
    %% Analyze rate and spikes
    if plot_rate & layer >= 4
        
        NXscaled = NX * xScale; % L1 size
        NYscaled = NY * yScale;
        
        disp('read time dependent rate')
        if 0
        R = wmax_readFile(r_file, l_name, xScale, yScale);
        size(R)
        avR = mean(R,1);
        end
        
        %% reshape and plot wMax field
        if 0
            figure('Name',['average R for ',l_name]);
            recon2D = reshape(R, [NXscaled, NYscaled] );
            %recon2D = rot90(recon2D);
            %recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            imagesc( recon2D' );  % plots recon2D as an image
            colorbar
            axis square
            axis off
        end
        

        if 0
        for n=1:size(V,2)
            plot(R(:,n),'-b');
            pause
        end
        end
        
        disp('read spikes');


        % Read spike events

        if read_spikes
            disp('read spikes')
            begin_step = 0;
            end_step = 2000;
            
            [spike_array, ave_rate] = ...
                stdp_readSparseSpikes(f_file, begin_step, end_step);
            disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate)]);
            tot_steps = size( spike_array, 1 );
            tot_neurons = size( spike_array, 2);
            fprintf('tot_steps = %d tot_neurons = %d\n',...
                tot_steps,tot_neurons);
        end
        % display average rate vs averate spike rate
        if 0
        avS = mean(spike_array,1);
        
        figure('Name','Average R vs Average Rate');
        plot(avS, avR,'ob');
        end
        
        % look at the distribution of R conditioned on a 
        % fixed number of spikes in this time interval
        if 0
        sumS = sum(spike_array,1);
        avR = mean(R,1);
        for k=1:9
            nbins = 20;
            fprintf('num spikes = %d\n',k);
            ind = find(sumS == k);
            hist(avR(ind),nbins);
            pause
            
        end
        end
        
        
        
    end % plot_rate
    
    %% Analysis of adaptive wMax thresholds
    if plot_wMax & layer >= 4
        
        NXscaled = NX * xScale; % L1 size
        NYscaled = NY * yScale;

        %fname = 'Wmax4_last.pvp';
        fname = 'L1_Wmax_last.pvp';
        wMax = wmax_readLastFile(fname, l_name, xScale, yScale);
        
        %% histogram plot
        figure('Name',['wMax Histogram for ',l_name]);
        hist(wMax,50);
        
        
        %% reshape and plot wMax field
        if 1
            figure('Name',['wMax for ',l_name]);
            recon2D = reshape(wMax, [NXscaled, NYscaled] );
            %recon2D = rot90(recon2D);
            %recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            imagesc( recon2D' );  % plots recon2D as an image
            colorbar
            axis square
            axis off
        end

        fname = 'L1_R_last.pvp';
        R = wmax_readLastFile(fname,l_name,xScale, yScale);

        %% histogram plot
        figure('Name',['R Histogram for ',l_name]);
        hist(R,50);
        
        %% reshape and plot wMax field
        if 1
            figure('Name',['R for ',l_name]);
            recon2D = reshape(R, [NXscaled, NYscaled] );
            %recon2D = rot90(recon2D);
            %recon2D = 1 - recon2D;
            %figure('Name','Rate Array ');
            imagesc( recon2D' );  % plots recon2D as an image
            colorbar
            axis square
            axis off
        end

        figure('Name','wMax and Rate');
        plot(wMax, averageR - R, 'ob');
        xlabel('wMax');
        ylabel('avRate - Rate');
        title('Rate vs wMax');

    end
    
    %% read weights and compute learning scores for each neuron in 
    % this layer
    if comp_scores == 1 & layer >= 4
        disp('compute the learning score for each neuron: ');
        [S, F] = wmax_compScores(w_last,xScale,yScale); 
        sym = {'or','ob','og','ok'};
        
        for f=1:numel(w_last)
            figure('Name','Scores and wMax');
            plot(wMax,S{f},sym{f});
            xlabel('wMax');
            ylabel('Score');
            title([l_name,' Score vs wMax']);
        end
        
        
        
    end
    
    %% use scores to color plot 
    
    
end % loop over layers