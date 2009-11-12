%%
close all
%clear all

% Make the global parameters available at the command-line for convenience.
global N  NX NY n_time_steps begin_step tot_steps
global spike_array num_target rate_array target_ndx vmem_array
global input_dir output_path input_path
global patch_size write_step

%input_dir = '/Users/manghel/Documents/workspace/marian/output/';
input_dir = '/Users/manghel/Documents/workspace/STDP/output/';

num_layers = 1;
n_time_steps = 1000; % the argument of -n; even when dt = 0.5 
patch_size = 16;  % nxp * nyp
write_step = 100; % set in writePostPatch() in HyPerConn.cpp


begin_step = 1;  % where we start the analysis
stim_begin = 1;  % generally not true, but I read spikes
                      % starting from begin_step
stim_end = 10000;
stim_length = stim_end - stim_begin + 1;
stim_begin = stim_begin - begin_step + 1;
stim_end = stim_end - begin_step + 1;
stim_steps = stim_begin : stim_end;
bin_size = 10;

NX=32;
NY=32;

my_gray = [.666 .666 .666];

target_ndx = cell(num_layers,1);
bkgrnd_ndx = cell(num_layers,1);

parse_tiff = 1;
read_spikes = 0;
simple_movie = 0;
plot_raster = 0;
plot_spike_activity = 0;
plot_weights_rate_evolution = 0;
plot_membrane_potential = 0;
plot_weights_field = 1;
plot_weights_histogram = 0;
plot_patch = 0;

if read_spikes
    spike_array = cell(num_layers,1);
    spike_array_bkgrnd = spike_array;
end

if parse_tiff
    targ = [];
   tiff_path = [input_dir 'images/img2D_0.00.tif'];
   %imshow(tiff_path);
   [targ, Xtarg, Ytarg] = stdp_parseTiff( tiff_path );
   disp('parse tiff -> done');
   %pause
end


for layer = 1:num_layers;

    % Read parameters from file which pv created: LAYER
    [f_file, v_file, w_file] = stdp_globals( layer );

    % Read spike events
    
    
    if read_spikes
        disp('read spikes')
        [spike_array{layer}, ave_rate] = stdp_readSparseSpikes(f_file);
        disp(['ave_rate(',num2str(layer),') = ', num2str(ave_rate)]);
        tot_steps = size( spike_array{layer}, 1 );
        num_bins = fix( tot_steps / bin_size );
        excess_steps = tot_steps - num_bins * bin_size;
        stim_bin_length = fix( ( stim_end - stim_begin + 1 ) / bin_size );
        stim_bin_begin = fix( ( stim_begin - begin_step + 1 ) / bin_size );
        stim_bin_end = fix( ( stim_end - begin_step + 1 ) / bin_size );
        stim_bins = stim_bin_begin : stim_bin_end;     
    end
    
    
    if simple_movie
        disp('simple movie')
        for t=1:n_time_steps
            fprintf('%d\n',t);
            A = reshape(spike_array{layer}(t,:),NX,NY);
            imagesc(A')
            pause(0.1)
        end
        disp('pause')
        pause
    end
    
    
    
    %parse the object identification files
    % pv_objects
    if parse_tiff
        % TODO: support for multiple objects--multiple calls to parseTiff?
        target_ndx{layer} = targ;
        bkgrnd_ndx{layer} = find(ones(N,1));
        bkgrnd_ndx{layer}(target_ndx{layer}) = 0;
        bkgrnd_ndx{layer} = find(bkgrnd_ndx{layer});
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

        spikes_array{layer} = 1000 * full( mean(spike_array{layer}(stim_steps,:),2) );
        % this is Tx1 array
        plot_title = ['Spiking activity for layer  ',int2str(layer)];
        figure('Name',plot_title);
        plot(spikes_array{layer},'ob')
        xlabel=('t');
        ylabel=('num spikes');
        pause
        
        % redo this computation
        tot_steps = size( spike_array{layer}, 1 );
        num_bins = fix( tot_steps / bin_size );
        plot_title = ['Moving window rate average for layer  ',int2str(layer)];
        figure('Name',plot_title);
        bin_size = 100;
        tot_steps = size( spike_array{layer}, 1 );
        num_bins = fix( tot_steps / bin_size );
        moving_rate_array{layer} = ...
            mean(reshape(spikes_array{layer}, bin_size, num_bins), 1);
        % reshape returns a bin_size x num_bins array: it oputs the first 
        % bin_size values in column 1, the next bin_size values in column 2, 
        % and so on, while mean applied to this matrix returns a
        % 1 x num_bins array
        plot(moving_rate_array{layer},'or')
        pause
        
        % plot spikes for selected indices
        % stdp_plotSpikes(spike_array{layer},[]);
        
    end
    
    
    if plot_weights_rate_evolution
        disp('plot weights rate evolution');
        % this is a cell aray: for each neuron it returns a temporal
        % array for the sum of its synaptic weights, i.e. symW{n} is 
        % a T x 1 array where T is the number of weights snapshots
        [sumW, T] = stdp_compAverageWeightsEvol(w_file);
        % pass T-1
        [W,R] = stdp_analyzeWeightsRate(sumW, T-1, spike_array{layer});
        
        
        
    end

    
    % plot maximum stimulated membrane potential
    
    %read membrane potentials
    if plot_membrane_potential
        disp('plot_membrane_potential')
        vmem_array = stdp_readV(v_file, target_ndx{layer});
        num_max = 7;
        if ~isempty(spike_array{layer})
            plot_title = ['Membrane potential for layer = ',int2str(layer)];
            figure('Name',plot_title);
            color_order = get(gca,'ColorOrder');
            if ~isempty(spike_array{layer})
                % extract the mean rate of target neurons
                rate_array2 = rate_array{layer}( target_ndx{layer} );
                % sort in decreasing rate order
                [sorted_rate, sorted_ndx] = sort(rate_array2, 2, 'descend');
                spike_array2 = spike_array{layer}( :, target_ndx{layer} );
                vmem_rate = zeros(num_max,1);
                for i_max = 1:num_max
                    vmem_color = color_order(i_max,:);
                    lh = plot(vmem_array(:,sorted_ndx(i_max)), '-k');
                    set(lh, 'Color', vmem_color);
                    set(lh, 'LineWidth', 2.0);
                    hold on;
                    [spike_times, spike_id, spike_vals] = ...
                        find(spike_array2(:,sorted_ndx(i_max)));
                    vmem_rate(i_max) = length(spike_times);
                    lh = line( [max(1,spike_times-1), max(1,spike_times-1)]', ...
                        [vmem_array(max(1,spike_times-1), sorted_ndx(i_max)), 0.5*spike_vals]' );
                    set(lh, 'Color', vmem_color);
                    set(lh, 'LineWidth', 2.0);
                    display( [ 'rank:', num2str(i_max), ' vmem_rate(', num2str(layer), ') = ', ...
                        num2str( 1000*vmem_rate(i_max) / tot_steps ) ] );
                end
            end
        end
        clear col_ndx row_ndx Vmem_ndx vmem_array spike_array2
        pause
    end %plot_membrane_potential

    % Analyze the weights field and its distribution
    % Analyze the weights evoltion
    % NOTE: The number of weights distribution recorded is
    % (n_time_steps/write_step) 
    if plot_weights_field == 1
        disp('plot weights field')
        if isempty(Xtarg) 
            disp('No target image: random retina noise only');
        end
        stdp_plotWeightsOnly(w_file,Xtarg,Ytarg);
        
        % this is 
        %pv_reconstruct( vmem_array(:), ['Weights for layer = ', int2str(layer)] );
        %clear Vmem_ndx vmem_array weight3D
        %pause
    end
    
    if plot_weights_histogram == 1
        disp('plot weights histogram')
        nbins = 200;
        TSTEP = 10;
        W = stdp_plotWeightsHistogramOnly(w_file,nbins,TSTEP);% W is t
        %pause
    end
    
    
    if plot_patch % define the symbols using xtarg and ytarg so that
                  % it adapts to what a patch sees!!!
        patch_indexes = [];
        NXP=3;
        NYP=3;
        a = (NXP-1)/2;
        b = (NYP-1)/2;
             
        d='y';
        while d == 'y'
        %for p=1:length(Xtarg)
            I = input('input I index: ');
            J = input('input J index: ');
            %I=Xtarg(p);
            %J=Ytarg(p);
            fprintf('patch evolution for I= %d J= %d \n',I,J);
            linear_patch_index = [];
            k=0; % linear patch index 
            for j=1:NYP
                Jpatch = J-b + (j-1); % global XY image index
                %fprintf('\t');
                for i=1:NXP
                    k=k+1;
                    Ipatch = I-a + (i-1); % global XY image index
                    linear_patch_index(k) = (Ipatch-1)*NY + Jpatch;
                                      % global linear index
                    %fprintf('%d ',   linear_patch_index(k));
                    %fprintf('(%d %d) ',Jpatch,Ipatch)
                end
                %fprintf('\n');
            end
            plot_symbol = ismember(linear_patch_index, targ)
            %pause
            sym={'o-g','o-r'};
            plot_title = ['Weights evolution for neuron (' num2str(I) ',' num2str(J) ')'];
            PATCH = stdp_plotPatch(w_file, I,J, plot_title );
            figure('Name', plot_title);
            for k=1:9,
                %plot(PATCH(:,k),sym{k}),hold on,
                plot(PATCH(:,k),sym{plot_symbol(k)+1}),hold on,
            end
            AVERAGE_PATCH = reshape(mean(PATCH,1),[NXP NYP]);
            figure('Name', ['Average weights for neuron (' num2str(I) ',' num2str(J) ')']);
            imagesc(AVERAGE_PATCH', 'CDataMapping','direct');
            % NOTE: It seems that I need to transpose PATCH_RATE
            colorbar
        
            d = input('more patches? (y/n): ','s');
            if isempty(d)
                d='y';
            end
            
        end
    end
    
end % loop over all layers

plot_rates = tot_steps > 9 ;
if plot_rates
    plot_title = ['PSTH target pixels'];
    figure('Name',plot_title);
    hold on
    co = get(gca,'ColorOrder');
    lh = zeros(4,1);
    for layer = 1:num_layers
        lh(layer) = plot((1:num_bins)*bin_size, psth_target{layer,i_target}, '-r');
        set(lh(layer),'Color',co(layer,:));
        set(lh(layer),'LineWidth',2);
    end
    leg_h = legend(lh(1:num_layers), ...
        [ ...
        'Retina  '; ...
        'V1      '; ...
        'V1 Inh  ' ...
        ]);
end

%%
plot_movie = 0; %tot_steps > 9;
if plot_movie
    for layer = [1 3 5 6]
        spike_movie = pv_reconstruct_movie( spike_array, layer);
        movie2avi(spike_movie,['layer',num2str(layer),'.avi'], ...
            'compression', 'None');
        show_movie = 1;
        if show_movie
            scrsz = get(0,'ScreenSize');
            [h, w, p] = size(spike_movie(1).cdata);  % use 1st frame to get dimensions
            fh = figure('Position',[scrsz(3)/4 scrsz(4)/4 560 420]);
            axis off
            box off
            axis square;
            axis tight;
            movie(fh,spike_movie,[1, 256+[1:256]], 12,[0 0 560 420]);
            close(fh);
        end
    end % layer
    clear spike_movie
end % plot_movie



%%

plot_xcorr = 0 && tot_steps > 9;
if plot_xcorr
    xcorr_eigenvector = cell( num_layers, 2);
    for layer = 1:num_layers
        pv_globals(layer);
%         pack;

        % autocorrelation of psth
        plot_title = ['Auto Correlations for layer = ',int2str(layer)];
        figure('Name',plot_title);
        ave_target_tmp = full(ave_target{layer,i_target}(stim_steps));
        maxlag= fix(length(ave_target_tmp)/4);
        target_xcorr = ...
            xcorr( ave_target_tmp, maxlag, 'unbiased' );
        target_xcorr = ( target_xcorr - mean(ave_target_tmp)^2 ) / ...
            (mean(ave_target_tmp)+(mean(ave_target_tmp)==0))^2;
        lh_target = plot((-maxlag:maxlag), target_xcorr, '-r');
        set(lh_target, 'LineWidth', 2);
        hold on
        ave_clutter_tmp = full(ave_clutter{layer,1}(stim_steps));
        clutter_xcorr = ...
            xcorr( ave_clutter_tmp, maxlag, 'unbiased' );
        clutter_xcorr = ( clutter_xcorr - mean(ave_clutter_tmp)^2 ) / ...
            (mean(ave_clutter_tmp)+(mean(ave_clutter_tmp)==0))^2;
        lh_clutter = plot((-maxlag:maxlag), clutter_xcorr, '-b');
        set(lh_clutter, 'LineWidth', 2);
        ave_bkgrnd_tmp = full(ave_bkgrnd{layer,1}(stim_steps));
        bkgrnd_xcorr = ...
            xcorr( ave_bkgrnd_tmp, maxlag, 'unbiased' );
        bkgrnd_xcorr = ( bkgrnd_xcorr - mean(ave_bkgrnd_tmp)^2 ) / ...
            (mean(ave_bkgrnd_tmp)+(mean(ave_bkgrnd_tmp)==0))^2;
        lh_bkgrnd = plot((-maxlag:maxlag), bkgrnd_xcorr, '-k');
        set(lh_bkgrnd, 'Color', my_gray);
        set(lh_bkgrnd, 'LineWidth', 2);
        target2clutter_xcorr = ...
            xcorr( ave_target_tmp, ave_clutter_tmp, maxlag, 'unbiased' );
        target2clutter_xcorr = ...
            ( target2clutter_xcorr - mean(ave_target_tmp) * mean(ave_clutter_tmp) ) / ...
            ( (mean(ave_target_tmp)+(mean(ave_target_tmp)==0)) * ...
            (mean(ave_clutter_tmp)+(mean(ave_clutter_tmp)==0)) );
        lh_target2clutter = plot((-maxlag:maxlag), target2clutter_xcorr, '-g');
        set(lh_target2clutter, 'LineWidth', 2);
        axis tight
        
        %plot power spectrum
        plot_title = ['Power for layer = ',int2str(layer)];
        figure('Name',plot_title);
        freq = 1000*(0:2*maxlag)/(2*maxlag + 1);
        min_ndx = find(freq > 400, 1,'first');
        target_fft = fft(target_xcorr);
        lh_target = plot(freq(2:min_ndx),...
            abs(target_fft(2:min_ndx)), '-r');
        set(lh_target, 'LineWidth', 2);
        hold on
        clutter_fft = fft(clutter_xcorr);
        lh_clutter = plot(freq(2:min_ndx),...
            abs(clutter_fft(2:min_ndx)), '-b');
        set(lh_clutter, 'LineWidth', 2);
        bkgrnd_fft = fft(bkgrnd_xcorr);
        lh_bkgrnd = plot(freq(2:min_ndx),...
            abs(bkgrnd_fft(2:min_ndx)), '-k');
        set(lh_bkgrnd, 'LineWidth', 2);
        set(lh_bkgrnd, 'Color', my_gray);
        target2clutter_fft = fft(target2clutter_xcorr);
        lh_target2clutter = plot(freq(2:min_ndx),...
            abs(target2clutter_fft(2:min_ndx)), '-g');
        set(lh_target2clutter, 'LineWidth', 2);
        axis tight

        %plot power reconstruction
        num_rate_sig = -1.0;
        mean_rate = mean( rate_array{layer} );
        std_rate = std( rate_array{layer} );
        rate_mask = find( rate_array{layer} > ( mean_rate + num_rate_sig * std_rate ) );
        num_rate_mask = numel(rate_mask);
        disp( [ 'mean_rate(', num2str(layer), ') = ', ...
            num2str(mean_rate), ' +/- ', num2str(std_rate) ] );
        disp( ['num_rate_mask(', num2str(layer), ') = ', num2str(num_rate_mask), ' > ', ...
            num2str( mean_rate + num_rate_sig * std_rate ) ] );
        while num_rate_mask > 2^14
            num_rate_sig = num_rate_sig + 0.5;
            rate_mask = find( rate_array{layer} > ( mean_rate + num_rate_sig * std_rate ) );
            num_rate_mask = numel(rate_mask);
            disp( ['num_rate_mask(', num2str(layer), ') = ', num2str(num_rate_mask), ' > ', ...
                num2str( mean_rate + num_rate_sig * std_rate ) ] );
        end
%         if num_rate_mask < 64
%             break;
%         end
        spike_full = full( spike_array{layer}(:, rate_mask) );
        freq_vals = 1000*(0:tot_steps-1)/tot_steps;
        min_ndx = find(freq_vals >= 40, 1,'first');
        max_ndx = find(freq_vals <= 60, 1,'last');
        power_array = fft( spike_full );
        power_array = power_array .* conj( power_array );
        peak_power = max(power_array(min_ndx:max_ndx,:));
        peak_power = sparse( rate_mask, 1, peak_power, N, 1 );
        pv_reconstruct(full(peak_power),  ['Power reconstruction for layer = ', int2str(layer)]);
        clear ave_target_tmp target_xcorr
        clear ave_clutter_tmp clutter_xcorr
        clear ave_bkgrnd_tmp bkgrnd_xcorr
        clear target2clutter_xcorr
        clear target_fft clutter_fft
        clear power_array spike_full bkgrnd_fft target2clutter_fft

        %cross-correlation using spike_array
        plot_pairwise_xcorr = 0;
        if plot_pairwise_xcorr
            plot_title = ['Cross Correlations for layer = ',int2str(layer)];
            figure('Name',plot_title);
            maxlag = fix( stim_length / 4 );
            freq = 1000*(0:2*maxlag)/(2*maxlag + 1);
            min_ndx = find(freq_vals >= 20, 1,'first');
            max_ndx = find(freq_vals <= 60, 1,'last');
            disp(['computing pv_xcorr_ave(', num2str(layer),')']);
            xcorr_ave_target = pv_xcorr_ave( spike_array{layer}(stim_steps, target_ndx{layer, i_target}), maxlag );
            lh_target = plot((-maxlag:maxlag), xcorr_ave_target, '-r');
            set(lh_target, 'LineWidth', 2);
            hold on
            xcorr_ave_clutter = pv_xcorr_ave( spike_array{layer}(stim_steps, clutter_ndx{layer}), maxlag );
            lh_clutter = plot((-maxlag:maxlag), xcorr_ave_clutter, '-b');
            set(lh_clutter, 'LineWidth', 2);
            hold on
            xcorr_ave_target2clutter = ...
                pv_xcorr_ave( spike_array{layer}(stim_steps, target_ndx{layer, i_target}), maxlag, ...
                spike_array{layer}(stim_steps, clutter_ndx{layer}) );
            lh_target2clutter = plot((-maxlag:maxlag), xcorr_ave_target2clutter, '-g');
            set(lh_target2clutter, 'LineWidth', 2);
            hold on
        end

        %make eigen amoebas
        plot_eigen = layer == -1 || layer == -3 || layer == -5 || layer == -6;
        if plot_eigen
            calc_power_mask = 1;
            if calc_power_mask
                num_power_sig = -1.0;
                mean_power = mean( peak_power(rate_mask) );
                std_power = std( peak_power(rate_mask) );
                disp( [ 'mean_power(', num2str(layer), ') = ', ...
                    num2str(mean_power), ' +/- ', num2str(std_power) ] );
                power_mask = find( peak_power > ( mean_power + num_power_sig * std_power ) );
                num_power_mask = numel(power_mask);
                disp( ['num_power_mask(', num2str(layer), ') = ', num2str(num_power_mask), ' > ', ...
                    num2str( mean_power + num_power_sig * std_power ) ] );
                while num_power_mask > 2^12
                    num_power_sig = num_power_sig + 0.5;
                    power_mask = find( peak_power > ( mean_power + num_power_sig * std_power ) );
                    num_power_mask = numel(power_mask);
                    disp( ['num_power_mask(', num2str(layer), ') = ', num2str(num_power_mask), ' > ', ...
                        num2str( mean_power + num_power_sig * std_power ) ] );
                end
%                 if num_power_mask < 64
%                     break;
%                 end
            else
                power_mask = sort( [target_ndx{layer, 1}, clutter_ndx{layer,1}] );
                num_power_mask = numel(power_mask);
            end % calc_power_mask
            disp(['computing xcorr(', num2str(layer),')']);
            %             pack;
            maxlag = fix( stim_length / 4 );
            freq_vals = 1000*(0:2*maxlag)/(2*maxlag + 1);
            min_ndx = find(freq_vals >= 20, 1,'first');
            max_ndx = max(find(freq_vals <= 60, 1,'last'), min_ndx);
            sparse_corr = ....
                pv_xcorr( spike_array{layer}(stim_steps, power_mask), ...
                spike_array{layer}(stim_steps, power_mask), ...
                maxlag, min_ndx,  max_ndx);
            for sync_true = 1 : 1 + (maxlag > 1)

                % find eigen vectors
                disp(['computing eigenvectors(', num2str(layer),')']);
                options.issym = 1;
                num_eigen = 6;
                [eigen_vec, eigen_value, eigen_flag] = ...
                    eigs( (1/2)*(sparse_corr{sync_true} + sparse_corr{sync_true}'), num_eigen, 'lm', options);
                for i_vec = 1:num_eigen
                    disp(['eigenvalues(', num2str(layer), ',' ...
                        , num2str(i_vec),') = ', num2str(eigen_value(i_vec,i_vec))]);
                end
                [max_eigen, max_eigen_ndx] = max( diag( eigen_value ) );
                xcorr_eigenvector{layer, sync_true} = eigen_vec(:,max_eigen_ndx);
                for i_vec = 1:num_eigen
                    plot_title = ['Eigen Reconstruction(', num2str(layer), '): sync_true = ', ...
                        num2str(2 - sync_true), ', i_vec = ', num2str(i_vec)];
                    %                     if mean(eigen_vec(power_mask,i_vec)) < 0
                    %                         eigen_vec(power_mask,i_vec) = -eigen_vec(power_mask,i_vec);
                    %                     end
                    eigen_vec_tmp = ...
                        sparse( power_mask, 1, eigen_vec(:,i_vec), ...
                        N, 1, num_power_mask);
                    pv_reconstruct(eigen_vec_tmp, plot_title);
                end % i_vec
            end % for sync_flag = 0:1
        end % plot_eigen
    end % layer
end % if plot_xcorr

