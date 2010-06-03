%%
close all
clear all

% set paths, may not be applicable to all octave installations
pvp_matlabPath;

if ( uioctave )
    setenv('GNUTERM', 'x11');
end%%if

% Make the following global parameters available to all functions for convenience.
global N_image NROWS_image NCOLS_image
global N NROWS NCOLS % for the current layer
global NFEATURES  % for the current layer
global NO NK dK % for the current layer
global n_time_steps begin_step end_step time_steps tot_steps
global stim_begin_step stim_end_step stim_steps
global stim_begin_bin stim_end_bin stim_bins
global analysis_start_time analysis_stop_time
global bin_size dt
global begin_time end_time
global stim_begin_time stim_end_time
global num_targets
global rate_array
global output_path spike_path
global NUM_BIN_PARAMS
global NUM_WGT_PARAMS
NUM_BIN_PARAMS = 20;
NUM_WGT_PARAMS = 6;

global FLAT_ARCH_FLAG
FLAT_ARCH_FLAG = 1;

spike_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/test_amoeba_target_4fc/';
output_path = spike_path;

input_path = '/Users/gkenyon/Documents/eclipse-workspace/PetaVision/mlab/amoebaGen/128_png/4/';

pvp_order = 1;

% begin step, end_step and stim_begin_step may be adusted by
% pvp_readSparseSpikes
begin_time = 0.0;  % (msec) start analysis here, used to exclude start up artifacts
end_time = inf;
stim_begin_time = 1.0;  % times (msec) before this and after begin_time can be used to calculate background
stim_end_time = 1000.0;
stim_exclude_time = 50.0;
bin_size = 4.0;  % (msec) used for all rate calculations
analysis_start_time = 1.0;
analysis_stop_time = 1000.0;
dt = 1.0; % msec

% initialize to size of image (if known), these should be overwritten by each layer
NROWS_image=128;
NCOLS_image=128;
NROWS = NROWS_image;
NCOLS = NCOLS_image;
NFEATURES = 12;

NO = NFEATURES; % number of orientations
NK = 1; % number of curvatures
dK = 0; % spacing between curvatures (1/radius)

my_gray = [.666 .666 .666];
num_targets = 1;
fig_list = [];

global N_LAYERS
global SPIKING_FLAG
SPIKING_FLAG = 1;
[layerID, layerIndex] = pvp_layerID();

read_spikes = 1:N_LAYERS;  % list of spiking layers whose spike train are to be analyzed
num_layers = N_LAYERS;

min_plot_steps = 20;  % time-dependent quantities only plotted if tot_steps exceeds this threshold
plot_reconstruct = read_spikes; %uimatlab;
plot_raster = read_spikes; %uimatlab;
plot_reconstruct_target = 0;
plot_vmem = 0;
plot_autocorr = [];
plot_xcorr = [];

spike_array = cell(num_layers,1);
ave_rate = zeros(num_layers,1);

%% read input image
global image_path target_path
image_filename = [input_path 't/tar_0000_a.png'];
target_filename{1} = [input_path 'a/tar_0000_a.png'];

invert_image_flag = 0;
[target, clutter, image, fig_tmp] = ...
    pvp_parseTarget( image_filename, target_filename, invert_image_flag, 1);
fig_list = [fig_list; fig_tmp];
disp('parse BMP -> done');

% images may be annotated to indicate target and clutter pixels
ave_target = cell(num_layers,num_targets);
ave_clutter = cell(num_layers,1);
ave_bkgrnd = cell(num_layers,1);
psth_target = cell(num_layers,num_targets);
psth_clutter = cell(num_layers,1);
psth_bkgrnd = cell(num_layers,1);
target_ndx = cell(num_layers, num_targets);
target_ndx_max = cell(num_layers, num_targets);
clutter_ndx = cell(num_layers, num_targets);
clutter_ndx_max = cell(num_layers, num_targets);
bkgrnd_ndx = cell(num_layers, num_targets);
num_target_neurons = zeros(num_layers, num_targets);
num_clutter_neurons = zeros(num_layers, 1);
num_bkgrnd_neurons = zeros(num_layers, 1);
num_rows = ones(num_layers,1);
num_cols = ones(num_layers,1);
num_features = ones(num_layers,1);
num_neurons = ones(num_layers,1);
target_rate = cell(num_layers, num_targets);
target_rate_ndx = cell(num_layers, num_targets);
clutter_rate = cell(num_layers, 1);
clutter_rate_ndx = cell(num_layers, 1);


%% Analyze spiking activity layer by layer
for layer = read_spikes;
    disp(['analyzing layer: ', num2str(layer)]);
    
    % Read spike events
    disp('reading spikes');
    [spike_array{layer}, ave_rate(layer)] = pvp_readSparseSpikes(layer, pvp_order);
    disp([ layerID{layer}, ': ave_rate(',num2str(layer),') = ', num2str(ave_rate(layer))]);
    if isempty(spike_array{layer})
        continue;
    end%%if
    
    num_bins = fix( tot_steps / bin_size );
    excess_steps = tot_steps - num_bins * bin_size;
    if ( analysis_start_time < time_steps(1) * dt )
        analysis_start_time = time_steps(1) * dt;
    end%%if
    analysis_start_step = find( time_steps * dt >= analysis_start_time, 1, 'first' );
    if ( analysis_stop_time > time_steps(end) * dt )
        analysis_start_time = time_steps(end) * dt;
    end%%if
    analysis_stop_step = find( time_steps * dt <= analysis_stop_time, 1, 'last' );
    analysis_start_bin = ceil( analysis_start_step * dt / bin_size );
    analysis_stop_bin = fix( analysis_stop_step * dt / bin_size );
    analysis_bins = analysis_start_bin : analysis_stop_bin;
    if ( stim_begin_time < analysis_start_time )
        stim_begin_time = analysis_start_time;
    end%%if
    stim_begin_step = find( time_steps * dt >= stim_begin_time, 1, 'first' );
    stim_begin_bin = ceil( ( stim_begin_step / bin_size ) );
    if ( stim_end_time > analysis_stop_time )
        stim_end_time = analysis_stop_time;
    end%%if
    stim_end_step = find( time_steps * dt <= stim_end_time, 1, 'last' );
    stim_end_bin = fix( ( stim_end_step / bin_size ) );
    stim_steps = stim_begin_step : stim_end_step;
    stim_bins = stim_begin_bin : stim_end_bin;
    num_stim_bins = length(stim_bins);
    
    num_rows(layer) = NROWS;
    num_cols(layer) = NCOLS;
    num_features(layer) = NFEATURES;
    num_neurons(layer) = NFEATURES * NCOLS * NROWS;
    
    % parse object segmentation info input image
    use_max = 1;
    bkgrnd_ndx{layer} = find(ones(N,1));
    [clutter_ndx{layer}, clutter_ndx_max{layer}] = ...
        pvp_image2layer( spike_array{layer}, clutter, stim_steps, ...
        use_max, pvp_order);
    num_clutter_neurons(layer) = length( clutter_ndx{layer} );
    ave_clutter{layer} = ...
        full( 1000*sum(spike_array{layer}(:,clutter_ndx{layer}),2) / ...
        ( num_clutter_neurons(layer) + ( num_clutter_neurons(layer)==0 ) ) );
    bkgrnd_ndx{layer}(clutter_ndx{layer}) = 0;
    for i_target = 1:num_targets
        [target_ndx{layer, i_target}, target_ndx_max{layer, i_target}] = ...
            pvp_image2layer( spike_array{layer}, target{i_target}, ...
            stim_steps, use_max, pvp_order);
        num_target_neurons(layer, i_target) = length( target_ndx{layer, i_target} );
        ave_target{layer,i_target} = ...
            full( 1000*sum(spike_array{layer}(:,target_ndx{layer, i_target}),2) / ...
            ( num_target_neurons(layer, i_target) + ( num_target_neurons(layer, i_target)==0 ) ) );
        bkgrnd_ndx{layer}(target_ndx{layer, i_target}) = 0;
    end%%for % i_target
    bkgrnd_ndx{layer} = find(bkgrnd_ndx{layer});
    num_bkgrnd_neurons(layer) = N - sum(num_target_neurons(layer,:)) - num_clutter_neurons(layer);
    ave_bkgrnd{layer} = ...
        full( 1000*sum(spike_array{layer}(:,bkgrnd_ndx{layer}),2) / ...
        ( num_bkgrnd_neurons(layer) + (num_bkgrnd_neurons(layer) == 0) ) );
    
    disp([layerID{layer}, ': ave_target(',num2str(layer),') = ', num2str( mean( ave_target{layer,i_target}(:) ) ), 'Hz']);
    disp([layerID{layer}, ': ave_clutter(',num2str(layer),') = ', num2str( mean( ave_clutter{layer,1}(:) ) ), 'Hz']);
    disp([layerID{layer}, ': ave_bkgrnd(',num2str(layer),') = ', num2str( mean( ave_bkgrnd{layer,1}(:) ) ), 'Hz']);
    
    plot_rates = tot_steps > min_plot_steps;
    if ( plot_rates == 1 )
        plot_title = [layerID{layer}, ' PSTH(', int2str(layer), ')'];
        fig_tmp = figure('Name',plot_title);
        fig_list = [fig_list; fig_tmp];
        for i_target = 1 : num_targets
            psth_target{layer,i_target} = ...
                mean( reshape( ave_target{layer,i_target}(1:bin_size*num_bins), ...
                bin_size, num_bins  ), 1);
            lh = plot((analysis_bins)*bin_size, psth_target{layer,i_target}(analysis_bins), '-r');
            set(lh, 'LineWidth', 2);
            hold on
        end%%for % i_target
        psth_clutter{layer,1} = ...
            mean( reshape( ave_clutter{layer,1}(1:bin_size*num_bins), ...
            bin_size, num_bins  ), 1);
        lh = plot((analysis_bins)*bin_size, psth_clutter{layer,1}(analysis_bins), '-b');
        set(lh, 'LineWidth', 2);
        psth_bkgrnd{layer,1} = ...
            mean( reshape( ave_bkgrnd{layer,1}(1:bin_size*num_bins), ...
            bin_size, num_bins  ), 1);
        lh = plot((analysis_bins)*bin_size, psth_bkgrnd{layer,1}(analysis_bins), '-k');
        set(lh, 'LineWidth', 2);
        set(lh, 'Color', my_gray);
    end%%if % plot_rates
    
    
    % raster plot
    plot_raster2 = ismember( layer, plot_raster ) && tot_steps > min_plot_steps && ~isempty(spike_array{layer});
    if plot_raster2
        plot_title = [layerID{layer}, ' Raster(',int2str(layer)', ')'];
        fig_tmp = figure('Name',plot_title);
        fig_list = [fig_list; fig_tmp];
        %axis([analysis_start_step analysis_stop_step 0 num_neurons(layer)]);
        [spike_time, spike_id] = find((spike_array{layer}));
        lh = plot(spike_time, spike_id, '.k');
        set(lh,'Color',my_gray);
        hold on
        
        for i_target=1:num_targets
            [spike_time, spike_id] = ...
                find((spike_array{layer}(analysis_start_step:analysis_stop_step, ...
                target_ndx{layer, i_target})));
            plot(spike_time, target_ndx{layer, i_target}(spike_id), '.r');
        end%%for % i_target
        
        [spike_time, spike_id] = ...
            find((spike_array{layer}(analysis_start_step:analysis_stop_step, ...
            clutter_ndx{layer})));
        plot(spike_time*dt, clutter_ndx{layer}(spike_id), '.b');
        
    end%%if  % plot_raster
    
    
    rate_array{layer} = 1000 * full( mean(spike_array{layer}(stim_steps,:),1) );
    size_recon = [1 num_features(layer), num_cols(layer), num_rows(layer)];
    for i_target = 1:num_targets
        target_rate{layer, i_target} = rate_array{layer}(1,target_ndx{layer, i_target});
        target_rate_array_tmp = ...
            sparse(1, target_ndx{layer, i_target}, target_rate{layer, i_target}, 1 , N, num_target_neurons(layer, i_target) );
        if (plot_reconstruct_target && tot_steps > min_plot_steps)
            plot_title = ...
                [layerID{layer}, ...
                ' Target(', ...
                int2str(layer), ...
                ',', ...
                int2str(i_target), ')'];
            fig_tmp = ...
                pvp_reconstruct(full(target_rate_array_tmp), ...
                plot_title, ...
                [], ...
                size_recon );
            fig_list = [fig_list; fig_tmp];
        end%%if %  reconstruc target/clutter
        [target_rate{layer, i_target}, target_rate_ndx{layer, i_target}] = ...
            sort( target_rate{layer, i_target}, 2, 'descend');
        for i_rank = [ 1:3 ] % , ceil(num_target_neurons(layer, i_target)/2), num_target_neurons(layer, i_target) ]
            tmp_rate = target_rate{layer, i_target}(i_rank);
            tmp_ndx = target_rate_ndx{layer, i_target}(i_rank);
            k = target_ndx{layer, i_target}(tmp_ndx);
            [kf, kcol, krow] = ind2sub([num_features(layer), num_cols(layer), num_rows(layer)], k);
            disp(['rank:',num2str(i_rank),...
                ', target_rate(',num2str(layer),', ', num2str(i_target), ')', ...
                num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
                num2str([k-1, kcol-1, krow-1, kf-1]) ]);
        end%%for % i_rank
    end%%for % i_target
    clutter_rate{layer, 1} = rate_array{layer}(1,clutter_ndx{layer, 1});
    [clutter_rate{layer, 1}, clutter_rate_ndx{layer, 1}] = ...
        sort( clutter_rate{layer, 1}, 2, 'descend');
    clutter_rate_array_tmp = ...
        sparse(1, clutter_ndx{layer, 1}, clutter_rate{layer, 1}, 1 , N, num_clutter_neurons(layer, 1) );
    if (plot_reconstruct_target && tot_steps > min_plot_steps)
        plot_title = ...
            [layerID{layer}, ...
            ' Clutter(', ...
            int2str(layer), ...
            ')'];
        fig_tmp = ...
            pvp_reconstruct(full(clutter_rate_array_tmp), ...
            plot_title, ...
            [], ...
            size_recon);
        fig_list = [fig_list; fig_tmp];
    end%%if
    for i_rank = [ 1:3 ] % , ceil(num_clutter_neurons(layer, 1)/2), num_clutter_neurons(layer, 1) ]
        tmp_rate = clutter_rate{layer, 1}(i_rank);
        tmp_ndx = clutter_rate_ndx{layer, 1}(i_rank);
        k = clutter_ndx{layer, 1}(tmp_ndx);
        [kf, kcol, krow] = ind2sub([num_features(layer), num_cols(layer), num_rows(layer)], k);
        disp(['rank:',num2str(i_rank),...
            ', clutter_rate(', num2str(layer),')', ...
            num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
            num2str([k-1, kcol-1, krow-1, kf-1]) ]);
    end%%for % i_rank
    
    % plot reconstructed image
    plot_rate_reconstruction = ( ismember( layer, plot_reconstruct ) && tot_steps > ...
        min_plot_steps );
    if plot_rate_reconstruction
        plot_title = ...
            [layerID{layer}, ...
            'Image(', ...
            int2str(layer), ...
            ')'];
        fig_tmp = ...
            pvp_reconstruct(rate_array{layer}, ...
            plot_title, ...
            [], ...
            size_recon);
        fig_list = [fig_list; fig_tmp];
    end%%if
    
    
    
    %% plot spike movie
    % original version does not work in octave, which lacks getframe, movie2avi, etc
    plot_movie = 0; %tot_steps > 9;
    if plot_movie
        spike_movie = pvp_movie( spike_array, layer);
    end%%if % plot_movie
    
    
end%%for % layer


%% plot connections
global N_CONNECTIONS
global NXP NYP NFP
[connID, connIndex] = pvp_connectionID();
plot_weights = 1:0;%N_CONNECTIONS;
weights = cell(N_CONNECTIONS, 1);
pvp_header = cell(N_CONNECTIONS, 1);
nxp = cell(N_CONNECTIONS, 1);
nyp = cell(N_CONNECTIONS, 1);
nfp = cell(N_CONNECTIONS, 1);
for i_conn = plot_weights
    [weights{i_conn}, nxp{i_conn}, nyp{i_conn}, pvp_header{i_conn}, pvp_index ] ...
        = pvp_readWeights(i_conn);
    NK = 1;
    NO = floor( NFEATURES / NK );
    pvp_header_tmp = pvp_header{i_conn};
    num_patches = pvp_header_tmp(pvp_index.WGT_NUMPATCHES);
    NFP = pvp_header_tmp(pvp_index.WGT_NFP);
    skip_patches = 1; %num_patches;
    for i_patch = 1 : skip_patches : num_patches
        NXP = nxp{i_conn}(i_patch);
        NYP = nyp{i_conn}(i_patch);
        N = NFP * NXP * NYP;
        plot_title = ...
            [connID{i_conn}, ...
            '(', ...
            int2str(i_conn), ...
            ',', ...
            int2str(i_patch), ...
            ')'];
        size_recon = ...
            [1, NFP, NXP, NYP];
        fig_tmp = ...
            pvp_reconstruct(weights{i_conn}{i_patch}, ...
            plot_title, ...
            [], ...
            size_recon);
        fig_list = [fig_list; fig_tmp];
    end%%for % i_patch
end%%for % i_conn


%%read membrane potentials from point probes
if plot_vmem
    disp('plot_vmem')
    
    % TODO: the following info should be read from a pv ouptut file
    vmem_file_list = {'Vmem_LGNA1.txt', ...
        'Vmem_LGNC1.txt', ...
        'Vmem_LGNInhFFA1.txt', ...
        'Vmem_LGNInhFFC1.txt', ...
        'Vmem_LGNInhA1.txt', ...
        'Vmem_LGNInhC1.txt', ...
        'Vmem_V1A1.txt', ...
        'Vmem_V1C1.txt', ...
        'Vmem_V1InhFFA1.txt', ...
        'Vmem_V1InhFFC1.txt', ...
        'Vmem_V1InhA1.txt', ...
        'Vmem_V1InhC1.txt'};
    %  vmem_layers = [2,3,4,5];
    num_vmem_files = size(vmem_file_list,2);
    vmem_time = cell(num_vmem_files, 1);
    vmem_G_E = cell(num_vmem_files, 1);
    vmem_G_I = cell(num_vmem_files, 1);
    vmem_G_IB = cell(num_vmem_files, 1);
    vmem_V = cell(num_vmem_files, 1);
    vmem_Vth = cell(num_vmem_files, 1);
    vmem_a = cell(num_vmem_files, 1);
    AX_vmem = cell(num_vmem_files, 1);
    H1_vmem = cell(num_vmem_files, 1);
    H2_vmem = cell(num_vmem_files, 1);
    vmem_skip = 2;
    vmem_start_time = analysis_start_time;
    vmem_stop_time = analysis_stop_time;
    for i_vmem = 1 : vmem_skip : num_vmem_files
        %    vmem_layer = vmem_layers(i_vmem);
        %   if ( ~ismember( vmem_layer, read_spikes ) )
        %     continue; % NROWS = 1, NFEATURES = 1;
        %   end%%if
        %    NROWS = num_rows(vmem_layer);
        %    NFEATURES = num_features(vmem_layer);
        [vmem_time{i_vmem}, vmem_G_E{i_vmem}, vmem_G_I{i_vmem}, vmem_G_IB{i_vmem}, vmem_V{i_vmem}, vmem_Vth{i_vmem}, vmem_a{i_vmem} ] = ...
            ptprobe_readV(vmem_file_list{i_vmem});
        % if pvp_order
        %   vmem_index = ( vmem_row * num_cols(vmem_layer) + vmem_col ) * num_features(vmem_layers) + vmem_feature;
        % end%%if
        vmem_start = find(vmem_time{i_vmem} == vmem_start_time);
        vmem_stop = find(vmem_time{i_vmem} == vmem_stop_time);
        plot_title = [ 'Vmem data: ', vmem_file_list{i_vmem} ];
        fh = figure('Name', plot_title);
        [AX_vmem{i_vmem},H1_vmem{i_vmem},H2_vmem{i_vmem}] = ...
            plotyy( vmem_time{i_vmem}(vmem_start:vmem_stop), ...
            [vmem_V{i_vmem}(vmem_start:vmem_stop), ...
            vmem_Vth{i_vmem}(vmem_start:vmem_stop)], ...
            vmem_time{i_vmem}(vmem_start:vmem_stop),  ...
            [vmem_G_E{i_vmem}(vmem_start:vmem_stop), ...
            vmem_G_I{i_vmem}(vmem_start:vmem_stop), ...
            vmem_G_IB{i_vmem}(vmem_start:vmem_stop)] );
        set(H1_vmem{i_vmem}(1), 'Color', [0 0 0] );
        set(H1_vmem{i_vmem}(2), 'Color', [1 0 0] );
        set(H2_vmem{i_vmem}(1), 'Color', [0 1 0] );
        set(H2_vmem{i_vmem}(2), 'Color', [0 0 1] );
        set(H2_vmem{i_vmem}(3), 'Color', [0 1 1] );
    end%%for % i_vmem
end%%if %plot_vmem



%% plot psth's of all layers together
plot_rates = ( ( tot_steps > min_plot_steps ) && ( length(read_spikes) == ...
    N_LAYERS ) );
if plot_rates
    plot_title = ['PSTH target pixels'];
    fig_tmp = ...
        figure('Name',plot_title);
    hold on
    co = get(gca,'ColorOrder');
    lh = zeros(4,1);
    for layer = 1:N_LAYERS
        lh(layer) = plot((analysis_bins)*bin_size, psth_target{layer,i_target}(analysis_bins), '-r');
        set(lh(layer),'Color',co(layer,:));
        set(lh(layer),'LineWidth',2);
    end%%for
    legend_str = ...
        {'retina  '; ...
        'LGN     '; ...
        'LGNInhFF'; ...
        'LGNInh  '; ...
        'V1      '; ...
        'V1InhFF '; ...
        'V1Inh   '};
    if uimatlab
        leg_h = legend(lh(1:num_layers), legend_str);
    elseif uioctave
        legend(legend_str);
    end%%if
    fig_list = [fig_list; fig_tmp];
end%%if



%% autocorr
%stft_array = cell( num_layers, 1);
power_array = cell( num_layers, 2);
for layer = 1:num_layers
    plot_autocorr2 = ( ismember( layer, plot_autocorr ) && tot_steps > ...
        min_plot_steps );
    if ~plot_autocorr2
        continue;
    end%%if
    
    NK = 1;
    NO = num_features(layer);
    NROWS = num_rows(layer);
    NCOLS = num_cols(layer);
    
    % autocorrelation of psth
    disp( [ 'computing AutoCorr(', num2str(layer), ')'] );
    plot_title = [layerID{layer}, 'Auto Corr(',int2str(layer), ')'];
    fig_tmp = figure('Name',plot_title);
    fig_list = [fig_list; fig_tmp];
    ave_target_tmp = full(ave_target{layer,i_target}(stim_steps));
    max_lag= min( 128/dt, fix(length(ave_target_tmp)/8) );
    [target_xcorr, target_lag] = ...
        xcorr( ave_target_tmp', [], max_lag, 'unbiased' );
    target_xcorr = ( target_xcorr - mean(ave_target_tmp)^2 ) / ...
        (mean(ave_target_tmp)+(mean(ave_target_tmp)==0))^2;
    target_xcorr(max_lag + 1) = 0;
    lh_target = plot((-max_lag:max_lag)*dt, target_xcorr, '-r');
    set(lh_target, 'LineWidth', 2);
    hold on
    ave_clutter_tmp = full(ave_clutter{layer,1}(stim_steps));
    [clutter_xcorr, clutter_lag] = ...
        xcorr( ave_clutter_tmp', [], max_lag, 'unbiased' );
    clutter_xcorr = ( clutter_xcorr - mean(ave_clutter_tmp)^2 ) / ...
        (mean(ave_clutter_tmp)+(mean(ave_clutter_tmp)==0))^2;
    clutter_xcorr(max_lag + 1) = 0;
    lh_clutter = plot((-max_lag:max_lag)*dt, clutter_xcorr, '-b');
    set(lh_clutter, 'LineWidth', 2);
    ave_bkgrnd_tmp = full(ave_bkgrnd{layer,1}(stim_steps));
    [bkgrnd_xcorr, bdgrnd_lag] = ...
        xcorr( ave_bkgrnd_tmp', [], max_lag, 'unbiased' );
    bkgrnd_xcorr = ( bkgrnd_xcorr - mean(ave_bkgrnd_tmp)^2 ) / ...
        (mean(ave_bkgrnd_tmp)+(mean(ave_bkgrnd_tmp)==0))^2;
    bkgrnd_xcorr(max_lag + 1) = 0;
    lh_bkgrnd = plot((-max_lag:max_lag)*dt, bkgrnd_xcorr, '-k');
    set(lh_bkgrnd, 'Color', my_gray);
    set(lh_bkgrnd, 'LineWidth', 2);
    [target2clutter_xcorr, target2clutter_lag] = ...
        xcorr( ave_target_tmp', ave_clutter_tmp', max_lag, 'unbiased' );
    target2clutter_xcorr = ...
        ( target2clutter_xcorr - mean(ave_target_tmp) * mean(ave_clutter_tmp) ) / ...
        ( (mean(ave_target_tmp)+(mean(ave_target_tmp)==0)) * ...
        (mean(ave_clutter_tmp)+(mean(ave_clutter_tmp)==0)) );
    lh_target2clutter = plot((-max_lag:max_lag)*dt, target2clutter_xcorr, '-g');
    set(lh_target2clutter, 'LineWidth', 2);
    axis tight
    
    %plot power spectrum
    disp( [ 'computing Power(', num2str(layer), ')'] );
    plot_title = ['Power: layer = ',int2str(layer)];
    figure('Name',plot_title);
    freq = (1/dt)*1000*(0:2*max_lag)/(2*max_lag + 1);
    min_ndx = find(freq > 128, 1,'first');
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
    clear ave_target_tmp target_xcorr
    clear ave_clutter_tmp clutter_xcorr
    clear ave_bkgrnd_tmp bkgrnd_xcorr
    clear target2clutter_xcorr
    clear target_fft clutter_fft bkgrnd_fft target2clutter_fft
    
    %plot power reconstruction
    disp( [ 'computing Power Recon(', num2str(layer), ')'] );
    stft_win_size = max_lag/dt;
    stft_inc = ceil( stft_win_size );
    stft_num_coef = stft_win_size;
    stft_w_type = 3; % rectangle
    num_trials = fix( length( stim_steps ) / stft_win_size );
    freq_vals = 1000*(1/dt)*(0:stft_win_size-1)/stft_win_size;
    min_ndx = find(freq_vals >= 20, 1,'first');
    max_ndx = find(freq_vals <= 60, 1,'last');
    stft_array = zeros( stft_win_size, N );
    for i_trial = 1 : num_trials
        stft_start = stim_begin_step + ( i_trial - 1 ) * stft_inc;
        stft_stop = stft_start + stft_win_size - 1;
        stft_array = stft_array + ...
            fft( spike_array{layer}( stft_start : stft_stop, :) );
    end%%for
    stft_array = stft_array / num_trials;
    stft_array = stft_array .* conj( stft_array );
    power_array{layer, 1} = max(stft_array(min_ndx:max_ndx,:));
    power_array{layer, 2} = mean(stft_array(min_ndx:max_ndx,:));
    clear stft_array
    plot_power_reconstruction = ( ismember( layer, plot_reconstruct ) && tot_steps > ...
        min_plot_steps );
    if plot_power_reconstruction
        pvp_reconstruct(power_array{layer, 1},  [layerID{layer}, ': Peak Power: layer = ', int2str(layer)]);
        pvp_reconstruct(power_array{layer, 2},  [layerID{layer}, ': Ave Power: layer = ', int2str(layer)]);
    end%%if
end%%for % layer

%% xcorr & eigen analysis
xcorr_eigenvector = cell( num_layers, 2, 2);
power_mask = cell(num_layers,2);
num_power_mask = zeros(num_layers, 2);
for layer = 1:num_layers
    plot_xcorr2 = ( ismember( layer, plot_xcorr ) && tot_steps > ...
        min_plot_steps );
    if ~plot_xcorr2
        continue;
    end%%if
    
    NK = 1;
    NO = num_features(layer);
    NROWS = num_rows(layer);
    NCOLS = num_cols(layer);
    NFEATURES = num_features(layer);
    N = NROWS * NCOLS * NFEATURES;
    
    %computer power mask
    calc_power_mask = 1;
    for i_mode = 2 : 2
        if i_mode == 1
            disp('calculating peak power mask...');
        elseif i_mode == 2
            disp('calculating ave power mask...');
        end%%if
        if calc_power_mask
            num_power_sig = 0.0;
            mean_power = mean( power_array{layer, i_mode} );
            std_power = std( power_array{layer, i_mode} );
            disp( [ 'mean_power(', num2str(layer), ') = ', ...
                num2str(mean_power), ' +/- ', num2str(std_power) ] );
            power_mask{layer, i_mode} = find( power_array{layer, i_mode} > ( mean_power + num_power_sig * std_power ) );
            num_power_mask(layer, i_mode) = numel(power_mask{layer, i_mode});
            disp( ['num_power_mask(', num2str(layer), ') = ', num2str(num_power_mask(layer,i_mode)), ' > ', ...
                num2str( mean_power + num_power_sig * std_power ) ] );
            while num_power_mask(layer,i_mode) > ( length(target) + length(clutter) )
                num_power_sig = num_power_sig + 0.5;
                power_mask{layer, i_mode} = find( power_array{layer, i_mode} > ( mean_power + num_power_sig * std_power ) );
                num_power_mask(layer,i_mode) = numel(power_mask{layer,i_mode});
                disp( ['num_power_mask(', num2str(layer), ') = ', num2str(num_power_mask(layer,i_mode)), ' > ', ...
                    num2str( mean_power + num_power_sig * std_power ) ] );
            end%%while
        else
            power_mask{layer,i_mode} = clutter_ndx{layer,1};
            for i_target = 1 : num_targets
                power_mask{layer,i_mode} = [power_mask{layer,i_mode}, target_ndx{layer, i_target}];
            end%%for
            power_mask{layer,i_mode} = sort( power_mask{layer,i_mode} );
            num_power_mask(layer,i_mode) = numel(power_mask{layer,i_mode});
        end%%if % calc_power_mask
    end%%for % i_mode
    
    %% compute xcorr
    for i_mode = 2 : 2  % 2 : 2
        
        disp(['computing xcorr(', num2str(layer), '): i_mode = ', ...
            num2str(i_mode) ] );
        size_layer = [num_rows(layer), num_cols(layer), num_features(layer)];
        plot_interval = fix( num_power_mask(layer, i_mode)^2 / 1 );
        [xcorr_array, xcorr_std, xcorr_dist, xcorr_mean, xcorr_lags, xcorr_figs] = ...
            pvp_xcorr( spike_array{layer}(stim_steps, power_mask{layer, i_mode}), ...
            spike_array{layer}(stim_steps, power_mask{layer, i_mode}), ...
            max_lag, ...
            power_mask{layer, i_mode}, size_layer, ...
            power_mask{layer, i_mode}, size_layer, ...
            1, plot_interval );
        fig_list = [fig_list; xcorr_figs];
        plot_title = [layerID{layer}, ' mass xcorr(',int2str(layer), ')'];
        fig_tmp = figure('Name',plot_title);
        fig_list = [fig_list; fig_tmp];
        mass_xcorr = zeros( 2 * max_lag + 1, 1 );
        mass_xcorr(max_lag + 1 : end) = ...
            squeeze( sum( sum( xcorr_array, 2 ), 1 ) ) / ...
            ( num_power_mask(layer, i_mode)^2 );
        mass_xcorr_flip = ...
            mass_xcorr( max_lag + 2 : 2 * max_lag + 1, 1 );
        mass_xcorr_flip = ...
            flipdim( mass_xcorr_flip );
        mass_xcorr(1 : max_lag, 1) =  ...
            mass_xcorr_flip;
        %clear mass_xcorr_flip;
        plot( (-max_lag : max_lag)*dt, mass_xcorr, '-k');
        mass_xcorr_std = ...
            sqrt( squeeze( sum( xcorr_std(:).^2 ) ) ) / ...
            ( num_power_mask(layer, i_mode)^2 );
        lh = line( [-max_lag, max_lag]*dt, ...
            [ mass_xcorr_std mass_xcorr_std ] );
        
        % extract scalar pairwise correlations
        
        % find eigen vectors
        calc_eigen = 0;
        if calc_eigen
            disp(['computing eigenvectors(', num2str(layer),')']);
            options.issym = 1;
            num_eigen = 64;
            [eigen_vec, eigen_value, eigen_flag] = ...
                eigs( (1/2)*(sparse_corr{sync_true} + sparse_corr{sync_true}'), num_eigen, 'lm', options);
            for i_vec = 1:num_eigen
                disp(['eigenvalues(', num2str(layer), ',' ...
                    , num2str(i_vec),') = ', num2str(eigen_value(i_vec,i_vec))]);
            end%%for
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
                pvp_reconstruct(eigen_vec_tmp, plot_title);
            end%%for % i_vec
            
        end%%if % calc_eigen
        
    end%%for % i_mode
    
end%%for % layer

