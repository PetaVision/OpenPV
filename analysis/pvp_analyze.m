%%
close all
clear all

				% set paths, may not be applicable to all octave installations
local_pwd = pwd;
user_index1 = findstr(local_pwd, 'Users')';
user_index1 = user_index1(1);
if ~isempty(user_index1)
  user_name = local_pwd(user_index1+6:length(local_pwd));
  user_index2 = findstr(user_name, '/');
  if isempty(user_index2)
    user_index2 = length(user_name);
  endif
  user_name = user_name(1:user_index2-1);
  matlab_dir = ['/Users/', user_name, '/Documents/MATLAB'];
  addpath(matlab_dir);
endif

if ( uioctave )
  setenv("GNUTERM", "x11");
endif

				% Make the following global parameters available to all functions for convenience.
global N_image NROWS_image NCOLS_image
global N NROWS NCOLS % for the current layer
global NFEATURES  % for the current layer
global NO NK dK % for the current layer
global n_time_steps begin_step end_step time_steps tot_steps
global stim_begin_step stim_end_step stim_steps
global analysis_start_time analysis_stop_time
global bin_size dt
global begin_time end_time stim_begin_time stim_end_time
global num_targets
global rate_array
global output_path input_path
global NUM_BIN_PARAMS 
global NUM_WGT_PARAMS
NUM_BIN_PARAMS = 20;
NUM_WGT_PARAMS = 6;

input_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/';
output_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/output/';

pvp_order = 1;

				% begin step, end_step and stim_begin_step may be adusted by
				% pvp_readSparseSpikes
begin_time = 0.0;  % (msec) start analysis here, used to exclude start up artifacts
end_time = inf;
stim_begin_time = 500.0;  % times (msec) before this and after begin_time can be used to calculate background
stim_end_time = 1500.0;
bin_size = 10.0;  % (msec) used for all rate calculations
analysis_start_time = 50.0;
analysis_stop_time = 2000.0;
dt = 1.0; % msec

%stim_begin_bin = fix( ( stim_begin_step - begin_step + 1 ) / bin_size );
%stim_end_bin = fix( ( stim_end_step - begin_step + 1 ) / bin_size );
%stim_bins = stim_begin_bin : stim_end_bin;
%num_stim_bins = length(stim_bins);


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

read_spikes = 1:7;  % list of spiking layers whose spike train are to be analyzed
num_layers = length(read_spikes);

min_plot_steps = 20;  % time-dependent quantities only plotted if tot_steps exceeds this threshold
plot_reconstruct = [ ]; %uimatlab;
plot_raster = [2 5]; %uimatlab;
plot_reconstruct_target = 0;
plot_vmem = 1;
plot_xcorr = [];

spike_array = cell(num_layers,1);
ave_rate = zeros(num_layers,1);

%% read input image
bmp_path = [input_path 'hard4.bmp'];
[target, clutter, fig_tmp] = pvp_parseBMP( bmp_path, 1 );
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
target_rate = cell(num_layers, num_targets);
target_rate_ndx = cell(num_layers, num_targets);
clutter_rate = cell(num_layers, 1);
clutter_rate_ndx = cell(num_layers, 1);


%% Analyze spiking activity layer by layer
global N_LAYERS
[layerID, layerIndex] = pvp_layerID();
for layer = read_spikes;
  disp(['analyzing layer: ', num2str(layer)]);
  
				% Read spike events
  disp('reading spikes');
  [spike_array{layer}, ave_rate(layer)] = pvp_readSparseSpikes(layer, pvp_order);
  disp([ layerID{layer}, ': ave_rate(',num2str(layer),') = ', num2str(ave_rate(layer))]);
  if isempty(spike_array{layer})
    continue;
  endif
  
  num_bins = fix( tot_steps / bin_size );
  excess_steps = tot_steps - num_bins * bin_size;
  if ( analysis_start_time < time_steps(1) * dt )
    analysis_start_time = time_steps(1) * dt;
  endif
  analysis_start_step = find( time_steps * dt >= analysis_start_time, 1, 'first' );
  if ( analysis_stop_time > time_steps(end) * dt )
    analysis_start_time = time_steps(end) * dt;
  endif
  analysis_stop_step = find( time_steps * dt >= analysis_stop_time, 1, 'last' );
  analysis_start_bin = fix( analysis_start_step * dt / bin_size );
  analysis_stop_bin = fix( analysis_stop_step * dt / bin_size );
  analysis_bins = analysis_start_bin : analysis_stop_bin;
  num_rows(layer) = NROWS;
  num_cols(layer) = NCOLS;
  num_features(layer) = NFEATURES;
  
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
  endfor % i_target
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
    plot_title = [layerID{layer}, ' PSTH: layer = ',int2str(layer)];
    figure('Name',plot_title);
    for i_target = 1 : num_targets
      psth_target{layer,i_target} = ...
          mean( reshape( ave_target{layer,i_target}(1:bin_size*num_bins), ...
			bin_size, num_bins  ), 1);
      lh = plot((analysis_bins)*bin_size, psth_target{layer,i_target}(analysis_bins), '-r');
      set(lh, 'LineWidth', 2);
      hold on
    endfor % i_target
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
  endif % plot_rates
  
  
				% raster plot
  plot_raster2 = ismember( layer, plot_raster ) && tot_steps > min_plot_steps && ~isempty(spike_array{layer});
  if plot_raster2
    plot_title = [layerID{layer}, ' Raster: layer = ',int2str(layer)'];
    fig_tmp = figure('Name',plot_title);
    fig_list = [fig_list; fig_tmp];
    if uioctave
      axis([analysis_start_time analysis_stop_time 0 N], 'ticx');
    else
      axis([analysis_start_time analysis_stop_time 0 N]);
    endif % uioctave
    hold on
    if ~uioctave
      box on
    endif % uioctave
    [spike_time, spike_id] = find((spike_array{layer}));
    lh = plot(spike_time, spike_id, '.k');
    set(lh,'Color',my_gray);
    
    for i_target=1:num_targets
      [spike_time, spike_id] = ...
          find((spike_array{layer}(analysis_start_step:analysis_stop_step, ...
				   target_ndx{layer, i_target})));
      plot(spike_time, target_ndx{layer, i_target}(spike_id), '.r');
    endfor % i_target
    
    [spike_time, spike_id] = ...
	find((spike_array{layer}(analysis_start_step:analysis_stop_step, ...
				 clutter_ndx{layer})));
    plot(spike_time*dt, clutter_ndx{layer}(spike_id), '.b');
    
  endif  % plot_raster
  

  rate_array{layer} = 1000 * full( mean(spike_array{layer}(stim_steps,:),1) );
  for i_target = 1:num_targets
    target_rate{layer, i_target} = rate_array{layer}(1,target_ndx{layer, i_target});
    target_rate_array_tmp = ...
        sparse(1, target_ndx{layer, i_target}, target_rate{layer, i_target}, 1 , N, num_target_neurons(layer, i_target) );
    if (plot_reconstruct_target && tot_steps > min_plot_steps)
      fig_tmp = pvp_reconstruct(full(target_rate_array_tmp), ...
				[layerID{layer}, ' Target rate reconstruction; layer = ', ...
				 int2str(layer), ', target index = ', ...
				 int2str(i_target)]);     
      fig_list = [fig_list; fig_tmp];
    endif %  reconstruc target/clutter
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
    endfor % i_rank
  endfor % i_targetr
  clutter_rate{layer, 1} = rate_array{layer}(1,clutter_ndx{layer, 1});
  [clutter_rate{layer, 1}, clutter_rate_ndx{layer, 1}] = ...
      sort( clutter_rate{layer, 1}, 2, 'descend');
  clutter_rate_array_tmp = ...
      sparse(1, clutter_ndx{layer, 1}, clutter_rate{layer, 1}, 1 , N, num_clutter_neurons(layer, 1) );
  if (plot_reconstruct_target && tot_steps > min_plot_steps)
    fig_tmp = pvp_reconstruct(full(clutter_rate_array_tmp), ...
			      [layerID{layer}, ' Clutter rate reconstruction: layer = ', ...
			       int2str(layer)]);
    fig_list = [fig_list; fig_tmp];
  endif
  for i_rank = [ 1:3 ] % , ceil(num_clutter_neurons(layer, 1)/2), num_clutter_neurons(layer, 1) ]
    tmp_rate = clutter_rate{layer, 1}(i_rank);
    tmp_ndx = clutter_rate_ndx{layer, 1}(i_rank);
    k = clutter_ndx{layer, 1}(tmp_ndx);
    [kf, kcol, krow] = ind2sub([num_features(layer), num_cols(layer), num_rows(layer)], k);
    disp(['rank:',num2str(i_rank),...
          ', clutter_rate(', num2str(layer),')', ...
          num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
	  num2str([k-1, kcol-1, krow-1, kf-1]) ]);
  endfor % i_rank
  
				% plot reconstructed image
  plot_rate_reconstruction = ( ismember( layer, plot_reconstruct ) && tot_steps > ...
			      min_plot_steps );
  if plot_rate_reconstruction
    pvp_reconstruct(rate_array{layer}, ...
		    [layerID{layer}, ' Rate reconstruction: layer = ', int2str(layer)]);
    fig_list = [fig_list; fig_tmp];
  endif



  %% plot spike movie
  % original version does not work in octave, which lacks getframe, movie2avi, etc
  plot_movie = 0; %tot_steps > 9;
  if plot_movie
    spike_movie = pvp_movie( spike_array, layer);
  endif % plot_movie
  

endfor % layer


%% plot connections
global N_CONNECTIONS
global NXP NYP NFP
[connID, connIndex] = pvp_connectionID();
plot_weights = 1:0;%N_CONNECTIONS;
weights = cell(N_CONNECTIONS, 1);
pvp_header = cell(N_CONNECTIONS, 1);
nxp = cell(N_CONNECTIONS, 1);
nyp = cell(N_CONNECTIONS, 1);
for i_conn = plot_weights
  [weights{i_conn}, nxp{i_conn}, nyp{i_conn}, pvp_header{i_conn}, pvp_index ] ...
      = pvp_readWeights(i_conn);
  NK = 1;
  NO = floor( NFEATURES / NK );
  pvp_header_tmp = pvp_header{i_conn};
  num_patches = pvp_header_tmp(pvp_index.WGT_NUMPATCHES);
  skip_patches = num_patches;
  for i_patch = 1 : skip_patches : num_patches
    NCOLS = nxp{i_conn}(i_patch);
    NROWS = nyp{i_conn}(i_patch);
    N = NROWS * NCOLS * NFEATURES;
    pvp_reconstruct(weights{i_conn}{i_patch}, [connID{i_conn}, ' Weight recon: i_conn = ', ...
					       int2str(i_conn), ': i_patch = ', ...
					       int2str(i_patch) ]);
  endfor % i_patch
endfor % i_conn


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
				%   endif
				%    NROWS = num_rows(vmem_layer);
				%    NFEATURES = num_features(vmem_layer);
    [vmem_time{i_vmem}, vmem_G_E{i_vmem}, vmem_G_I{i_vmem}, vmem_G_IB{i_vmem}, vmem_V{i_vmem}, vmem_Vth{i_vmem}, vmem_a{i_vmem} ] = ...
        ptprobe_readV(vmem_file_list{i_vmem});
				% if pvp_order
				%   vmem_index = ( vmem_row * num_cols(vmem_layer) + vmem_col ) * num_features(vmem_layers) + vmem_feature;
				% endif 
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
  endfor % i_vmem
endif %plot_vmem



%% plot psth's of all layers together
plot_rates = tot_steps > min_plot_steps ;
if plot_rates
  plot_title = ['PSTH target pixels'];
  figure('Name',plot_title);
  hold on
  co = get(gca,'ColorOrder');
  lh = zeros(4,1);
  for layer = 1:num_layers
    lh(layer) = plot((analysis_bins)*bin_size, psth_target{layer,i_target}(analysis_bins), '-r');
    set(lh(layer),'Color',co(layer,:));
    set(lh(layer),'LineWidth',2);
  endfor
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
  endif
endif



%% xcorr and eigen vectors
xcorr_eigenvector = cell( num_layers, 2);
for layer = 1:num_layers
  plot_xcorr2 = ( ismember( layer, plot_xcorr ) && tot_steps > ...
		min_plot_steps );
  if ~plot_xcorr2
    continue;
  endif

        % autocorrelation of psth
        plot_title = ['Auto Corr: layer = ',int2str(layer)];
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
            sparse_corr = ...
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

