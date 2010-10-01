%%
close all
clear all

				% set paths, may not be applicable to all octave installations
				% pvp_matlabPath;

				% if ( uioctave )
if exist('setenv')
  setenv('GNUTERM', 'x11');
endif %%
				% endif %%

				% Make the following global parameters available to all functions for convenience.
global N_image NROWS_image NCOLS_image
global N NROWS NCOLS % for the current layer
global NFEATURES  % for the current layer
global NO NK dK % for the current layer
global BIN_STEP_SIZE DELTA_T
global BEGIN_TIME END_TIME
global NUM_BIN_PARAMS
global NUM_WGT_PARAMS
NUM_BIN_PARAMS = 20;
NUM_WGT_PARAMS = 6;

global FLAT_ARCH_FLAG
FLAT_ARCH_FLAG = 1;

global SPIKING_FLAG
SPIKING_FLAG = 1;

global OUTPUT_PATH SPIKE_PATH
				%SPIKE_PATH = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/128/spiking_2fc/';
SPIKE_PATH = '/nh/home/gkenyon/workspace/kernel/input/128/spiking_4fc_G1/';
OUTPUT_PATH = SPIKE_PATH; %'/nh/home/gkenyon/workspace/kernel/output/';

				%input_path = '/Users/gkenyon/Documents/eclipse-workspace/PetaVision/mlab/amoebaGen/128_png/2/';
input_path = '/nh/home/gkenyon/Documents/MATLAB/amoeba/128_png/4/';

				%global image_path target_path
image_filename = [input_path 't/tar_0041_a.png'];
target_filename{1} = [input_path 'a/tar_0041_a.png'];

pvp_order = 1;

%% set duration of simulation, if known (determined automatically otherwise)
BEGIN_TIME = 1000.0;  % (msec) start analysis here, used to exclude start up artifacts
END_TIME = inf;
stim_begin_time = 0.0;  % stim begin/end times (msec) relative to begining of each epoch, must be >= 0
stim_end_time = -0.0;  % relative to end of epoch, must be <= 0
BIN_STEP_SIZE = 5.0;  % (msec) used for all rate calculations
DELTA_T = 1.0; % msec
if ( stim_begin_time > 0.0 )
  stim_begin_time = 0.0;
endif %%
stim_begin_step = floor( stim_begin_time / DELTA_T ) + 1;
stim_begin_bin = floor( stim_begin_step / BIN_STEP_SIZE ) + 1;
if ( stim_end_time > 0.0 )
  stim_end_time = -0.0;
endif %%


%% get layers and layer specific analysis flags
global N_LAYERS
[layerID, layerIndex] = pvp_layerID();
num_layers = N_LAYERS;
read_spikes =  2:N_LAYERS;  %[layerIndex.l1];%list of spiking layers whose spike train are to be analyzed

%% plot flags
plot_reconstruct = read_spikes; %uimatlab;
plot_raster = read_spikes; %[layerIndex.l1];%read_spikes; %uimatlab;
plot_reconstruct_target = [];%read_spikes; %[layerIndex.l1];
plot_vmem = 1;
plot_autocorr = [layerIndex.lgn, layerIndex.lgninh, layerIndex.l1, layerIndex.l1inh];
plot_xcorr = plot_autocorr;

num_epochs = 8;

				% initialize to size of image (if known), these should be overwritten by each layer
NROWS_image=128;
NCOLS_image=128;
NROWS = NROWS_image;
NCOLS = NCOLS_image;
NFEATURES = 8;
NO = NFEATURES; % number of orientations
NK = 1; % number of curvatures
dK = 0; % spacing between curvatures (1/radius)

my_gray = [.666 .666 .666];
num_targets = 1;
fig_list = [];

				% allocate spike data structures
global SPIKE_ARRAY
SPIKE_ARRAY = [];

%%global LAYER  % current layer

				% data structures for correlation analysis
				%stft_array = cell( num_layers, 1);
min_freq = 40;
max_freq = 60;
xcorr_flag = 0;
num_eigen = 3;
num_modes = 1;
calc_power_mask = 1;
num_sig = 10;
calc_eigen = 1;
xcorr_eigenvector = cell( num_layers, num_modes, num_eigen);
xcorr_array = cell(num_layers, num_modes);
size_border_mask = 4;
border_mask = cell(num_layers, 1);
power_mask = cell(num_layers, num_modes);
num_power_mask = zeros(num_layers, num_modes);
power_array = cell( num_layers, num_modes);

				% data structures for epochs
epoch_struct = struct;
epoch_struct.begin_time = repmat(BEGIN_TIME, [num_epochs, num_layers]);
epoch_struct.end_time = repmat(END_TIME, [num_epochs, num_layers]);
epoch_struct.exclude_offset = zeros(num_epochs, num_layers);
epoch_struct.total_spikes = zeros(num_epochs, num_layers);
epoch_struct.total_steps = zeros(num_epochs, num_layers);
epoch_struct.epoch_time = zeros(1, num_layers);
epoch_struct.epoch_steps = zeros(1, num_layers);
epoch_struct.epoch_bins = zeros(1, num_layers);
time_origin = zeros(num_layers,1);

				% data structures for layer shape
num_rows = ones(num_layers,1);
num_cols = ones(num_layers,1);
num_features = ones(num_layers,1);
num_neurons = ones(num_layers,1);

				% target/clutter segmentation data structures
rate_array =  cell(num_layers,1);
psth_array =  cell(num_layers,1);
ave_target = cell(num_layers,num_targets);
ave_clutter = cell(num_layers,1);
ave_bkgrnd = cell(num_layers,1);
psth_target = cell(num_layers,num_targets);
psth_clutter = cell(num_layers,1);
psth_bkgrnd = cell(num_layers,1);
target_ndx_max = cell(num_layers, num_targets);
target_ndx_all = cell(num_layers, num_targets);
clutter_ndx_max = cell(num_layers, 1);
clutter_ndx_all = cell(num_layers, 1);
bkgrnd_ndx_max = cell(num_layers, 1);
bkgrnd_ndx_all = cell(num_layers, 1);
num_target_neurons_all = zeros(num_layers, num_targets);
num_target_neurons_max = zeros(num_layers, num_targets);
num_clutter_neurons_all = zeros(num_layers, 1);
num_clutter_neurons_max = zeros(num_layers, 1);
num_bkgrnd_neurons_max = zeros(num_layers, 1);
num_bkgrnd_neurons_all = zeros(num_layers, 1);
target_rate = cell(num_layers, num_targets);
target_rate_ndx = cell(num_layers, num_targets);
clutter_rate = cell(num_layers, 1);
clutter_rate_ndx = cell(num_layers, 1);

%% read input image segmentation info
invert_image_flag = 0;
plot_input_image = 0;
[target, clutter, image, fig_tmp] = ...
    pvp_parseTarget( image_filename, ...
		    target_filename, ...
		    invert_image_flag, ...
		    plot_input_image);
fig_list = [fig_list; fig_tmp];
disp('parse BMP -> done');

%% get firing rates for computing max ndx
for layer = read_spikes;
  disp(['analyzing layer rate: ', num2str(layer)]);
  
  %% re-initialize begin/end times for each layer
  BEGIN_TIME = epoch_struct.begin_time(1,layer);
  END_TIME = epoch_struct.end_time(1,layer);
  
  %% get adjusted begin/end times from spike array
  disp('opening sparse spikes');
  [fid, ...
   total_spikes, ...
   total_steps,...
   exclue_spikes,...
   exclude_steps, ...
   exclude_offset ] = ...
      pvp_openSparseSpikes(layer);
  fclose(fid);
  time_origin(layer) = BEGIN_TIME;
  total_time = END_TIME - BEGIN_TIME;
  disp( [layerID{layer}, ...
         ' total_time(', num2str(layer), ') = ', ...
         num2str(total_time)]);
  ave_rate(layer) = 1000 * total_spikes / ( N * total_time );
  disp( [layerID{layer}, ...
         ' ave_rate(',num2str(layer),') = ', ...
         num2str(ave_rate(layer))] );
  int_fmt = '%i9';
  disp( ['total_spikes(', num2str(layer), ') = ', ...
         num2str(total_spikes, int_fmt) ]);
  disp( ['total_steps(', num2str(layer), ') = ', ...
         num2str(total_steps, int_fmt) ]);
  
  epoch_struct.begin_time(:,layer) = BEGIN_TIME;
  epoch_struct.end_time(:,layer) = END_TIME;
  epoch_struct.exclude_offset(:,layer) = exclude_offset;
  
  %% set layer dimensions
  num_rows(layer) = NROWS;
  num_cols(layer) = NCOLS;
  num_features(layer) = NFEATURES;
  num_neurons(layer) = NFEATURES * NCOLS * NROWS;
  size_layer = [num_features(layer), num_cols(layer), num_rows(layer)];
  
  %% fit equal number of time steps and PSTH bins in each epoch
  epoch_time = floor( total_time / num_epochs );
  epoch_steps = floor( total_steps / num_epochs );
  if total_steps > num_epochs * epoch_steps
    total_steps =  num_epochs * epoch_steps;
  endif %%
  begin_step = 1;
  end_step = epoch_steps;
  time_steps = begin_step:end_step;
  
  epoch_bins = floor( epoch_steps / BIN_STEP_SIZE );
  epoch_steps = epoch_bins * BIN_STEP_SIZE;
  time_bins = (1:epoch_bins)*BIN_STEP_SIZE*DELTA_T;
  
  epoch_struct.epoch_time(layer) = epoch_time;
  epoch_struct.epoch_steps(layer) = epoch_steps;
  epoch_struct.epoch_bins(layer) = epoch_bins;
  
  stim_end_step = epoch_struct.epoch_steps(layer) - floor( stim_end_time / DELTA_T );
  stim_end_bin = epoch_struct.epoch_bins(layer) - floor( stim_end_step / BIN_STEP_SIZE );
  stim_steps = stim_begin_step : stim_end_step;
  stim_bins = stim_begin_bin : stim_end_bin;
  max_lag = min( 128/DELTA_T, fix(epoch_steps/8) );
  
  %% init rate array
  rate_array{layer} = zeros(1, N);
  
  %% init power array
  power_array{layer,1} = zeros(1, N);
  power_array{layer,2} = zeros(1, N);
  
  %% compute border mask
  border_mask{layer} = ...
      ones( NFEATURES, NCOLS, NROWS );
  border_mask{layer}(:, 1:size_border_mask, :) = 0;
  border_mask{layer}(:, NCOLS-size_border_mask:NCOLS, :) = 0;
  border_mask{layer}(:, :, 1:size_border_mask) = 0;
  border_mask{layer}(:, :, NROWS-size_border_mask:NROWS) = 0;
  border_mask{layer} = ...
      find( border_mask{layer}(:) );
  
  %% start loop over epochs
  for i_epoch = 1 : num_epochs
    disp(['i_epoch = ', num2str(i_epoch)]);
    
    %% init BEGIN/END times for each epoch
    epoch_struct.begin_time(i_epoch, layer) = ...
        time_origin(layer) + ...
        ( i_epoch - 1 ) * epoch_time;
    BEGIN_TIME = epoch_struct.begin_time(i_epoch, layer);
    epoch_struct.end_time(i_epoch,layer) = BEGIN_TIME + epoch_time;
    END_TIME = epoch_struct.end_time(i_epoch, layer);
    
    %% determine BEGIN/END times, etc for this epoch
    [fid, ...
     total_spikes, ...
     total_steps,...
     exclude_spikes,...
     exclude_steps, ...
     exclude_offset ] = ...
        pvp_openSparseSpikes(layer);
    
				%    epoch_struct.begin_time(i_epoch,layer) = BEGIN_TIME;
				%    epoch_struct.end_time(i_epoch,layer) = END_TIME;
    epoch_struct.exclude_offset(i_epoch,layer) = exclude_offset;
    epoch_struct.total_spikes(i_epoch,layer) = total_spikes;
    epoch_struct.total_steps(i_epoch,layer) = total_steps;
    disp( [layerID{layer}, ...
           ' epoch_struct.begin_time(', ...
				     num2str(i_epoch), ...
				     ',', ...
				     num2str(layer), ') = ', ...
           num2str(epoch_struct.begin_time(i_epoch,layer) )] );
    disp( [layerID{layer}, ...
           ' epoch_struct.end_time(', ...
				   num2str(i_epoch), ...
				   ',', ...
				   num2str(layer), ') = ', ...
           num2str(epoch_struct.end_time(i_epoch,layer) )] );        
    disp( [layerID{layer}, ...
           ' epoch_struct.exclude_offset(', ...
					 num2str(i_epoch), ...
					 ',', ...
					 num2str(layer), ') = ', ...
           num2str(epoch_struct.exclude_offset(i_epoch,layer) )] );        
    disp( [layerID{layer}, ...
           ' epoch_struct.total_spikes(', ...
				       num2str(i_epoch), ...
				       ',', ...
				       num2str(layer), ') = ', ...
           num2str(epoch_struct.total_spikes(i_epoch,layer) )] );        
    
    
    
    %% read spike train for this epoch
    SPIKE_ARRAY = [];
    [spike_count, ...
     step_count] = ...
        pvp_readSparseSpikes(fid, ...
			     exclude_offset, ...
			     total_spikes, ...
			     total_steps, ...
			     pvp_order);
    fclose(fid);
    
    if isempty(SPIKE_ARRAY)
      continue;
    endif %%
    if spike_count ~= total_spikes
      error('spike_count ~= total_spikes');
    endif
    if step_count ~= total_steps
      error('step_count ~= total_steps');
    endif
    
    %% accumulate rate info
    rate_array{layer} = rate_array{layer} + ...
        1000 * full( mean(SPIKE_ARRAY(stim_steps,:),1) ) / DELTA_T;
    
    %%accumulate power
    disp([layerID{layer}, ...
          ' ', 'mass_power', ...
          '(', ...
            num2str(layer), ',', num2str(i_epoch), ')']);
    layer_ndx = [1:N];
    power_method = 0;
    if power_method == 0
      [power_array_tmp, ...
       mass_autocorr_tmp] = ...
          pvp_autocorr( max_lag, ...
                       layer_ndx, ...
                       min_freq, ...
                       max_freq );
    elseif power_method == 1
      [power_array_tmp, ...
       mass_power_tmp] = ...
          pvp_power( power_win_size, ...
                    layer_ndx, ...
                    min_freq, ...
                    max_freq )
    endif %%  % power_method
    power_array{layer, 1} = power_array{layer, 1} + ...
        power_array_tmp(:,1).';
    power_array{layer, 2} = power_array{layer, 2} + ...
        power_array_tmp(:,2).';
    
    %% plot spike movie
				% original version does not work in octave, which lacks getframe, movie2avi, etc
    plot_movie = 0;
    if plot_movie
      spike_movie = pvp_movie( SPIKE_ARRAY, layer);
    endif %% % plot_movie
    
    SPIKE_ARRAY = [];
    
  endfor %% % i_epoch
  
  
  %% plot averages over epochs
				%rate_array{layer,1} = rate_array{layer,1} / num_epochs;
  disp([layerID{layer}, ': rate_array(',num2str(layer),') = ', ...
        num2str( mean( rate_array{layer,1}(stim_steps) / num_epochs ) ), 'Hz']);    
  
  
  %% reconstruct image from layer activity
				%rate_array{layer} = rate_array{layer} / num_epochs;
  size_recon = [1 num_features(layer), num_cols(layer), num_rows(layer)];
  plot_rate_reconstruction = ismember( layer, plot_reconstruct );
  if plot_rate_reconstruction
    plot_title = ...
        [layerID{layer}, ...
         ' Rate(', ...
		int2str(layer), ...
		')'];
    rate_array_tmp = rate_array{layer}/num_epochs;
				%        fig_tmp = figure;
				%        set(fig_tmp, 'Name', plot_title);
    fig_tmp = pvp_reconstruct(rate_array_tmp, ...
			      plot_title, ...
			      [], ...
			      size_recon);
    fig_list = [fig_list; fig_tmp];
  endif %%plot_rate_reconstruction
  [rate_array_tmp, rate_rank_tmp] = ...
      sort( rate_array{layer, 1}/num_epochs, 2, 'descend');
  for i_rank = [ 1:3 ] % , ceil(num_clutter_neurons(layer, 1)/2), num_clutter_neurons(layer, 1) ]
    tmp_rate = rate_array_tmp(i_rank);
    k = rate_rank_tmp(i_rank);
    [kf, kcol, krow] = ind2sub([num_features(layer), num_cols(layer), num_rows(layer)], k);
    disp(['rank:', num2str(i_rank), ...
          ', rate_array(', num2str(layer),')', ...
          num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
          num2str([k-1, kcol-1, krow-1, kf-1]) ]);
  endfor %% % i_rank
  
  
  %%plot power reconstruction
  disp( [ 'ploting Power Recon(', num2str(layer), ')'] );
  plot_power_reconstruction = ismember( layer, plot_reconstruct );
  if plot_power_reconstruction
    plot_title = ...
        [layerID{layer}, ' Peak Power(', int2str(layer), ')' ];
    fig_tmp = ...
        pvp_reconstruct(power_array{layer, 1},  plot_title);
    fig_list = [fig_list; fig_tmp];
    plot_title = ...
        [layerID{layer}, ' Ave Power(', int2str(layer), ')' ];
    fig_tmp = ...
        pvp_reconstruct(power_array{layer, 2},  plot_title);
    fig_list = [fig_list; fig_tmp];
  endif %%
  
  
  %%computer power mask
  for i_mode = 1 : num_modes  % 1 == peak, 2 = mean
    if i_mode == 1
      disp('calculating peak power mask...');
    elseif i_mode == 2
      disp('calculating ave power mask...');
    endif %%
    if calc_power_mask
      num_power_sig = 0.0;
      mean_power = mean( power_array{layer, i_mode} );
      std_power = std( power_array{layer, i_mode} );
      disp( [ 'mean_power(', num2str(layer), ') = ', ...
             num2str(mean_power), ' +/- ', num2str(std_power) ] );
      power_mask{layer, i_mode} = ...
          find( power_array{layer, i_mode} > ( mean_power + num_power_sig * std_power ) );
      power_mask{layer,i_mode} = ...
          intersect(power_mask{layer,i_mode}, border_mask{layer} );
      num_power_mask(layer,i_mode) = numel(power_mask{layer,i_mode});
      disp( ['num_power_mask(', num2str(layer), ') = ', num2str(num_power_mask(layer,i_mode)), ' > ', ...
             num2str( mean_power + num_power_sig * std_power ) ] );
      while num_power_mask(layer,i_mode) > num_sig * ( length(target) + length(clutter) )
        num_power_sig = num_power_sig + 0.5;
        power_mask{layer, i_mode} = ...
            find( power_array{layer, i_mode} > ( mean_power + num_power_sig * std_power ) );
        power_mask{layer,i_mode} = ...
            intersect(power_mask{layer,i_mode}, border_mask{layer} );
        num_power_mask(layer,i_mode) = numel(power_mask{layer,i_mode});
        disp( ['num_power_mask(', num2str(layer), ') = ', num2str(num_power_mask(layer,i_mode)), ' > ', ...
               num2str( mean_power + num_power_sig * std_power ) ] );
      endwhile %%
    else
      power_mask{layer,i_mode} = clutter_ndx_max{layer,1};
      for i_target = 1 : num_targets
        power_mask{layer,i_mode} = [power_mask{layer,i_mode}; target_ndx_max{layer, i_target}];
      endfor %% % i_target
    endif %% % calc_power_mask
    power_mask{layer,i_mode} = sort( power_mask{layer,i_mode} );
    power_mask{layer,i_mode} = ...
        intersect(power_mask{layer,i_mode}, border_mask{layer} );
    num_power_mask(layer,i_mode) = numel(power_mask{layer,i_mode});
  endfor %% % i_mode
  
				%    pvp_saveFigList( fig_list, SPIKE_PATH, 'png');
				%    fig_list = [];
				%    close all;
  
endfor %% % layer


%% Analyze spiking activity layer by layer
for layer = read_spikes;
  disp(['analyzing layer target and clutter segments: ', num2str(layer)]);
  
  %% set layer dimensions
  NROWS = num_rows(layer);
  NCOLS = num_cols(layer);
  NFEATURES = num_features(layer);
  size_layer = [num_features(layer), num_cols(layer), num_rows(layer)];
  
  stim_end_step = epoch_struct.epoch_steps(layer) - floor( stim_end_time / DELTA_T );
  stim_end_bin = epoch_struct.epoch_bins(layer) - floor( stim_end_step / BIN_STEP_SIZE );
  stim_steps = stim_begin_step : stim_end_step;
  stim_bins = stim_begin_bin : stim_end_bin;
  
  
  %%init PSTH
  for i_target = 1 : num_targets
    ave_target{layer,i_target} = zeros( epoch_steps, 1 );
    psth_target{layer,i_target} = zeros( epoch_bins, 1 );
  endfor %% % i_target
  ave_clutter{layer,1} = zeros( epoch_steps, 1 );
  psth_clutter{layer,1} = zeros( epoch_bins, 1 );
  ave_bkgrnd{layer,1} = zeros( epoch_steps, 1 );
  psth_bkgrnd{layer,1} = zeros( epoch_bins, 1 );
  
  %% init raster
  plot_raster2 = ismember( layer, plot_raster );
  
  %% segment targets/clutter for each layer
  use_max = 1;  % rate not available unless read in all epochs first
%  bkgrnd_ndx_all{layer,1} = find(ones(N,1));
%  bkgrnd_ndx_max{layer,1} = find(ones(N,1));
  for i_target = 1:num_targets
    [target_ndx_all{layer, i_target}, target_ndx_max{layer, i_target}] = ...
        pvp_image2layer( layer, target{i_target}, ...
			stim_steps, use_max, rate_array{layer}, pvp_order);
    num_target_neurons_max(layer, i_target) = ...
        length( target_ndx_max{layer, i_target} );
    num_target_neurons_all(layer, i_target) = ...
        length( target_ndx_all{layer, i_target} );
%    bkgrnd_ndx_all{layer}(target_ndx_all{layer, i_target}) = 0;
%    bkgrnd_ndx_max{layer}(target_ndx_max{layer, i_target}) = 0;
  endfor %% % i_target
  
  [clutter_ndx_all{layer}, clutter_ndx_max{layer}] = ...
      pvp_image2layer( layer, clutter, stim_steps, ...
		      use_max, rate_array{layer}, pvp_order);
  num_clutter_neurons_max(layer,1) = length( clutter_ndx_max{layer} );
  num_clutter_neurons_all(layer,1) = length( clutter_ndx_all{layer} );
  
%  bkgrnd_ndx_all{layer}(clutter_ndx_all{layer}) = 0;
%  bkgrnd_ndx_max{layer}(clutter_ndx_max{layer}) = 0;
  
%  bkgrnd_ndx_all{layer} = find(bkgrnd_ndx_all{layer});
%  bkgrnd_ndx_max{layer} = find(bkgrnd_ndx_max{layer});
%  num_bkgrnd_neurons_all(layer) = ...
%      N - sum(num_target_neurons_all(layer,:)) - num_clutter_neurons_all(layer);
%  num_bkgrnd_neurons_max(layer) = ...
%      N - sum(num_target_neurons_max(layer,:)) - num_clutter_neurons_max(layer);
  
  %% mass corr data structures
  xcorr_freqs = (1/DELTA_T)*1000*(0:2*max_lag)/(2*max_lag + 1);
  mass_target_xcorr = cell(num_targets, 1);
  mass_target_autocorr = cell(num_targets, 1);
  target_xcorr = cell(num_targets, 1);
  for i_target = 1 : num_targets
    mass_target_xcorr{i_target, 1} = zeros( 2 * max_lag + 1, 1 );
    mass_target_autocorr{i_target, 1} = zeros( 2 * max_lag + 1, 1 );
    target_xcorr{i_target, 1}  = ...
        zeros( num_target_neurons_max(layer, i_target),  ...
              num_target_neurons_max(layer, i_target), 2 );
  endfor %%
  mass_clutter_xcorr = zeros( 2 * max_lag + 1, 1 );
  mass_clutter_autocorr = zeros( 2 * max_lag + 1, 1 );
  clutter_xcorr = ...
      zeros( num_clutter_neurons_max(layer, 1),  num_clutter_neurons_max(layer, 1), 2 );
  mass_target2clutter_xcorr = cell(num_targets, 1);
  mass_target2clutter_autocorr = cell(num_targets, 1);
  target2clutter_xcorr = cell(num_targets, 1);
  for i_target = 1 : num_targets
    mass_target2clutter_xcorr{i_target, 1} = zeros( 2 * max_lag + 1, 1 );
    mass_target2clutter_autocorr{i_target, 1} = zeros( 2 * max_lag + 1, 1 );
    target2clutter_xcorr{i_target, 1} = ...
        zeros( num_target_neurons_max(layer, i_target),  ...
              num_clutter_neurons_max(layer, 1), 2 );
  endfor %%
%  calc_bkgrnd_xcorr = 0;
%  if calc_bkgrnd_xcorr
%    mass_bkgrnd_xcorr = zeros( 2 * max_lag + 1, 1 );
%    mass_bkgrnd_autocorr = zeros( 2 * max_lag + 1, 1 );
%  endif %%
  
  
  %% start loop over epochs
  for i_epoch = 1 : num_epochs
    disp(['i_epoch = ', num2str(i_epoch)]);
    
    %% init BEGIN/END times for each epoch
    BEGIN_TIME = epoch_struct.begin_time(i_epoch, layer);
    END_TIME = BEGIN_TIME;  % read 1 line
    
    %% determine BEGIN/END times, etc for this epoch
    [fid, ...
     spike_count, ...
     step_count,...
     exclude_spikes,...
     exclude_steps, ...
     exclude_offset ] = ...
        pvp_openSparseSpikes(layer);
    
    BEGIN_TIME = epoch_struct.begin_time(i_epoch, layer);
    END_TIME = epoch_struct.end_time(i_epoch, layer);
    exclude_offset = epoch_struct.exclude_offset(i_epoch, layer);
    total_spikes = epoch_struct.total_spikes(i_epoch, layer);
    total_steps = epoch_struct.total_steps(i_epoch, layer);        
    
    %% read spike train for this epoch
    SPIKE_ARRAY = [];
    [spike_count, ...
     step_count] = ...
        pvp_readSparseSpikes(fid, ...
			     exclude_offset, ...
			     total_spikes, ...
			     total_steps, ...
			     pvp_order);
    fclose(fid);
    
    if isempty(SPIKE_ARRAY)
      continue;
    endif %%
    if spike_count ~= total_spikes
      error('spike_count ~= total_spikes');
    endif
    if step_count ~= total_steps
      error('step_count ~= total_steps');
    endif
    
    %% compute average activity for target and clutter
    for i_target = 1:num_targets
      ave_tmp = ...
          full( 1000*sum(SPIKE_ARRAY(:,target_ndx_all{layer, i_target}),2) / ...
               ( num_target_neurons_all(layer, i_target) + ( num_target_neurons_all(layer, i_target)==0 ) ) );
      ave_target{layer,i_target} = ave_target{layer,i_target} + ave_tmp(1:epoch_steps);
      psth_target{layer,i_target} = psth_target{layer,i_target} + ...
          mean( reshape( ave_tmp(1:epoch_steps), ...
			BIN_STEP_SIZE, epoch_bins  ), 1)';
    endfor %% % i_target
    ave_tmp = ...
        full( 1000*sum(SPIKE_ARRAY(:,clutter_ndx_all{layer}),2) / ...
             ( num_clutter_neurons_all(layer) + ( num_clutter_neurons_all(layer)==0 ) ) );
    ave_clutter{layer} =  ave_clutter{layer} + ave_tmp(1:epoch_steps);
    psth_clutter{layer,1} = psth_clutter{layer,1} + ...
        mean( reshape( ave_tmp(1:epoch_steps), ...
		      BIN_STEP_SIZE, epoch_bins  ), 1)';
    ave_tmp = ...
        full( 1000*sum(SPIKE_ARRAY,2) / N );
    ave_bkgrnd{layer} = ave_bkgrnd{layer} + ave_tmp(1:epoch_steps);
    psth_bkgrnd{layer,1} = ...
        mean( reshape( ave_tmp(1:epoch_steps), ...
		      BIN_STEP_SIZE, epoch_bins  ), 1)';
    
    %% raster plot
    raster_epoch = [1, num_epochs];
    if ismember( layer, plot_raster ) && ismember( i_epoch, raster_epoch )
      plot_title = ...
          [layerID{layer}, ...
           ' Raster',...
           '(', ...
             int2str(layer)', ...
             ',', ...
             int2str(i_epoch), ...
             ')'];
      fig_tmp = figure('Name',plot_title);
      fig_list = [fig_list; fig_tmp];
      axis([BEGIN_TIME END_TIME 0 num_neurons(layer)])
      axis normal
      hold on
      [spike_time, spike_id] = ...
          find(SPIKE_ARRAY);
      spike_time = spike_time*DELTA_T + BEGIN_TIME;
      lh = plot(spike_time, spike_id, '.k');
      axis normal
      set(lh,'Color',my_gray);
      
      for i_target=1:num_targets
        [spike_time, spike_id] = ...
            find(SPIKE_ARRAY(:, ...
			     target_ndx_all{layer, i_target}));
        spike_time = spike_time*DELTA_T + BEGIN_TIME;
        plot(spike_time, target_ndx_all{layer, i_target}(spike_id), '.r');
        axis normal
      endfor %% % i_target
      
      [spike_time, spike_id] = ...
          find(SPIKE_ARRAY(:, ...
			   clutter_ndx_all{layer}));
      spike_time = spike_time*DELTA_T + BEGIN_TIME;
      plot(spike_time*DELTA_T, clutter_ndx_all{layer}(spike_id), '.b');
      axis normal
    endif %%  % plot_raster
    
    %% mass correlation analysis
    plot_autocorr2 = ismember( layer, plot_autocorr );
    if plot_autocorr2
      
      %% accumulate massXCorr of target and distractor and background elements separately
      disp([layerID{layer}, ...
            ' ', 'mass_target_xcorr', ...
            '(', ...
              num2str(layer), ',', num2str(i_epoch), ')']);
      is_auto = 1;
      for i_target = 1 : num_targets
        [mass_xcorr_tmp, ...
         mass_autocorr_tmp, ...
         mass_xcorr_mean, ...
         mass_xcorr_std, ...
         mass_xcorr_lags, ...
         xcorr_tmp, ...
         xcorr_dist, ...
         min_freq_ndx, ...
         max_freq_ndx] = ...
            pvp_xcorr2( SPIKE_ARRAY(stim_steps, target_ndx_max{layer, i_target}), ...
                       [], ...
                       max_lag, ...
                       target_ndx_max{layer, i_target}, ...
                       size_layer, ...
                       target_ndx_max{layer, i_target}, ...
                       size_layer, ...
                       is_auto, ...
                       min_freq, ...
                       max_freq, ...
                       xcorr_flag);
        mass_target_xcorr{i_target} = mass_target_xcorr{i_target} + ...
            mass_xcorr_tmp;
        mass_target_autocorr{i_target} = mass_target_autocorr{i_target} + ...
            mass_autocorr_tmp;
        target_xcorr{i_target} = target_xcorr{i_target} + ...
            xcorr_tmp;
      endfor %%
      
      disp([layerID{layer}, ...
            ' ', 'mass_clutter_xcorr', ...
            '(', ...
              num2str(layer), ',', num2str(i_epoch), ')']);
      is_auto = 1;
      [mass_xcorr_tmp, ...
       mass_autocorr_tmp, ...
       mass_xcorr_mean, ...
       mass_xcorr_std, ...
       mass_xcorr_lags, ...
       xcorr_tmp, ...
       xcorr_dist, ...
       min_freq_ndx, ...
       max_freq_ndx] = ...
          pvp_xcorr2(  SPIKE_ARRAY(stim_steps, clutter_ndx_max{layer, 1}), ...
		     [], ...
		     max_lag, ...
		     clutter_ndx_max{layer, 1}, ...
		     size_layer, ...
		     clutter_ndx_max{layer, 1}, ...
		     size_layer, ...
		     is_auto, ...
		     min_freq, ...
		     max_freq, ...
		     xcorr_flag);
      mass_clutter_xcorr = mass_clutter_xcorr + ...
          mass_xcorr_tmp;
      mass_clutter_autocorr = mass_clutter_autocorr + ...
          mass_autocorr_tmp;
      clutter_xcorr = clutter_xcorr + ...
          xcorr_tmp;
      
%%      if 0%calc_bkgrnd_xcorr
%%        [mass_xcorr_tmp, bdgrnd_lag] = ...
%%            xcorr( ave_bkgrnd{layer,1}(stim_steps)', [], max_lag, 'unbiased' );
%%        mass_autocorr_tmp = mass_bkgrnd_xcorr;
%%        mass_xcorr = mass_bkgrnd_xcorr + ...
%%            mass_bkgrnd_xcorr_tmp;
%%        mass_autocorr = mass_bkgrnd_autocorr + ...
%%            mass_bkgrnd_autocorr_tmp;
%%      endif %%
      
      disp([layerID{layer}, ...
            ' ', 'mass_target2clutter_xcorr', ...
            '(', ...
              num2str(layer), ',', num2str(i_epoch), ')']);
      is_auto = 0;
      for i_target = 1 : num_targets
        [mass_xcorr_tmp, ...
         mass_autocorr_tmp, ...
         mass_xcorr_mean, ...
         mass_xcorr_std, ...
         mass_xcorr_lags, ...
         xcorr_tmp, ...
         xcorr_dist, ...
         min_freq_ndx, ...
         max_freq_ndx] = ...
            pvp_xcorr2( SPIKE_ARRAY(stim_steps, target_ndx_max{layer, i_target}), ...
                       SPIKE_ARRAY(stim_steps, clutter_ndx_max{layer, 1}), ...
                       max_lag, ...
                       target_ndx_max{layer, 1}, ...
                       size_layer, ...
                       clutter_ndx_max{layer, 1}, ...
                       size_layer, ...
                       is_auto, ...
                       min_freq, ...
                       max_freq, ...
                       xcorr_flag);
        mass_target2clutter_xcorr{i_target} = ...
	    mass_target2clutter_xcorr{i_target} + ...
            mass_xcorr_tmp;
        mass_target2clutter_autocorr{i_target} = ...
	    mass_target2clutter_autocorr{i_target} + ...
            mass_autocorr_tmp;
        target2clutter_xcorr{i_target} = target2clutter_xcorr{i_target} + ...
            xcorr_tmp;
      endfor %% % i_target
      
    endif %%
    
  endfor %% % i_epoch
  
  
  %% plot averages over epochs
  %% compute averages for different image segments
  for i_target = 1 : num_targets
				%ave_target{layer,i_target} = ave_target{layer,i_target} / num_epochs;
    disp([layerID{layer}, ': ave_target(',num2str(layer),',', num2str(i_target), ') = ', ...
          num2str( mean( ave_target{layer,i_target}(stim_steps) / num_epochs ) ), 'Hz']);
  endfor
				%ave_clutter{layer,1} = ave_clutter{layer,1} / num_epochs;
  disp([layerID{layer}, ': ave_clutter(',num2str(layer),') = ', ...
        num2str( mean( ave_clutter{layer,1}(stim_steps) / num_epochs ) ), 'Hz']);
				%ave_bkgrnd{layer,1} = ave_bkgrnd{layer,1} / num_epochs;
  disp([layerID{layer}, ': ave_bkgrnd(',num2str(layer),') = ', ...
        num2str( mean( ave_bkgrnd{layer,1}(stim_steps) / num_epochs ) ), 'Hz']);
  
  
  %% plot PSTH
  plot_title = [layerID{layer}, ' PSTH(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
  set(gca, 'plotboxaspectratiomode', 'manual');
  axis normal
  for i_target = 1 : num_targets
				%psth_target{layer,i_target} =  psth_target{layer,i_target} / num_epochs;
    lh = plot(time_bins, psth_target{layer,i_target} / num_epochs, '-r');
    set(lh, 'LineWidth', 2);
    hold on
  endfor %% % i_target
				%psth_clutter{layer,1} = psth_clutter{layer,1} / num_epochs;
  lh = plot(time_bins, psth_clutter{layer,1} / num_epochs, '-b');
  set(lh, 'LineWidth', 2);
				%psth_bkgrnd{layer,1} = psth_bkgrnd{layer,1} / num_epochs;
  lh = plot(time_bins, psth_bkgrnd{layer,1} / num_epochs, '-k');
  set(lh, 'LineWidth', 2);
  set(lh, 'Color', my_gray);
  
  
  %% reconstruct image segments from layer activity
				%rate_array{layer} = rate_array{layer} / num_epochs;
  size_recon = [1 num_features(layer), num_cols(layer), num_rows(layer)];
  for i_target = 1:num_targets
    target_rate{layer, i_target} = ...
        rate_array{layer}(1,target_ndx_all{layer, i_target}) / num_epochs;
    rate_array_tmp = ...
        sparse(1, target_ndx_all{layer, i_target}, target_rate{layer, i_target}, 1 , N, num_target_neurons_all(layer, i_target) );
    if ismember( layer, plot_reconstruct_target )
      plot_title = ...
          [layerID{layer}, ...
           ' Target(', ...
                    int2str(layer), ...
                    ',', ...
                    int2str(i_target), ')'];
      fig_tmp = ...
          pvp_reconstruct(full(rate_array_tmp), ...
			  plot_title, ...
			  [], ...
			  size_recon );
      fig_list = [fig_list; fig_tmp];
    endif %% %  reconstruc target/clutter
    [target_rate{layer, i_target}, target_rate_ndx{layer, i_target}] = ...
        sort( target_rate{layer, i_target}, 2, 'descend');
    for i_rank = [ 1:3 ] % , ceil(num_target_neurons(layer, i_target)/2), num_target_neurons(layer, i_target) ]
      tmp_rate = target_rate{layer, i_target}(i_rank);
      tmp_ndx = target_rate_ndx{layer, i_target}(i_rank);
      k = target_ndx_all{layer, i_target}(tmp_ndx);
      [kf, kcol, krow] = ind2sub([num_features(layer), num_cols(layer), num_rows(layer)], k);
      disp(['rank:',num2str(i_rank),...
            ', target_rate(',num2str(layer),', ', num2str(i_target), ')', ...
            num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
            num2str([k-1, kcol-1, krow-1, kf-1]) ]);
    endfor %% % i_rank
  endfor %% % i_target
  clutter_rate{layer, 1} = rate_array{layer}(1,clutter_ndx_all{layer, 1});
  [clutter_rate{layer, 1}, clutter_rate_ndx{layer, 1}] = ...
      sort( clutter_rate{layer, 1}, 2, 'descend');
  rate_array_tmp = ...
      sparse(1, clutter_ndx_all{layer, 1}, clutter_rate{layer, 1}, 1 , N, num_clutter_neurons_all(layer, 1) );
  if ismember( layer, plot_reconstruct_target )
    plot_title = ...
        [layerID{layer}, ...
         ' Clutter(', ...
		   int2str(layer), ...
		   ')'];
    fig_tmp = ...
        pvp_reconstruct(full(rate_array_tmp), ...
			plot_title, ...
			[], ...
			size_recon);
    fig_list = [fig_list; fig_tmp];
  endif %%
  for i_rank = [ 1:3 ] % , ceil(num_clutter_neurons(layer, 1)/2), num_clutter_neurons(layer, 1) ]
    tmp_rate = clutter_rate{layer, 1}(i_rank);
    tmp_ndx = clutter_rate_ndx{layer, 1}(i_rank);
    k = clutter_ndx_all{layer, 1}(tmp_ndx);
    [kf, kcol, krow] = ind2sub([num_features(layer), num_cols(layer), num_rows(layer)], k);
    disp(['rank:',num2str(i_rank),...
          ', clutter_rate(', num2str(layer),')', ...
          num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
          num2str([k-1, kcol-1, krow-1, kf-1]) ]);
  endfor %% % i_rank
  
  
  plot_autocorr2 = ismember( layer, plot_autocorr );
  if ~plot_autocorr2
    continue;
  endif %%
  
  
  
  %% plot MassXCorr
  plot_title = [layerID{layer}, ' MassXCorr(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
				%mass_target_xcorr{1} = mass_target_xcorr{1} * ( 1000 / DELTA_T )^2 / num_epochs;
  for i_target = 1 : num_targets
    lh_target = plot((-max_lag:max_lag)*DELTA_T, squeeze( mass_target_xcorr{i_target} * ( 1000 / DELTA_T )^2 / num_epochs ), '-r');
    set(lh_target, 'LineWidth', 2);
    hold on
  endfor %% % i_target
				%mass_clutter_xcorr = mass_clutter_xcorr * ( 1000 / DELTA_T )^2 / num_epochs;
  lh_clutter = plot((-max_lag:max_lag)*DELTA_T, squeeze( mass_clutter_xcorr * ( 1000 / DELTA_T )^2 / num_epochs ), '-b');
  set(lh_clutter, 'LineWidth', 2);
				%mass_target2clutter_xcorr = mass_target2clutter_xcorr * ( 1000 / DELTA_T )^2 / num_epochs;
  for i_target = 1 : num_targets
    lh_target2clutter = plot((-max_lag:max_lag)*DELTA_T, squeeze( mass_target2clutter_xcorr{i_target} * ( 1000 / DELTA_T )^2 / num_epochs ), '-g');
    set(lh_target2clutter, 'LineWidth', 2);
    axis tight
  endfor %% % i_target
%%  if calc_bkgrnd_xcorr
%%				%mass_bkgrnd_xcorr = mass_bkgrnd_xcorr / num_epochs;
%%    lh_bkgrnd = plot((-max_lag:max_lag)*DELTA_T, squeeze( mass_bkgrnd_xcorr / num_epochs ), '-k');
%%    set(lh_bkgrnd, 'Color', my_gray);
%%    set(lh_bkgrnd, 'LineWidth', 2);
%%  endif %%
  
  
  
  %% plot MassAutoCorr
  plot_title = [layerID{layer}, ' MassAutoCorr(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
				%mass_target_autocorr{1} = mass_target_autocorr{1} * ( 1000 / DELTA_T )^2 / num_epochs;
  for i_target = 1 : num_targets
    lh_target = plot((-max_lag:max_lag)*DELTA_T, squeeze(mass_target_autocorr{i_target} * ( 1000 / DELTA_T )^2 / num_epochs ), '-r');
    set(lh_target, 'LineWidth', 2);
    hold on
  endfor %% % i_target
				%mass_clutter_autocorr = mass_clutter_autocorr * ( 1000 / DELTA_T )^2 / num_epochs;
  lh_clutter = plot((-max_lag:max_lag)*DELTA_T, squeeze( mass_clutter_autocorr * ( 1000 / DELTA_T )^2 / num_epochs ), '-b');
  set(lh_clutter, 'LineWidth', 2);
%%  if calc_bkgrnd_xcorr
%%    lh_bkgrnd = plot((-max_lag:max_lag)*DELTA_T, squeeze( mass_bkgrnd_autocorr / num_epochs ), '-k');
%%    set(lh_bkgrnd, 'Color', my_gray);
%%    set(lh_bkgrnd, 'LineWidth', 2);
%%  endif %%
  axis tight
  
  
  
  %%plot cross power
  %%disp( [ 'computing XPower(', num2str(layer), ')'] );
  plot_title = ...
      [layerID{layer}, ' XPower(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
  min_ndx = find(xcorr_freqs > 128, 1,'first');
  for i_target = 1 : num_targets
    fft_xcorr_tmp = fft( squeeze( mass_target_xcorr{i_target} * ( 1000 / DELTA_T )^2 / num_epochs ) );
    lh_target = plot(xcorr_freqs(2:min_ndx),...
		     abs( fft_xcorr_tmp(2:min_ndx) ), '-r');
    set(lh_target, 'LineWidth', 2);
    hold on
  endfor %% % i_target
  fft_xcorr_tmp = fft( squeeze( mass_clutter_xcorr * ( 1000 / DELTA_T )^2 / num_epochs ) );
  lh_clutter = plot(xcorr_freqs(2:min_ndx),...
		    abs( fft_xcorr_tmp(2:min_ndx) ), '-b');
  set(lh_clutter, 'LineWidth', 2);
%%  if calc_bkgrnd_xcorr %exist('xcorr')
%%    fft_xcorr_tmp = fft( squeeze( mass_bkgrnd_xcorr / num_epochs ) );
%%    lh_bkgrnd = plot(xcorr_freqs(2:min_ndx),...
%%		     abs( fft_xcorr_tmp(2:min_ndx) ), '-k');
%%    set(lh_bkgrnd, 'LineWidth', 2);
%%    set(lh_bkgrnd, 'Color', my_gray);
%%  endif %% % calc_bkgrnd_xcorr
  for i_target = 1 : num_targets
    fft_xcorr_tmp = fft( squeeze( mass_target2clutter_xcorr{i_target} * ( 1000 / DELTA_T )^2 / num_epochs ) );
    lh_target2clutter = plot(xcorr_freqs(2:min_ndx),...
			     abs( fft_xcorr_tmp(2:min_ndx)), '-g');
    set(lh_target2clutter, 'LineWidth', 2);
  endfor %% % i_target
  axis tight
  
  
  %% plot auto power
  plot_title = ...
      [layerID{layer}, ' AutoPower(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
  for i_target = 1 : num_targets
    fft_autocorr_tmp = fft( squeeze( mass_target_autocorr{i_target} * ( 1000 / DELTA_T )^2 / num_epochs ) );
    lh_target = plot(xcorr_freqs(2:min_ndx),...
		     abs( fft_autocorr_tmp(2:min_ndx)), '-r');
    set(lh_target, 'LineWidth', 2);
    hold on
  endfor %% % i_target
  fft_autocorr_tmp = fft( squeeze( mass_clutter_autocorr * ( 1000 / DELTA_T )^2 / num_epochs ) );
  lh_clutter = plot(xcorr_freqs(2:min_ndx),...
		    abs( fft_autocorr_tmp(2:min_ndx)), '-b');
  set(lh_clutter, 'LineWidth', 2);
%%  if calc_bkgrnd_xcorr %exist('xcorr')
%%    fft_autocorr_tmp = fft( squeeze( mass_bkgrnd_autocorr / num_epochs ) );
%%    lh_bkgrnd = plot(xcorr_freqs(2:min_ndx),...
%%		     abs( fft_autocorr_tmp(2:min_ndx)), '-k');
%%    set(lh_bkgrnd, 'LineWidth', 2);
%%    set(lh_bkgrnd, 'Color', my_gray);
%%  endif %% % calc_bkgrnd_xcorr
  axis tight
  figure(fig_tmp);
  
endfor %% % layer


%% Eigen analysis
if calc_eigen
  disp('beginning eigen analysis...');
else
  pvp_saveFigList( fig_list, SPIKE_PATH, 'png');
  error('abort eigen analysis');
endif
for layer = read_spikes;
  plot_xcorr2 = ismember( layer, plot_xcorr );
  if ~plot_xcorr2
    continue;
  endif %%
  disp(['layer( ', num2str(layer), ')']);
  
  %% set layer dimensions
  NROWS = num_rows(layer);
  NCOLS = num_cols(layer);
  NFEATURES = num_features(layer);
  size_layer = [num_features(layer), num_cols(layer), num_rows(layer)];

  stim_end_step = epoch_struct.epoch_steps(layer) - floor( stim_end_time / DELTA_T );
  stim_end_bin = epoch_struct.epoch_bins(layer) - floor( stim_end_step / BIN_STEP_SIZE );
  stim_steps = stim_begin_step : stim_end_step;
  stim_bins = stim_begin_bin : stim_end_bin;
  
  mass_xcorr = cell(2,1);
  mass_autocorr = cell(2,1);
  xcorr_array = cell(2,1);
  for i_mode = 1 : num_modes
    mass_xcorr{i_mode,1} = zeros( 2 * max_lag + 1, 1 );
    mass_autocorr{i_mode,1} = zeros( 2 * max_lag + 1, 1 );
    xcorr_array{i_mode,1} = ...
        zeros( num_power_mask(layer, i_mode) );
  endfor %%  %i_mode
  
  for i_epoch = 1 : num_epochs
    disp(['i_epoch = ', num2str(i_epoch)]);
    
    %% init BEGIN/END times for each epoch
    BEGIN_TIME = epoch_struct.begin_time(i_epoch, layer);
    END_TIME = BEGIN_TIME;  % read 1 line
    
    [fid, ...
     spike_count, ...
     step_count,...
     exclude_spikes,...
     exclude_steps, ...
     exclude_offset ] = ...
        pvp_openSparseSpikes(layer);
    
    BEGIN_TIME = epoch_struct.begin_time(i_epoch, layer);
    END_TIME = epoch_struct.end_time(i_epoch, layer);
    exclude_offset = epoch_struct.exclude_offset(i_epoch, layer);
    total_spikes = epoch_struct.total_spikes(i_epoch, layer);
    total_steps = epoch_struct.total_steps(i_epoch, layer);
    
    %% read spike train for this epoch
    SPIKE_ARRAY = [];
    [spike_count, ...
     step_count] = ...
        pvp_readSparseSpikes(fid, ...
			     exclude_offset, ...
			     total_spikes, ...
			     total_steps, ...
			     pvp_order);
    fclose(fid);
    
    if isempty(SPIKE_ARRAY)
      continue;
    endif %%
    if spike_count ~= total_spikes
      error('spike_count ~= total_spikes');
    endif
    if step_count ~= total_steps
      error('step_count ~= total_steps');
    endif
    
    %% compute/accumulate xcorr
    %% extract scalar pairwise correlations
    for i_mode = 1 : num_modes  % 1 = peak, 2 = mean
      
      disp( ['computing xcorr', ...
             '(', ...
               num2str(layer), ...
               ',', ...
               num2str(i_mode), ...
               ')'] );
      size_layer = [num_features(layer), num_cols(layer), num_rows(layer) ];
				%plot_interval = fix( num_power_mask(layer, i_mode)^2 / 1 );
      xcorr_flag = 1;
      is_auto = 1;
      [mass_xcorr_tmp, ...
       mass_autocorr_tmp, ...
       mass_xcorr_mean, ...
       mass_xcorr_std, ...
       mass_xcorr_lags, ...
       xcorr_array_tmp, ...
       xcorr_dist, ...
       min_freq_ndx, ...
       max_freq_ndx] = ...
          pvp_xcorr2(SPIKE_ARRAY(stim_steps, power_mask{layer, i_mode}), ...
                     SPIKE_ARRAY(stim_steps, power_mask{layer, i_mode}), ...
                     max_lag, ...
                     power_mask{layer, i_mode}, ...
                     size_layer, ...
                     power_mask{layer, i_mode}, ...
                     size_layer, ...
                     is_auto,  ...
                     min_freq, ...
                     max_freq, ...
                     xcorr_flag);
      mass_xcorr{i_mode, 1} =  mass_xcorr{i_mode, 1} + ...
          ( mass_xcorr_tmp - mass_xcorr_mean );
      mass_autocorr{i_mode, 1} =  mass_autocorr{i_mode, 1} + ...
          ( mass_autocorr_tmp - mass_xcorr_mean );
      xcorr_array{i_mode, 1} = xcorr_array{i_mode, 1} + ...
          xcorr_array_tmp(:, :, i_mode);
      
    endfor %% % i_mode
    
    SPIKE_ARRAY = [];
    
  endfor %% % i_epoch
  
  
  
  %% plot mass xcorr
  for i_mode = 1 : num_modes  % 1 = peak, 2 = mean
    
    plot_title = ...
        [layerID{layer}, ...
         ' mass xcorr', ...
         '(', ...
           num2str(layer), ',',  ...
           num2str(i_mode), ...
           ')'];
    fig_tmp = figure('Name',plot_title);
    fig_list = [fig_list; fig_tmp];
    plot( (-max_lag : max_lag)*DELTA_T, mass_xcorr{i_mode, 1} * ( 1000 / DELTA_T )^2 / num_epochs, '-k');
    
    plot_title = ...
        [layerID{layer}, ...
         ' fft mass xcorr', ...
         '(', ...
           num2str(layer), ',',  ...
           num2str(i_mode), ...
           ')'];
    fig_tmp = figure('Name',plot_title);
    fig_list = [fig_list; fig_tmp];
    fft_xcorr_tmp = real( fft( squeeze( mass_xcorr{i_mode, 1} * ( ( 1000 / DELTA_T )^2 / num_epochs ) ) ) );
    fft_xcorr_tmp = fft_xcorr_tmp / ( fft_xcorr_tmp(1) + fft_xcorr_tmp(1) > 0 );
    lh_fft_xcorr = plot(xcorr_freqs(2:min_ndx),...
			abs( fft_xcorr_tmp(2:min_ndx) ), '-b');
    set(lh_clutter, 'LineWidth', 2);
    lh = line( [xcorr_freqs(min_freq_ndx) xcorr_freqs(min_freq_ndx)], [0 1] );
    lh = line( [xcorr_freqs(max_freq_ndx) xcorr_freqs(max_freq_ndx)], [0 1] );
    
    plot_title = ...
        [layerID{layer}, ...
         ' mass autocorr', ...
         '(', ...
           num2str(layer), ',',  ...
           num2str(i_mode), ...
           ')'];
    fig_tmp = figure('Name',plot_title);
    fig_list = [fig_list; fig_tmp];
    plot( (-max_lag : max_lag)*DELTA_T, mass_autocorr{i_mode, 1} * ( 1000 / DELTA_T )^2 / num_epochs, '-k');
    
  endfor %% % i_mode
  
  
  %%find eigen vectors
  for i_mode = 1 : num_modes  % 1 = peak, 2 = mean
    
    size_recon = ...
        [1, num_features(layer), num_cols(layer), num_rows(layer) ];
    disp(['computing eigenvectors(', num2str(layer),')']);
    options.issym = 1;
    mean_xcorr_array = mean( xcorr_array{i_mode}(:) );
    [eigen_vec, eigen_value, eigen_flag] = ...
        eigs( ( (1/2)*(xcorr_array{i_mode} + ...
		       xcorr_array{i_mode}') - ...
               mean_xcorr_array ) / ...
             num_epochs, num_eigen, 'lm', options);
    [sort_eigen, sort_eigen_ndx] = sort( diag( eigen_value ), 'descend' );
    for i_vec = 1:num_eigen
      plot_title = ...
          [layerID{layer}, ...
           'Eigen', ...
           '(', ...
             num2str(layer), ',', ...
             num2str(i_mode), ',', ...
             num2str(i_vec), ')'];
      disp([layerID{layer}, ...
            ' eigenvalues', ...
            '(', num2str(layer), ',' num2str(i_mode), ',', num2str(i_vec),') = ', ...
            num2str(eigen_value(i_vec,i_vec))]);
      xcorr_eigenvector{layer, i_mode, i_vec} = eigen_vec(:, sort_eigen_ndx(i_vec));
      eigen_vec_tmp = ...
          sparse( power_mask{layer, i_mode}, ...
                 1, ...
                 real( eigen_vec(:, sort_eigen_ndx(i_vec) ) ), ...
                 num_neurons(layer), ...
                 1, ...
                 num_power_mask(layer, i_mode));
      if mean(eigen_vec_tmp(:)) < 0
        eigen_vec_tmp = -eigen_vec_tmp;
      endif %%
      fh_tmp = ...
          pvp_reconstruct( full(eigen_vec_tmp), plot_title, [], ...
			  size_recon);
      fig_list = [fig_list; fig_tmp];
    endfor %% % i_vec
    
  endfor %% % i_mode
  
  pvp_saveFigList( fig_list, SPIKE_PATH, 'png');
  close all;
  fig_list = [];
  
  SPIKE_ARRAY = [];
  
endfor %% % layer


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
  endfor %% % i_patch
endfor %% % i_conn


%%read membrane potentials from point probes
if plot_vmem
  disp('plot_vmem')
  
				% TODO: the following info should be read from a pv ouptut file
  vmem_file_list = {'Vmem_LGNa1.txt', ...
		    'Vmem_LGNc1.txt', ...
		    'Vmem_LGNInhFFa1.txt', ...
		    'Vmem_LGNInhFFc1.txt', ...
		    'Vmem_LGNInha1.txt', ...
		    'Vmem_LGNInhc1.txt', ...
		    'Vmem_V1a1.txt', ...
		    'Vmem_V1c1.txt', ...
		    'Vmem_V1InhFFa1.txt', ...
		    'Vmem_V1InhFFc1.txt', ...
		    'Vmem_V1Inha1.txt', ...
		    'Vmem_V1Inhc1.txt'};
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
  layer = read_spikes( find(read_spikes, 1) );
  BEGIN_TIME = epoch_struct.begin_time(1,layer);
  END_TIME = epoch_struct.end_time(2,layer);
  for i_vmem = 1 : vmem_skip : num_vmem_files
				%    vmem_layer = vmem_layers(i_vmem);
				%   if ( ~ismember( vmem_layer, read_spikes ) )
				%     continue; % NROWS = 1, NFEATURES = 1;
				%   endif %%
				%    NROWS = num_rows(vmem_layer);
				%    NFEATURES = num_features(vmem_layer);
    [vmem_time{i_vmem}, ...
     vmem_G_E{i_vmem}, ...
     vmem_G_I{i_vmem}, ...
     vmem_G_IB{i_vmem}, ...
     vmem_V{i_vmem}, ...
     vmem_Vth{i_vmem}, ...
     vmem_a{i_vmem} ] = ...
        ptprobe_readV(vmem_file_list{i_vmem});
				% if pvp_order
				%   vmem_index = ( vmem_row * num_cols(vmem_layer) + vmem_col ) * num_features(vmem_layers) + vmem_feature;
				% endif %%
    vmem_start = 1;
    vmem_stop = length(vmem_time{i_vmem});
    if isempty(vmem_stop)
      vmem_stop = END_TIME;
    endif
    plot_title = [ 'Vmem data: ', vmem_file_list{i_vmem} ];
    fh = figure('Name', plot_title);
    fig_list = [fig_list; fh];
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
  endfor %% % i_vmem
endif %% %plot_vmem



%% plot psth's of all layers together
plot_rates = length(read_spikes) > 1;
if plot_rates
  plot_title = ['PSTH target pixels'];
  fig_tmp = ...
      figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
  hold on
  co = get(gca,'ColorOrder');
  lh = zeros(num_layers,1);
  for layer = read_spikes
    lh(layer) = plot((time_bins)*BIN_STEP_SIZE, ...
		     psth_target{layer,i_target}(time_bins), '-r');
    set(lh(layer),'Color',co(1+mod(layer,size(co,1)),:));
    set(lh(layer),'LineWidth',2);
  endfor %%
  legend_str = ...
      {'retina  '; ...
       'LGN     '; ...
       'LGNInhFF'; ...
       'LGNInh  '; ...
       'V1      '; ...
       'V1InhFF '; ...
       'V1Inh   '};
				%    if uimatlab
				%        leg_h = legend(lh(1:num_layers), legend_str);
				%    elseif uioctave
  legend(legend_str);
				%    endif %%
  fig_list = [fig_list; fig_tmp];
endif %%

pvp_saveFigList( fig_list, SPIKE_PATH, 'png');
