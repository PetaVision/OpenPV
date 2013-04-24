%%
close all
clear all
more off

start_timer = time;

				% set paths, may not be applicable to all octave installations
				% pvp_matlabPath;

				% if ( uioctave )
%%if exist('setenv')
%%  setenv('GNUTERM', 'x11');
%%endif %%
				% endif %%

global NUM_PROCS 
NUM_PROCS = 4;    

global parallel_flag  
parallel_flag = 1;

				% Make the following global parameters available to all functions for convenience.
global N_image NROWS_image NCOLS_image
global N NROWS NCOLS % for the current layer
global NFEATURES  % for the current layer
global NO NK dK % for the current layer
global BIN_STEP_SIZE DELTA_T
global BEGIN_TIME END_TIME
global STIM_BEGIN_TIME STIM_END_TIME
global STIM_BEGIN_STEP STIM_END_STEP
global STIM_BEGIN_BIN STIM_END_BIN
global NUM_BIN_PARAMS
global NUM_WGT_PARAMS
NUM_BIN_PARAMS = 20;
NUM_WGT_PARAMS = 6;

global FLAT_ARCH_FLAG
FLAT_ARCH_FLAG = 1;

global SPIKING_FLAG
SPIKING_FLAG = 1;

workspace_path = ['../../../../Documents/workspace/'];
project_path = [workspace_path, 'BIDS/'];

global OUTPUT_PATH SPIKE_PATH
SPIKE_PATH = [project_path, 'output/'];
OUTPUT_PATH = [project_path, 'output/petavisionout/'];

global MOVIE_PATH
MOVIE_PATH = [OUTPUT_PATH, "Movie"];
mkdir(MOVIE_PATH);

%%image_path = ['amoebaX2/256_png/4/'];
image_path = [project_path, 'input/'];
image_filename = [image_path 'stimulus1.png'];
target_filename{1} = [image_path 'stimulus1.png'];

global pvp_order
pvp_order = 1;

%% set duration of simulation, if known (determined automatically otherwise)
BEGIN_TIME = 0.0;  % (msec) start analysis here, used to exclude start up artifacts
END_TIME = 24000.0;

%% stim begin/end times (msec) relative to begining/end of each epoch
STIM_BEGIN_TIME = 0.0;  % relative to begining of epoch, must be > 0
STIM_END_TIME = -0.0;  % relative to end of epoch, must be <= 0
BIN_STEP_SIZE = 5.0;  % (msec) used for all rate calculations
DELTA_T = 1.0; % msec
if ( STIM_BEGIN_TIME > 0.0 )
  STIM_BEGIN_TIME = 0.0;
endif %%
STIM_BEGIN_STEP = floor( STIM_BEGIN_TIME / DELTA_T ) + 1;
STIM_BEGIN_BIN = floor( STIM_BEGIN_STEP / BIN_STEP_SIZE ) + 1;
if ( STIM_END_TIME > 0.0 )
  STIM_END_TIME = -0.0;
endif %%
STIM_END_STEP = -floor( STIM_END_TIME / DELTA_T );
STIM_END_BIN = -floor( STIM_END_STEP / BIN_STEP_SIZE );


%% get layers and layer specific analysis flags
global N_LAYERS
[layerID, layerIndex] = pvp_layerID();
num_layers = N_LAYERS;
read_spikes = [layerIndex.ganglion, layerIndex.retina];% list of spiking layers whose spike train are to be analyzed

%% plot flags
plot_reconstruct = read_spikes; 
plot_raster = read_spikes; 
plot_movie = read_spikes; 
plot_reconstruct_target = [];
plot_vmem = 1;
plot_autocorr = ...
    [layerIndex.retina
     %layerIndex.lgn, ...
     %layerIndex.lgninh, ...
     %layerIndex.s1, ...
     %layerIndex.s1inh, ...
     %layerIndex.c1, ...
     %layerIndex.c1inh, ...
     %layerIndex.h1, ...
     %layerIndex.h1inh
	 ]; %%read_spikes;% 
plot_xcorr = plot_autocorr;

global my_gray
my_gray = [.666 .666 .666];
fig_list = [];

				% target/clutter segmentation data structures
target_struct = struct;
num_targets = 1;
target_struct.num_targets = num_targets;
rate_array =  cell(num_layers,1);
psth_array =  cell(num_layers,1);
ave_target = cell(num_layers,num_targets);
ave_clutter = cell(num_layers,1);
ave_bkgrnd = cell(num_layers,1);
psth_target = cell(num_layers,num_targets);
psth_clutter = cell(num_layers,1);
psth_bkgrnd = cell(num_layers,1);
target_struct.target_ndx_max = cell(num_layers, num_targets);
target_struct.target_ndx_all = cell(num_layers, num_targets);
target_struct.clutter_ndx_max = cell(num_layers, 1);
target_struct.clutter_ndx_all = cell(num_layers, 1);
target_struct.bkgrnd_ndx_max = cell(num_layers, 1);
target_struct.bkgrnd_ndx_all = cell(num_layers, 1);
writeNonspikingActivity = 0.0;target_struct.num_target_neurons_all = zeros(num_layers, num_targets);
target_struct.num_target_neurons_max = zeros(num_layers, num_targets);
target_struct.num_clutter_neurons_all = zeros(num_layers, 1);
target_struct.num_clutter_neurons_max = zeros(num_layers, 1);
target_struct.num_bkgrnd_neurons_max = zeros(num_layers, 1);
target_struct.num_bkgrnd_neurons_all = zeros(num_layers, 1);
target_rate = cell(num_layers, num_targets);
target_rate_ndx = cell(num_layers, num_targets);
clutter_rate = cell(num_layers, 1);
clutter_rate_ndx = cell(num_layers, 1);

%% read input image segmentation info
invert_image_flag = 0;
plot_input_image = 1;
if exist(image_filename, "file") &&  exist(target_filename, "file")
  [target_struct.target, target_struct.clutter, image_ndx, fig_tmp] = ...
      pvp_parseTarget(image_filename, ...
		      target_filename, ...
		      invert_image_flag, ...
		      plot_input_image);
  fig_list = [fig_list; fig_tmp];
  disp('parse BMP -> done');
else
  NROWS_image=256;
  NCOLS_image=256;
  num_pixels = NROWS_image * NCOLS_image;
  target_struct.target{1} = ...
      [1:num_pixels]';
  target_struct.clutter = [];
  image_ndx = target_struct.target;
endif

				% initialize to size of image (if known), these should be overwritten by each layer

NROWS = NROWS_image;
NCOLS = NCOLS_image;
NFEATURES = 8;
NO = NFEATURES; % number of orientations
NK = 1; % number of curvatures
dK = 0; % spacing between curvatures (1/radius)

				% data structures for layer shape
layer_struct = struct;
layer_struct.layerID = layerID;
layer_struct.layerIndex = layerIndex;
layer_struct.num_rows = ones(num_layers,1);
layer_struct.num_cols = ones(num_layers,1);
layer_struct.num_features = ones(num_layers,1);
layer_struct.num_neurons = ones(num_layers,1);
layer_struct.size_layer = cell(num_layers,1);


				% data structures for correlation analysis
				%stft_array = cell( num_layers, 1);
xcorr_struct = struct;
xcorr_struct.min_freq = 25;writeNonspikingActivity = 0.0;
xcorr_struct.max_freq = 75;
xcorr_struct.size_border_mask = 4;
max_lag = 128/DELTA_T; 
xcorr_struct.max_lag = max_lag; 
xcorr_struct.border_mask = cell(num_layers,1);
num_modes = 1;
xcorr_struct.num_modes = num_modes;
xcorr_struct.power_mask = cell(num_layers, num_modes);
xcorr_struct.num_power_mask = zeros(num_layers, num_modes);
xcorr_struct.power_method = 0;
xcorr_struct.power_win_size = zeros(num_layers, 1);
xcorr_struct.xcorr_flag = 0;
num_eigen = 3;
xcorr_struct.num_eigen = num_eigen;
xcorr_struct.calc_power_mask = 1;
xcorr_struct.num_sig = 4;  %% ? throws memory allocation error
calc_eigen = 0;
xcorr_struct.calc_eigen = calc_eigen;

xcorr_eigenvector = cell( num_layers, num_modes, num_eigen);
xcorr_array = cell(num_layers, num_modes);
power_array = cell( num_layers, num_modes);

				% data structures for epochs
epoch_struct = struct;
num_epochs = 1;
epoch_struct.num_epochs = num_epochs;
epoch_struct.sum_total_time = zeros(1, num_layers);
epoch_struct.sum_total_steps = zeros(1, num_layers);
epoch_struct.sum_total_spikes = zeros(1, num_layers);
epoch_struct.epoch_time = zeros(1, num_layers);
epoch_struct.epoch_steps = zeros(1, num_layers);
epoch_struct.epoch_bins = zeros(1, num_layers);
epoch_struct.time_origin = zeros(1,num_layers);
epoch_struct.stim_begin_step = zeros(1, num_layers); 
epoch_struct.stim_end_step = zeros(1, num_layers); 
epoch_struct.stim_begin_bin = zeros(1, num_layers); 
epoch_struct.stim_end_bin = zeros(1, num_layers); 
epoch_struct.begin_time = repmat(BEGIN_TIME, [num_epochs, num_layers]);
epoch_struct.end_time = repmat(END_TIME, [num_epochs, num_layers]);
epoch_struct.exclude_offset = zeros(num_epochs, num_layers);
epoch_struct.total_spikes = zeros(num_epochs, num_layers);
epoch_struct.total_steps = zeros(num_epochs, num_layers);


%% setup epoch_struct for each layer
for layer = read_spikes
  disp(['building epoch stuct: ', num2str(layer)]);

  %% re-initialize begin/end times for each layer
  BEGIN_TIME = epoch_struct.begin_time(1,layer);
  END_TIME = epoch_struct.end_time(1,layer);  

  [epoch_struct, layer_struct] = ...
      pvp_setEpochStruct(epoch_struct, layer, layer_struct, num_epochs);

  disp( [layer_struct.layerID{layer}, ...
         ' epoch_struct.sum_total_time(', num2str(layer), ') = ', ...
         num2str(epoch_struct.sum_total_time(layer))]);
  ave_rate_tmp = ...
      1000 * epoch_struct.sum_total_spikes(layer) / ...
      ( N * epoch_struct.sum_total_time(layer) );
  disp( [layer_struct.layerID{layer}, ...
         ' ave_rate(',num2str(layer),') = ', ...
         num2str(ave_rate_tmp)] );
  int_fmt = "%i12";
  disp( ['epoch_struct.sum_total_spikes(', num2str(layer), ') = ', ...
         num2str(epoch_struct.sum_total_spikes(layer), int_fmt) ]);
  disp( ['epoch_struct.sum_total_steps(', num2str(layer), ') = ', ...
         num2str(epoch_struct.sum_total_steps(layer), int_fmt) ]);
  
  for i_epoch = 1 : num_epochs
    disp( [layer_struct.layerID{layer}, ...
           ' epoch_struct.begin_time(', ...
				     num2str(i_epoch), ...
				     ',', ...
				     num2str(layer), ') = ', ...
           num2str(epoch_struct.begin_time(i_epoch,layer) )] );
    disp( [layer_struct.layerID{layer}, ...
           ' epoch_struct.end_time(', ...
				   num2str(i_epoch), ...
				   ',', ...
				   num2str(layer), ') = ', ...
           num2str(epoch_struct.end_time(i_epoch,layer) )] );        
    disp( [layer_struct.layerID{layer}, ...
           ' epoch_struct.exclude_offset(', ...
					 num2str(i_epoch), ...
					 ',', ...
					 num2str(layer), ') = ', ...
           num2str(epoch_struct.exclude_offset(i_epoch,layer) )] );        
    disp( [layer_struct.layerID{layer}, ...
           ' epoch_struct.total_spikes(', ...
				       num2str(i_epoch), ...
				       ',', ...
				       num2str(layer), ') = ', ...
           num2str(epoch_struct.total_spikes(i_epoch,layer) )] );        
  endfor
endfor
%% end loop for building epoch_struct

%% set up xcorr_struct, compute border mask
for layer = read_spikes
  xcorr_struct.max_lag = ...
      min( xcorr_struct.max_lag, fix( epoch_struct.epoch_steps(layer))/2 );
  border_mask = ...
      ones( layer_struct.size_layer{layer} );
  border_mask(:, 1:xcorr_struct.size_border_mask, :) = 0;
  col_index_min = layer_struct.num_cols(layer)-xcorr_struct.size_border_mask;
  col_index_max = layer_struct.num_cols(layer);
  border_mask(:, col_index_min:col_index_max, :) = 0;
  border_mask(:, :, 1:xcorr_struct.size_border_mask) = 0;
  row_index_min = layer_struct.num_rows(layer)-xcorr_struct.size_border_mask;
  row_index_max = layer_struct.num_rows(layer);
  border_mask(:, :, row_index_min:row_index_max) = 0;
  xcorr_struct.border_mask{layer} = ...
      find( border_mask(:) );
  xcorr_struct.power_win_size(layer) = ...
      epoch_struct.epoch_steps(layer);
endfor
border_mask = [];
xcorr_struct.freq_vals = ...
    1000*(1/DELTA_T)*(0:2*xcorr_struct.max_lag)/(1+2*xcorr_struct.max_lag);
xcorr_struct.min_freq_ndx = ...
    find(xcorr_struct.freq_vals >= xcorr_struct.min_freq, 1,'first');
xcorr_struct.max_freq_ndx = ...
    find(xcorr_struct.freq_vals <= xcorr_struct.max_freq, 1,'last');
disp([ 'min_freq_ndx = ', num2str(xcorr_struct.min_freq_ndx) ]);
disp([ 'max_freq_ndx = ', num2str(xcorr_struct.max_freq_ndx) ]);


%% get rate and power arrays
for layer = read_spikes;
  disp(['analyzing layer rate and power: ', num2str(layer)]);

  rate_array = ...
      pvp_analyzeRate(layer, ...
		      epoch_struct, ...
		      layer_struct, ...
		      rate_array);

  power_array = ...
      pvp_analyzePower(layer, ...
		       epoch_struct, ...
		       layer_struct, ...
		       xcorr_struct, ...
		       power_array);
  
  %% plot averages over epochs
  stim_steps = ...
      epoch_struct.stim_begin_step(layer) : epoch_struct.stim_end_step(layer);
				%rate_array{layer,1} = rate_array{layer,1} / num_epochs;
  disp([layer_struct.layerID{layer}, ': rate_array(',num2str(layer),') = ', ...
        num2str( mean( rate_array{layer,1}(stim_steps) / num_epochs ) ), 'Hz']);    
  
  
  %% reconstruct image from layer activity
				%rate_array{layer} = rate_array{layer} / num_epochs;
  size_recon = ...
      [1, layer_struct.num_features(layer), ...
       layer_struct.num_cols(layer), ...
       layer_struct.num_rows(layer)];
  plot_rate_reconstruction = ismember( layer, plot_reconstruct );
  if plot_rate_reconstruction
    plot_title = ...
        [layer_struct.layerID{layer}, ...
         ' Rate(', ...
		int2str(layer), ...
		')'];
    rate_array_tmp = ...
	rate_array{layer} / num_epochs;
				%        fig_tmp = figure;
				%        set(fig_tmp, 'Name', plot_title);
    fig_tmp = pvp_reconstruct(rate_array_tmp, ...
			      plot_title, ...
			      [], ...
			      size_recon);
    fig_list = [fig_list; fig_tmp];
  endif %%plot_rate_reconstruction

  %% display top ranked cells
  [rate_array_tmp, rate_rank_tmp] = ...
      sort( rate_array{layer, 1}/num_epochs, 2, 'descend');
  for i_rank = [ 1:3 ] % , ceil(num_clutter_neurons(layer, 1)/2), num_clutter_neurons(layer, 1) ]
    tmp_rate = rate_array_tmp(i_rank);
    k = rate_rank_tmp(i_rank);
    [kf, kcol, krow] = ...
	ind2sub(layer_struct.size_layer{layer}, k);
    disp(['rank:', num2str(i_rank), ...
          ', rate_array(', num2str(layer),')', ...
          num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
          num2str([k-1, kcol-1, krow-1, kf-1]) ]);
  endfor %% % i_rank

  if rate_array_tmp(1) == 0
    read_spikes = setdiff(read_spikes, layer);
    continue;
  endif
  
  
  %%plot power reconstruction
  disp( [ 'ploting Power Recon(', num2str(layer), ')'] );
  plot_power_reconstruction = ismember( layer, plot_reconstruct );
  if plot_power_reconstruction
    plot_title = ...
        [layer_struct.layerID{layer}, ' Peak Power(', int2str(layer), ')' ];
    fig_tmp = ...
        pvp_reconstruct(power_array{layer, 1},  plot_title);
    fig_list = [fig_list; fig_tmp];
    plot_title = ...
        [layer_struct.layerID{layer}, ' Ave Power(', int2str(layer), ')' ];
    fig_tmp = ...
        pvp_reconstruct(power_array{layer, 2},  plot_title);
    fig_list = [fig_list; fig_tmp];
  endif %%
  
  %%compute power mask
  [xcorr_struct] = ...
      pvp_powerMask(layer, xcorr_struct, target_struct, power_array);

  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor %% % layer




%% get target and clutter segments
for layer = read_spikes;
  disp(['analyzing target and clutter segments: ', num2str(layer)]);
  [target_struct] = ...
      pvp_setTargetStruct(layer, ...
			  epoch_struct, ...
			  layer_struct, ...
			  target_struct, ...
			  rate_array);
endfor %% layer


%% make raster plots
for layer = read_spikes;
  plot_raster2 = ismember(layer, plot_raster);
  raster_epoch = []; %[epoch_struct.num_epochs];
  if ismember( layer, plot_raster )
    fig_list = pvp_plotRaster(layer, ...
			      epoch_struct, ...
			      layer_struct, ...
			      target_struct, ...
			      raster_epoch, ...
			      fig_list);
  endif %% plot_raster
endfor %%layer
pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
close all;
fig_list = [];



%% make spike movie
for layer = read_spikes;
  plot_movie2 = ismember(layer, plot_movie);
  movie_epoch = [2]; 
  if ismember( layer, plot_movie )
    fig_tmp = ...
	pvp_plotMovie(layer, ...
		      epoch_struct, ...
		      layer_struct, ...
		      target_struct, ...
		      movie_epoch, ...
		      MOVIE_PATH);
  fig_list = [fig_list; fig_tmp];
  endif %% plot_movie
endfor %%layer
pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
close all;
fig_list = [];


%% calc target and clutter PSTH
for layer = read_spikes;
  [ave_target, ...
   psth_target, ...
   ave_clutter, ...
   psth_clutter, ...
   ave_bkgrnd, ...
   psth_bkgrnd] = ...
      pvp_targetRate(layer, ...
		     epoch_struct, ...
		     layer_struct, ...
		     target_struct, ...
		     ave_target, ...
		     psth_target, ...
		     ave_clutter, ...
		     psth_clutter, ...
		     ave_bkgrnd, ...
		     psth_bkgrnd);

endfor % layer

%% disp target and clutter firing rates
for layer = read_spikes;

  %% compute averages for different image segments
  stim_steps = ...
      epoch_struct.stim_begin_step(layer) : epoch_struct.stim_end_step(layer);
  for i_target = 1 : target_struct.num_targets
				%ave_target{layer,i_target} = ave_target{layer,i_target} / num_epochs;
    disp([layer_struct.layerID{layer}, ...
	  ': ave_target(',num2str(layer),',', num2str(i_target), ') = ', ...
          num2str( mean( ave_target{layer,i_target}(stim_steps) / num_epochs ) ), 'Hz']);
  endfor
				%ave_clutter{layer,1} = ave_clutter{layer,1} / num_epochs;
  disp([layer_struct.layerID{layer}, ': ave_clutter(',num2str(layer),') = ', ...
        num2str( mean( ave_clutter{layer,1}(stim_steps) / num_epochs ) ), 'Hz']);
				%ave_bkgrnd{layer,1} = ave_bkgrnd{layer,1} / num_epochs;
  disp([layer_struct.layerID{layer}, ': ave_bkgrnd(',num2str(layer),') = ', ...
        num2str( mean( ave_bkgrnd{layer,1}(stim_steps) / num_epochs ) ), 'Hz']);
  
endfor % layer


%% plot target and clutter PSTH
for layer = read_spikes;
  
  %% plot PSTH
  time_bins = (1: epoch_struct.epoch_bins(layer))*BIN_STEP_SIZE;
  plot_title = ...
      [layer_struct.layerID{layer}, ' PSTH(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
  set(gca, 'plotboxaspectratiomode', 'manual');
  axis normal
  for i_target = 1 : target_struct.num_targets
				%psth_target{layer,i_target} =  psth_target{layer,i_target} / num_epochs;
    lh = plot(time_bins, ...
	      psth_target{layer,i_target} / num_epochs, '-r');
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
  
  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor % layer


%% reconstruct image segments from layer activity
for layer = read_spikes;

				%rate_array{layer} = rate_array{layer} / num_epochs;
  size_recon = ...
      [1 ...
       layer_struct.num_features(layer), ...
       layer_struct.num_cols(layer), ...
       layer_struct.num_rows(layer)];
  for i_target = 1:target_struct.num_targets
    target_rate{layer, i_target} = ...
        rate_array{layer}(1, target_struct.target_ndx_all{layer, i_target}) / ...
	num_epochs;
    rate_array_tmp = ...
        sparse(1, ...
	       target_struct.target_ndx_all{layer, i_target}, ...
	       target_rate{layer, i_target} / num_epochs, ...
	       1 , ...
	       layer_struct.num_neurons(layer), ...
	       target_struct.num_target_neurons_all(layer, i_target) );
    if ismember( layer, plot_reconstruct_target )
      plot_title = ...
          [layer_struct.layerID{layer}, ...
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
      k = target_struct.target_ndx_all{layer, i_target}(tmp_ndx);
      [kf, kcol, krow] = ...
	  ind2sub(layer_struct.size_layer{layer}, k);
      disp(['rank:',num2str(i_rank),...
            ', target_rate(',num2str(layer),', ', num2str(i_target), ')', ...
            num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
            num2str([k-1, kcol-1, krow-1, kf-1]) ]);
    endfor %% % i_rank
  endfor %% % i_target
  if ~isempty(target_struct.clutter)
    clutter_rate{layer, 1} = ...
	rate_array{layer}(1,target_struct.clutter_ndx_all{layer, 1}) / ...
	num_epochs;
    rate_array_tmp = ...
	sparse(1, ...
	       target_struct.clutter_ndx_all{layer, 1}, ...
	       clutter_rate{layer, 1} / num_epochs, ...
	       1 , ...
	       layer_struct.num_neurons(layer), ...
	       target_struct.num_clutter_neurons_all(layer, 1) );
    if ismember( layer, plot_reconstruct_target )
      plot_title = ...
          [layer_struct.layerID{layer}, ...
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
    [clutter_rate{layer, 1}, clutter_rate_ndx{layer, 1}] = ...
	sort( clutter_rate{layer, 1}, 2, 'descend');
    for i_rank = [ 1:3 ] % , ceil(num_clutter_neurons(layer, 1)/2), num_clutter_neurons(layer, 1) ]
      tmp_rate = clutter_rate{layer, 1}(i_rank);
      tmp_ndx = clutter_rate_ndx{layer, 1}(i_rank);
      k = target_struct.clutter_ndx_all{layer, 1}(tmp_ndx);
      [kf, kcol, krow] = ...
	  ind2sub(layer_struct.size_layer{layer}, k);
      disp(['rank:',num2str(i_rank),...
            ', clutter_rate(', num2str(layer),')', ...
            num2str(tmp_rate),', [k, kcol, krow, kf] = ', ...
            num2str([k-1, kcol-1, krow-1, kf-1]) ]);
    endfor %% % i_rank
  endif %%  ~isempty(target_struct.clutter)

  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor %% layer


%% calc target and clutter xcorr
start_xcorr_timer = time;

%% init mass corr data structures
mass_target_xcorr = cell(target_struct.num_targets, num_layers);
mass_target_autocorr = cell(target_struct.num_targets, num_layers);
target_xcorr = cell(target_struct.num_targets, num_layers);
mass_clutter_xcorr = cell( 1, num_layers );
mass_clutter_autocorr = cell( 1, num_layers );
clutter_xcorr = cell(1, num_layers);
mass_target2clutter_xcorr = cell(target_struct.num_targets, num_layers);
mass_target2clutter_autocorr = cell(target_struct.num_targets, num_layers);
target2clutter_xcorr = cell(target_struct.num_targets, num_layers);
xcorr_freqs = 1000*(1/DELTA_T)*(0:2*xcorr_struct.max_lag)/(1+2*xcorr_struct.max_lag);
min_ndx = 3 * xcorr_struct.max_freq_ndx;

for layer = read_spikes;
  
  %% mass correlation analysis
  plot_autocorr2 = ismember( layer, plot_autocorr );
  if ~plot_autocorr2
    continue;
  endif %% 
  
  %% accumulate massXCorr of target and distractor and background elements separately
  disp([layer_struct.layerID{layer}, ...
        ' ', 'mass_target_xcorr', ...
        '(', ...
          num2str(layer), ',', num2str(i_epoch), ')']);
  
  parallel_flag = 1;
  if parallel_flag
    [mass_target_xcorr, ...
     mass_target_autocorr, ...
     mass_clutter_xcorr, ...
     mass_clutter_autocorr, ...
     mass_target2clutter_xcorr, ...
     mass_target2clutter_autocorr] = ...
	pvp_targetXCorrPCell(layer, ...
			     epoch_struct, ...
			     layer_struct, ...
			     target_struct, ...
			     xcorr_struct, ...
			     mass_target_xcorr, ...
			     mass_target_autocorr, ...
			     mass_clutter_xcorr, ...
			     mass_clutter_autocorr, ...
			     mass_target2clutter_xcorr, ...
			     mass_target2clutter_autocorr);
  else
    
    [mass_target_xcorr, ...
     mass_target_autocorr, ...
     mass_clutter_xcorr, ...
     mass_clutter_autocorr, ...
     mass_target2clutter_xcorr, ...
     mass_target2clutter_autocorr] = ...
	pvp_targetXCorr(layer, ...
			epoch_struct, ...
			layer_struct, ...
			target_struct, ...
			xcorr_struct, ...
			mass_target_xcorr, ...
			mass_target_autocorr, ...
			mass_clutter_xcorr, ...
			mass_clutter_autocorr, ...
			mass_target2clutter_xcorr, ...
			mass_target2clutter_autocorr);
  endif
  
endfor %% % layer



%% plot MassXCorr
for layer = read_spikes;
  plot_autocorr2 = ismember( layer, plot_autocorr );
  if ~plot_autocorr2
    continue;
  endif %% 

  plot_title = [layer_struct.layerID{layer}, ' TargetXCorr(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
  for i_target = 1 : num_targets
    lh_target = ...
	plot((-xcorr_struct.max_lag:xcorr_struct.max_lag)*DELTA_T, ...
	     squeeze( mass_target_xcorr{i_target, layer} * ...
		     ( 1000 / DELTA_T )^2 / num_epochs ), '-r');
    set(lh_target, 'LineWidth', 2);
    hold on
  endfor %% % i_target
  lh_clutter = ...
      plot((-xcorr_struct.max_lag:xcorr_struct.max_lag)*DELTA_T, ...
	   squeeze( mass_clutter_xcorr{1,layer} * ...
		   ( 1000 / DELTA_T )^2 / num_epochs ), '-b');
  set(lh_clutter, 'LineWidth', 2);
  for i_target = 1 : num_targets
    lh_target2clutter = ...
	plot((-xcorr_struct.max_lag:xcorr_struct.max_lag)*DELTA_T, ...
	     squeeze( mass_target2clutter_xcorr{i_target, layer} * ...
		     ( 1000 / DELTA_T )^2 / num_epochs ), '-g');
    set(lh_target2clutter, 'LineWidth', 2);
    axis tight
  endfor %% % i_target

  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor %% % layer




%% plot MassAutoCorr
for layer = read_spikes;
  
  plot_autocorr2 = ismember( layer, plot_autocorr );
  if ~plot_autocorr2
    continue;
  endif %% 

  plot_title = [layer_struct.layerID{layer}, ' TargetAutoCorr(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
  for i_target = 1 : num_targets
    lh_target = ...
	plot((-xcorr_struct.max_lag:xcorr_struct.max_lag)*DELTA_T, ...
	     squeeze(mass_target_autocorr{i_target, layer} * ...
		     ( 1000 / DELTA_T )^2 / num_epochs ), '-r');
    set(lh_target, 'LineWidth', 2);
    hold on
  endfor %% % i_target
  lh_clutter = ...
      plot((-xcorr_struct.max_lag:xcorr_struct.max_lag)*DELTA_T, ...
	   squeeze( mass_clutter_autocorr{1,layer} * ...
		   ( 1000 / DELTA_T )^2 / num_epochs ), '-b');
  set(lh_clutter, 'LineWidth', 2);
  axis tight

  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor %% % layer



%%plot cross power
for layer = read_spikes;
  
  plot_autocorr2 = ismember( layer, plot_autocorr );
  if ~plot_autocorr2
    continue;
  endif %% 

  %%disp( [ 'computing XPower(', num2str(layer), ')'] );
  plot_title = ...
      [layer_struct.layerID{layer}, ' TargetXPower(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
  for i_target = 1 : num_targets
    fft_xcorr_tmp = ...
	fft( squeeze( mass_target_xcorr{i_target, layer} * ...
		     ( 1000 / DELTA_T )^2 / num_epochs ) );
    lh_target = plot(xcorr_freqs(2:min_ndx),...
		     abs( fft_xcorr_tmp(2:min_ndx) ), '-r');
    set(lh_target, 'LineWidth', 2);
    hold on
  endfor %% % i_target
  fft_xcorr_tmp = ...
      fft( squeeze( mass_clutter_xcorr{1,layer} * ...
		   ( 1000 / DELTA_T )^2 / num_epochs ) );
  lh_clutter = plot(xcorr_freqs(2:min_ndx),...
		    abs( fft_xcorr_tmp(2:min_ndx) ), '-b');
  set(lh_clutter, 'LineWidth', 2);
  for i_target = 1 : num_targets
    fft_xcorr_tmp = ...
	fft( squeeze( mass_target2clutter_xcorr{i_target, layer} * ...
		     ( 1000 / DELTA_T )^2 / num_epochs ) );
    lh_target2clutter = plot(xcorr_freqs(2:min_ndx),...
			     abs( fft_xcorr_tmp(2:min_ndx)), '-g');
    set(lh_target2clutter, 'LineWidth', 2);
  endfor %% % i_target
  axis tight

  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor %% % layer


%% plot auto power
for layer = read_spikes;
  
  plot_autocorr2 = ismember( layer, plot_autocorr );
  if ~plot_autocorr2
    continue;
  endif %% 
  plot_title = ...
      [layer_struct.layerID{layer}, ' TargetAutoPower(', int2str(layer), ')'];
  fig_tmp = figure('Name',plot_title);
  fig_list = [fig_list; fig_tmp];
  for i_target = 1 : num_targets
    fft_autocorr_tmp = ...
	fft( squeeze( mass_target_autocorr{i_target, layer} * ...
		     ( 1000 / DELTA_T )^2 / num_epochs ) );
    lh_target = plot(xcorr_freqs(2:min_ndx),...
		     abs( fft_autocorr_tmp(2:min_ndx)), '-r');
    set(lh_target, 'LineWidth', 2);
    hold on
  endfor %% % i_target
  fft_autocorr_tmp = ...
      fft( squeeze( mass_clutter_autocorr{1,layer} * ...
		   ( 1000 / DELTA_T )^2 / num_epochs ) );
  lh_clutter = plot(xcorr_freqs(2:min_ndx),...
		    abs( fft_autocorr_tmp(2:min_ndx)), '-b');
  set(lh_clutter, 'LineWidth', 2);
  axis tight
  figure(fig_tmp);

  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor %% % layer


stop_xcorr_timer = time;
total_xcorr_timer = stop_xcorr_timer - start_xcorr_timer;
disp(["total_xcorr_timer = ", num2str(total_xcorr_timer)]);


%% Eigen analysis
start_eigen_timer = time;
start_sparsecorr_timer = time;
if calc_eigen
  disp('beginning eigen analysis...');
else
  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  warning('abort eigen analysis');
endif

mass_xcorr = cell(num_modes, num_layers);
mass_xcorr_mean = cell(num_modes, num_layers);
mass_xcorr_std = cell(num_modes, num_layers);
mass_autocorr = cell(num_modes, num_layers);
xcorr_array = cell(num_modes, num_layers);
xcorr_dist = cell(num_modes, num_layers);

for layer = read_spikes
  if ~calc_eigen 
    continue;
  endif
  plot_xcorr2 = ismember( layer, plot_xcorr );
  if ~plot_xcorr2
    continue;
  endif %%
  disp(['layer( ', num2str(layer), ')']);

  if parallel_flag
    [mass_xcorr, ...
     mass_autocorr, ...
     mass_xcorr_mean, ...
     mass_xcorr_std, ...
     xcorr_array, ...
     xcorr_dist ] = ...
	pvp_sparseXCorrPCell(layer, ...
			     epoch_struct, ...
			     layer_struct, ...
			     target_struct, ...
			     xcorr_struct, ...
			     mass_xcorr, ...
			     mass_autocorr, ...
			     mass_xcorr_mean, ...
			     mass_xcorr_std, ...
			     xcorr_array, ...
			     xcorr_dist);
  else
    [mass_xcorr, ...
     mass_autocorr, ...
     mass_xcorr_mean, ...
     mass_xcorr_std, ...
     xcorr_array, ...
     xcorr_dist ] = ...
	pvp_sparseXCorr(layer, ...
			epoch_struct, ...
			layer_struct, ...
			target_struct, ...
			xcorr_struct, ...
			mass_xcorr, ...
			mass_autocorr, ...
			mass_xcorr_mean, ...
			mass_xcorr_std, ...
			xcorr_array, ...
			xcorr_dist);
  endif

endfor %% % layer

%% plot mass xcorr
for layer = read_spikes;
  plot_xcorr2 = ismember( layer, plot_xcorr ) && calc_eigen;
  if ~plot_xcorr2
    continue;
  endif %%
  for i_mode = 1 : num_modes  % 1 = peak, 2 = mean
    
    plot_title = ...
        [layer_struct.layerID{layer}, ...
         ' mass xcorr', ...
         '(', ...
           num2str(layer), ',',  ...
           num2str(i_mode), ...
           ')'];
    fig_tmp = figure('Name',plot_title);
    fig_list = [fig_list; fig_tmp];
    plot( (-xcorr_struct.max_lag : xcorr_struct.max_lag)*DELTA_T, ...
	 mass_xcorr{i_mode, layer} * ...
	 ( 1000 / DELTA_T )^2 / num_epochs, '-k');

  endfor %% % i_mode

  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor %% % layer


for layer = read_spikes;
  plot_xcorr2 = ismember( layer, plot_xcorr ) && calc_eigen;
  if ~plot_xcorr2
    continue;
  endif %%
  for i_mode = 1 : num_modes  % 1 = peak, 2 = mean
    plot_title = ...
        [layer_struct.layerID{layer}, ...
         ' fft mass xcorr', ...
         '(', ...
           num2str(layer), ',',  ...
           num2str(i_mode), ...
           ')'];
    fig_tmp = figure('Name',plot_title);
    fig_list = [fig_list; fig_tmp];
    fft_xcorr_tmp = ...
	real( fft( squeeze( mass_xcorr{i_mode, layer} * ...
			   ( ( 1000 / DELTA_T )^2 / num_epochs ) ) ) );
    fft_xcorr_tmp = ...
	fft_xcorr_tmp / ( fft_xcorr_tmp(1) + (fft_xcorr_tmp(1) > 0) );
    lh_fft_xcorr = plot(xcorr_freqs(2:min_ndx),...
			abs( fft_xcorr_tmp(2:min_ndx) ), '-b');
    set(lh_fft_xcorr, 'LineWidth', 2);
    lh = line( [xcorr_freqs(xcorr_struct.min_freq_ndx) ...
		xcorr_freqs(xcorr_struct.min_freq_ndx)], [0 1] );
    lh = line( [xcorr_freqs(xcorr_struct.max_freq_ndx) ...
		xcorr_freqs(xcorr_struct.max_freq_ndx)], [0 1] );
    
  endfor %% % i_mode
endfor %% % layer


for layer = read_spikes;
  plot_xcorr2 = ismember( layer, plot_xcorr ) && calc_eigen;
  if ~plot_xcorr2
    continue;
  endif %%
  for i_mode = 1 : num_modes  % 1 = peak, 2 = mean
    plot_title = ...
        [layer_struct.layerID{layer}, ...
         ' mass autocorr', ...
         '(', ...
           num2str(layer), ',',  ...
           num2str(i_mode), ...
           ')'];
    fig_tmp = figure('Name',plot_title);
    fig_list = [fig_list; fig_tmp];
    plot( (-xcorr_struct.max_lag : xcorr_struct.max_lag)*DELTA_T, ...
	 mass_autocorr{i_mode, layer} * ...
	 ( 1000 / DELTA_T )^2 / num_epochs, '-k');
    
  endfor %% % i_mode
  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor %% % layer

stop_sparsecorr_timer = time;
total_sparsecorr_timer = stop_sparsecorr_timer - start_sparsecorr_timer;
disp(["total_sparsecorr_timer = ", num2str(total_sparsecorr_timer)]);

%%find eigen vectors
for layer = read_spikes;
  plot_xcorr2 = ismember( layer, plot_xcorr ) && calc_eigen;
  if ~plot_xcorr2
    continue;
  endif %%
  for i_mode = 1 : num_modes  % 1 = peak, 2 = mean
    
    size_recon = ...
        [1, ...
	 layer_struct.num_features(layer), ...
	 layer_struct.num_cols(layer), ...
	 layer_struct.num_rows(layer) ];
    disp(['computing eigenvectors(', num2str(layer),')']);
    options.issym = 1;
    xcorr_array{i_mode, layer} = ...
	(1/2)*(xcorr_array{i_mode, layer} + ...
	       xcorr_array{i_mode, layer}') / ...
	num_epochs;
    mean_xcorr_array = ...
	mean( xcorr_array{i_mode, layer}(:) );
    xcorr_array{i_mode, layer} = ...
	xcorr_array{i_mode, layer} - mean_xcorr_array;
    [eigen_vec, eigen_value, eigen_flag] = ...
        eigs(xcorr_array{i_mode, layer}, num_eigen, 'lm', options);
    [sort_eigen, sort_eigen_ndx] = ...
	sort( diag( eigen_value ), 'descend' );
    for i_vec = 1:num_eigen
      plot_title = ...
          [layer_struct.layerID{layer}, ...
           'Eigen', ...
           '(', ...
             num2str(layer), ',', ...
             num2str(i_mode), ',', ...
             num2str(i_vec), ')'];
      disp([layer_struct.layerID{layer}, ...
            ' eigenvalues', ...
            '(', num2str(layer), ',' num2str(i_mode), ',', ...
	      num2str(i_vec),') = ', ...
            num2str(eigen_value(i_vec,i_vec))]);
      xcorr_eigenvector{layer, i_mode, i_vec} = ...
	  eigen_vec(:, sort_eigen_ndx(i_vec));
      eigen_vec_tmp = ...
          sparse( xcorr_struct.power_mask{layer, i_mode}, ...
                 1, ...
                 real( eigen_vec(:, sort_eigen_ndx(i_vec) ) ), ...
                 layer_struct.num_neurons(layer), ...
                 1, ...
                 xcorr_struct.num_power_mask(layer, i_mode));
      if mean(eigen_vec_tmp(:)) < 0
        eigen_vec_tmp = -eigen_vec_tmp;
      endif %%
      fig_tmp = ...
          pvp_reconstruct(full(eigen_vec_tmp), ...
			  plot_title, [], ...
			  size_recon);
      fig_list = [fig_list; fig_tmp];
    endfor %% % i_vec
  endfor %% % i_mode
  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endfor %% % layer


stop_eigen_timer = time;
total_eigen_timer = stop_eigen_timer - start_eigen_timer;
disp(["total_eigen_timer = ", num2str(total_eigen_timer)]);


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
    time_bins = 1: epoch_struct.epoch_bins(layer);
    lh(layer) = plot((time_bins)*BIN_STEP_SIZE, ...
		     psth_target{layer,i_target}(time_bins) / ...
		     num_epochs, '-r');
    set(lh(layer),'Color',co(1+mod(layer,size(co,1)),:));
    set(lh(layer),'LineWidth',2);
  endfor %%
  legend_str = ...
      {'retina  '; ...
       'LGN     '; ...
       'LGNInhFF'; ...
       'LGNInh  '; ...
       'S1      '; ...
       'S1Inh '; ...
       'C1      '; ...
       'C1Inh   '};
				%    if uimatlab
				%        leg_h = legend(lh(1:num_layers), legend_str);
				%    elseif uioctave
  legend(legend_str);
				%    endif %%
  fig_list = [fig_list; fig_tmp];
  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  close all;
  fig_list = [];
endif %%



%%read membrane potentials from point probes
if plot_vmem
  disp('plot_vmem')
  
				% TODO: the following info should be read from a pv ouptut file
  vmem_file_list = {'LGN_Vmem.txt', ...
		    'LGNInhFF_Vmem.txt', ...
		    'LGNInh_Vmem.txt', ...
		    'S1_Vmem.txt', ...
		    'S1Inh_Vmem.txt', ...
		    'C1_Vmem.txt', ...
		    'C1Inh_Vmem.txt'};
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
  vmem_skip = 1;
  layer = read_spikes( find(read_spikes, 1) );
  BEGIN_TIME = epoch_struct.begin_time(ceil(num_epochs/2),layer);
  END_TIME = BEGIN_TIME + 500; %%epoch_struct.end_time(ceil(num_epochs/2),layer); %%BEGIN_TIME + 200; %epoch_struct.end_time(num_epochs,layer);
  for i_vmem = 1 : vmem_skip : num_vmem_files
    [vmem_time{i_vmem}, ...
     vmem_G_E{i_vmem}, ...
     vmem_G_I{i_vmem}, ...
     vmem_G_IB{i_vmem}, ...
     vmem_V{i_vmem}, ...
     vmem_Vth{i_vmem}, ...
     vmem_a{i_vmem} ] = ...
        ptprobe_readV(vmem_file_list{i_vmem});
    if isempty(vmem_time{i_vmem})
      continue;
    endif
    vmem_start = 1;
    vmem_stop = length(vmem_time{i_vmem});
    if isempty(vmem_stop)
      vmem_stop = END_TIME;
    endif
    vmem_str_last = strfind(vmem_file_list{i_vmem}, '.');
    vmem_str_root = vmem_file_list{i_vmem}(1:vmem_str_last(1));
    plot_title = vmem_str_root;
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

pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
close all;
fig_list = [];



stop_timer = time;
total_timer = stop_timer - start_timer;
disp(["total_timer = ", num2str(total_timer)]);




%% plot connections
global COMPRESSED_FLAG
COMPRESSED_FLAG = 1;
global N_CONNECTIONS
global NXP NYP NFP
global NUM_ARBORS
NUM_ARBORS = 1; %% spiking connectins don't use arbors
[connID, connIndex, num_arbors] = pvp_connectionID();
%%[connID, connIndex] = pvp_connectionID();
%%N_CONNECTIONS = 16;
plot_weights = 1:N_CONNECTIONS;
weights = cell(N_CONNECTIONS, 1);
pvp_header = cell(N_CONNECTIONS, 1);
nxp = cell(N_CONNECTIONS, 1);
nyp = cell(N_CONNECTIONS, 1);
nfp = cell(N_CONNECTIONS, 1);
j_arbor = 0;
for i_conn = plot_weights
  [weights{i_conn}, nxp{i_conn}, nyp{i_conn}, pvp_header{i_conn}, pvp_index ] ...
      = pvp_readWeights(i_conn,j_arbor);
  NK = 1;
  NO = floor( NFEATURES / NK );
  pvp_header_tmp = pvp_header{i_conn};
  num_patches = pvp_header_tmp(pvp_index.WGT_NUMPATCHES);
  NFP = pvp_header_tmp(pvp_index.WGT_NFP);
  NX_PROCS = pvp_header_tmp(pvp_index.NX_PROCS);
  NY_PROCS = pvp_header_tmp(pvp_index.NY_PROCS);
  num_patches = num_patches; %% / (NX_PROCS * NY_PROCS);
  skip_patches = 1; %num_patches;
  data_size = pvp_header_tmp(pvp_index.DATA_SIZE);
  if data_size > 1
    COMPRESSED_FLAG = 0;
  elseif data_size == 1
    COMPRESSED_FLAG = 1;
  endif
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
    if prod(size_recon) == 1
      continue;
    endif
    fig_tmp = ...
        pvp_reconstruct(weights{i_conn}{i_patch}, ...
			plot_title, ...
			[], ...
			size_recon, ...
			1);
    fig_list = [fig_list; fig_tmp];
  endfor %% % i_patch
  pvp_saveFigList( fig_list, OUTPUT_PATH, 'png');
  fig_list = [];
  close all;
endfor %% % i_conn



