%%
%close all
clear all

				% set paths, may not be applicable to all octave installations
pvp_matlabPath;

if ( uioctave )
  setenv("GNUTERM", "x11");
endif

				% Make the following global parameters available to all functions for convenience.
global N_image NROWS_image NCOLS_image
global N NROWS NCOLS % for the current layer
global NFEATURES  % for the current layer
global NO NK dK % for the current layer
global ROTATE_FLAG % orientation axis rotated by DTH / 2

global MIN_INTENSITY
MIN_INTENSITY = 0;

global NUM2STR_FORMAT
NUM2STR_FORMAT = "%03.3i";

global FLAT_ARCH_FLAG
FLAT_ARCH_FLAG = 1;

global TRAINING_FLAG
TRAINING_FLAG = 0;

global num_trials first_trial last_trial skip_trial
global first_training_trial last_training_trial training_trials
global first_testing_trial last_testing_trial testing_trials
global output_path

distractor_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/test_amoeba_distractor_4fc/';
target_path = [];
distractor_path = [];
target_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/test_amoeba_target_4fc/';
if ~isempty(target_path)
  output_path = target_path;
elseif ~isempty(distractor_path)
  output_path = distractor_path;
else
  output_path = [];
endif

min_target_flag = 2 - ~isempty(target_path);
max_target_flag = 1 + ~isempty(distractor_path);

pvp_order = 1;
ROTATE_FLAG = 1;
				% initialize to size of image (if known), these should be overwritten by each layer
NROWS_image=256;
NCOLS_image=256;
NROWS = NROWS_image;
NCOLS = NCOLS_image;
NFEATURES = 12;

NO = NFEATURES; % number of orientations
NK = 1; % number of curvatures
dK = 0; % spacing between curvatures (1/radius)

num_trials = 0;
first_trial =1;
last_trial = num_trials;
skip_trial = 1;

my_gray = [.666 .666 .666];
num_targets = 1;
fig_list = [];

global NUM_BIN_PARAMS
NUM_BIN_PARAMS = 20;

global NUM_WGT_PARAMS
NUM_WGT_PARAMS = 5;

global N_LAYERS
global SPIKING_FLAG
global pvp_index
SPIKING_FLAG = 0;
[layerID, layerIndex] = pvp_layerID;

read_activity = 1:N_LAYERS;  % list of spiking layers whose spike train are to be analyzed
num_layers = N_LAYERS;

if num_trials - first_trial + 1 > 10000000
  reconstruct_activity = [];
else
  reconstruct_activity = 1:N_LAYERS;
endif

%acivity_array = cell(num_layers, num_trials);
ave_activity = zeros(num_layers, num_trials);
sum_activity = zeros(num_layers, num_trials);
global hist_activity_bins num_hist_activity_bins
num_hist_activity_bins = 10;
hist_activity_bins = [0:num_hist_activity_bins-1]/num_hist_activity_bins;
hist_activity = zeros(2, num_hist_activity_bins, num_layers);
act_time = zeros(num_layers, num_trials);
twoAFC = zeros(2, num_layers, num_trials);
twoAFCsum = zeros(num_layers, 1);

num_rows = ones(num_layers, num_trials);
num_cols = ones(num_layers, num_trials);
num_features = ones(num_layers, num_trials);
pvp_layer_header = cell(N_LAYERS, num_trials);

for j_trial = first_trial : skip_trial : last_trial

close all;
fig_list = [];

%% Analyze activity layer by layer
  for layer = read_activity;

    % account for delays between layers
    i_trial = j_trial + (layer - 1);

    for target_flag = min_target_flag : max_target_flag

      if target_flag == 1
	output_path = target_path;
      elseif target_flag == 2
	output_path = distractor_path;
      endif
      
				% Read spike events
    [act_time(layer, j_trial), activity, ave_activity(layer, j_trial), ...
     sum_activity(layer, j_trial), hist_activity_tmp, pvp_layer_header{layer, j_trial}] = ...
	pvp_readActivity(layer, i_trial, pvp_order);
    disp([ layerID{layer}, ...
	  ': ave_activity(', num2str(layer), ',', num2str(j_trial), ') = ', ...
	  num2str(ave_activity(layer, j_trial))]);
    if isempty(activity)
      continue;
    endif
    hist_activity(target_flag, :, layer) = ...
	hist_activity(target_flag, :, layer) + ...
	hist_activity_tmp;

    twoAFC(target_flag, layer, j_trial) = ...
	sum_activity(layer, j_trial);

    write_activity_flag = 0;
    zip_activity_flag = 1;
    if write_activity_flag == 1
      if (zip_activity_flag == 1)
	activity_filename = ['V1_G', num2str(layer-1),'_', ...
			     num2str(j_trial, NUM2STR_FORMAT), '.mat.z']
	activity_filename = [output_path, activity_filename]
	save("-z", "-mat", activity_filename, "activity" );
      else
	activity_filename = ['V1_G', num2str(layer-1), '_', ...
			     num2str(j_trial, NUM2STR_FORMAT), '.mat']
	activity_filename = [output_path, activity_filename]
	save("-mat", activity_filename, "activity" );
      endif
    
    endif
     
  
    num_rows(layer, j_trial) = NROWS;
    num_cols(layer, j_trial) = NCOLS;
    num_features(layer, j_trial) = NFEATURES;
  
				% plot reconstructed image
    reconstruct_activity2 = ismember( layer, reconstruct_activity );
    if reconstruct_activity2
      size_activity = ...
	  [ 1 , num_features(layer, j_trial), ...
	   num_cols(layer, j_trial), num_rows(layer, j_trial) ];
      activity_filename = ...
	  ['V1_G', num2str(layer-1), '_', ...
	   num2str(j_trial, NUM2STR_FORMAT)];
      fig_tmp = pvp_reconstruct(activity, ...
				activity_filename, [], ...
				size_activity);
      fig_list = [fig_list; fig_tmp];
    endif

  endfor % target_flag

  twoAFC_tmp = twoAFC(1, layer, j_trial) > twoAFC(2, layer, j_trial);
  twoAFCsum(layer, 1) = twoAFCsum(layer, 1) + twoAFC_tmp;

  endfor % layer
 
  num_figs = length(fig_list);
  for i_fig = 1 : num_figs
    fig_hndl = fig_list(i_fig);
    fig_filename = get(fig_hndl, 'Name');
    fig_filename = [output_path, fig_filename, '.png'];
    print(fig_hndl, fig_filename, '-dpng');
  endfor

endfor % j_trial



%% plot connections
global N_CONNECTIONS
global NXP NYP NFP
[connID, connIndex] = pvp_connectionID();
if TRAINING_FLAG
  plot_weights = 1 : N_CONNECTIONS;
else
  plot_weights = ( N_CONNECTIONS - 1 ) : ( N_CONNECTIONS+~TRAINING_FLAG );
endif
weights = cell(N_CONNECTIONS+1, 1);
weight_invert = ones(N_CONNECTIONS+~TRAINING_FLAG, 1);
weight_invert(5) = -1;
pvp_conn_header = cell(N_CONNECTIONS+~TRAINING_FLAG, 1);
nxp = cell(N_CONNECTIONS+~TRAINING_FLAG, 1);
nyp = cell(N_CONNECTIONS+~TRAINING_FLAG, 1);
for i_conn = plot_weights
    weight_min = 10000000.;
    weight_max = -10000000.;
    weight_ave = 0;
  if i_conn < N_CONNECTIONS+1
    [weights{i_conn}, nxp{i_conn}, nyp{i_conn}, pvp_conn_header{i_conn}, pvp_index ] ...
	= pvp_readWeights(i_conn);
    pvp_conn_header_tmp = pvp_conn_header{i_conn};
    num_patches = pvp_conn_header_tmp(pvp_index.WGT_NUMPATCHES);
    for i_patch = 1:num_patches
      weight_min = min( min(weights{i_conn}{i_patch}(:)), weight_min );
      weight_max = max( max(weights{i_conn}{i_patch}(:)), weight_max );
      weight_ave = weight_ave + mean(weights{i_conn}{i_patch}(:));
    endfor
    weight_ave = weight_ave / num_patches;
    disp( ['weight_min = ', num2str(weight_min)] );
    disp( ['weight_max = ', num2str(weight_max)] );
    disp( ['weight_ave = ', num2str(weight_ave)] );
    if ~TRAINING_FLAG
      continue;
    endif
  elseif i_conn == N_CONNECTIONS + 1
    disp('calculating geisler kernels');
    pvp_conn_header{i_conn} = pvp_conn_header{i_conn-2};
    pvp_conn_header_tmp = pvp_conn_header{i_conn};
    num_patches = pvp_conn_header_tmp(pvp_index.WGT_NUMPATCHES);
    nxp{i_conn} = nxp{i_conn-2};
    nyp{i_conn} = nyp{i_conn-2};
    for i_patch = 1:num_patches
      weights{i_conn}{i_patch} = ...
	  weights{i_conn-2}{i_patch} + weights{i_conn-1}{i_patch};
      weight_min = min( min(weights{i_conn}{i_patch}(:)), weight_min );
      weight_max = max( max(weights{i_conn}{i_patch}(:)), weight_max );
      weight_ave = weight_ave + mean(weights{i_conn}{i_patch}(:));
    endfor
    weight_ave = weight_ave / num_patches;
    disp( ['weight_min = ', num2str(weight_min)] );
    disp( ['weight_max = ', num2str(weight_max)] );
    disp( ['weight_ave = ', num2str(weight_ave)] );
    write_kernel_flag = 1;
    if write_kernel_flag
      NCOLS = 128; %pvp_conn_header_tmp(pvp_index.NX);
      NROWS = 128; %pvp_conn_header_tmp(pvp_index.NY);
      NFEATURES = pvp_conn_header_tmp(pvp_index.NF);
      NXP = pvp_conn_header_tmp(pvp_index.WGT_NXP);
      NYP = pvp_conn_header_tmp(pvp_index.WGT_NYP);
      NFP = pvp_conn_header_tmp(pvp_index.WGT_NFP);
      N = NROWS * NCOLS * NFEATURES;
      weights_size = [ NFP, NXP, NYP];
      pvp_writeKernel( weights{i_conn}, weights_size, 'geisler_clean' );
    endif
  else
    continue;
  endif
  NK = 1;
  NO = floor( NFEATURES / NK );
  skip_patches = 1; %num_patches;
  for i_patch = 1 : skip_patches : num_patches
    NCOLS = nxp{i_conn}(i_patch);
    NROWS = nyp{i_conn}(i_patch);
    N = NROWS * NCOLS * NFEATURES;
    patch_size = [1 NFEATURES  NCOLS NROWS];
    fig_tmp = pvp_reconstruct(weights{i_conn}{i_patch}*weight_invert(i_conn), ...
		    [connID{i_conn}, '(', ...
		     int2str(i_conn), ',', ...
		     int2str(i_patch), ')' ], ...
		    [], patch_size);
    fig_list = [fig_list; fig_tmp];
  endfor % i_patch
endfor % i_conn




if max_target_flag > min_target_flag
  tot_trials = length( first_trial : skip_trial : num_trials );
  subplot_index = 0;
  num_subplots = length(read_activity);
  hist_name = '2AFC';
  fig_tmp = figure('Name', hist_name);
  for layer = read_activity
    subplot_index = subplot_index + 1;
    twoAFCsum_tmp = ...
	twoAFCsum(layer,1) / ( tot_trials + (tot_trials == 0) );
    disp( ['twoAVCsum(', num2str(layer), ') = ', ...
	   num2str(twoAFCsum_tmp) ] );
    subplot(num_subplots, 1, subplot_index);
    hist_activity_tmp = ...
	hist_activity(1, :, layer) / ...
	sum( squeeze( hist_activity(1, :, layer) ) );
    cum_activity_target = ...
	1 - cumsum( hist_activity_tmp )
    hist_activity_tmp = ...
	hist_activity(2, :, layer) / ...
	sum( squeeze( hist_activity(2, :, layer) ) );
    cum_activity_distractor = ...
	1 - cumsum( hist_activity_tmp )
    twoAFC_correct = 0.5 + 0.5 * ...
	( cum_activity_target - cum_activity_distractor )
    bar( hist_activity_bins, twoAFC_correct );  
  endfor
  fig_list = [fig_list; fig_tmp];
endif

num_figs = length(fig_list);
for i_fig = 1 : num_figs
  fig_hndl = fig_list(i_fig);
  fig_filename = get(fig_hndl, 'Name');
  fig_filename = [output_path, fig_filename, '.png'];
  print(fig_hndl, fig_filename, '-dpng');
endfor

close all;
fig_list = [];

