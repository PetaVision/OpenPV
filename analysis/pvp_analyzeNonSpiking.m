%%
close all
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
global num_trials first_trial last_trial
global first_training_trial last_training_trial training_trials
global first_testing_trial last_testing_trial testing_trials
global output_path input_path

input_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/';
output_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/output/';

pvp_order = 1;
				% initialize to size of image (if known), these should be overwritten by each layer
NROWS_image=256;
NCOLS_image=256;
NROWS = NROWS_image;
NCOLS = NCOLS_image;
NFEATURES = 12;

NO = NFEATURES; % number of orientations
NK = 1; % number of curvatures
dK = 0; % spacing between curvatures (1/radius)

num_trials = 3;
first_trial = 3;
last_trial = num_trials;

my_gray = [.666 .666 .666];
num_targets = 1;
fig_list = [];

global N_LAYERS
global SPIKING_FLAG
global pvp_index
SPIKING_FLAG = 0;
[layerID, layerIndex] = pvp_layerID();

read_activity = [1 2]; % 1:N_LAYERS;  % list of spiking layers whose spike train are to be analyzed
num_layers = N_LAYERS;

plot_reconstruct = [1 2]; %uimatlab;

%acivity_array = cell(num_layers, num_trials);
ave_activity = zeros(num_layers, num_trials);
act_time = zeros(num_layers, num_trials);

%% read input image
image_path = [input_path 'test_amoebas/'];

num_rows = ones(num_layers, num_trials);
num_cols = ones(num_layers, num_trials);
num_features = ones(num_layers, num_trials);
pvp_layer_header = cell(N_LAYERS, num_trials);

for i_trial = first_trial : last_trial

%% Analyze activity layer by layer
  for layer = read_activity;
      
				% Read spike events
    [act_time(layer, i_trial), activity, ave_activity(layer, i_trial), pvp_layer_header{layer, i_trial}] = ...
	pvp_readActivity(layer, i_trial, pvp_order);
    disp([ layerID{layer},
	  ': ave_activity(', num2str(layer), ',', num2str(i_trial), ') = ', ...
	  num2str(ave_activity(layer, i_trial))]);
    if isempty(activity)
      continue;
    endif
  
    num_rows(layer, i_trial) = NROWS;
    num_cols(layer, i_trial) = NCOLS;
    num_features(layer, i_trial) = NFEATURES;
  
				% plot reconstructed image
    plot_reconstruct2 = ismember( layer, plot_reconstruct );
    if plot_reconstruct2
      size_activity = ...
	  [ 1 , num_features(layer, i_trial), ...
	   num_cols(layer, i_trial), num_rows(layer, i_trial) ];
      fig_tmp = pvp_reconstruct(activity, ...
				[layerID{layer}, ' reconstruct: layer = ', ...
				 int2str(layer), ', trial = ', ...
				 num2str(i_trial)], [], ...
				size_activity);
      fig_list = [fig_list; fig_tmp];
    endif

  endfor % layer
 
endfor % i_trial


%% plot connections
global N_CONNECTIONS
global NXP NYP NFP
[connID, connIndex] = pvp_connectionID();
plot_weights = 1:3;%N_CONNECTIONS;
weights = cell(N_CONNECTIONS, 1);
pvp_conn_header = cell(N_CONNECTIONS, 1);
nxp = cell(N_CONNECTIONS, 1);
nyp = cell(N_CONNECTIONS, 1);
for i_conn = plot_weights
  [weights{i_conn}, nxp{i_conn}, nyp{i_conn}, pvp_conn_header{i_conn}, pvp_index ] ...
      = pvp_readWeights(i_conn);
  NK = 1;
  NO = floor( NFEATURES / NK );
  pvp_conn_header_tmp = pvp_conn_header{i_conn};
  num_patches = pvp_conn_header_tmp(pvp_index.WGT_NUMPATCHES);
  skip_patches = 1; %num_patches;
  for i_patch = 1 : skip_patches : num_patches
    NCOLS = nxp{i_conn}(i_patch);
    NROWS = nyp{i_conn}(i_patch);
    N = NROWS * NCOLS * NFEATURES;
    pvp_reconstruct(weights{i_conn}{i_patch}, [connID{i_conn}, ' Weight recon: i_conn = ', ...
					       int2str(i_conn), ': i_patch = ', ...
					       int2str(i_patch) ]);
  endfor % i_patch
endfor % i_conn



