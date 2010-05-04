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

global TRAINING_FLAG
TRAINING_FLAG = 0;

global num_trials first_trial last_trial skip_trial
global first_training_trial last_training_trial training_trials
global first_testing_trial last_testing_trial testing_trials
global output_path input_path

% input_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/';
% input_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/target_4fc/';
				% output_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/output/';
 output_path = '/Users/gkenyon/Documents/eclipse-workspace/kernel/input/test_amoeba_distractor_4fc/';

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

num_trials = 996;
first_trial = 1;
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
[layerID, layerIndex] = pvp_layerID();

read_activity = 1:N_LAYERS;  % list of spiking layers whose spike train are to be analyzed
num_layers = N_LAYERS;

reconstruct_activity = []; %1:N_LAYERS; %[1 2 3]; %uimatlab;

%acivity_array = cell(num_layers, num_trials);
ave_activity = zeros(num_layers, num_trials);
act_time = zeros(num_layers, num_trials);

%% read input image
image_path = [input_path 'test_amoebas/'];

num_rows = ones(num_layers, num_trials);
num_cols = ones(num_layers, num_trials);
num_features = ones(num_layers, num_trials);
pvp_layer_header = cell(N_LAYERS, num_trials);

for j_trial = first_trial : skip_trial : last_trial

%% Analyze activity layer by layer
  for layer = read_activity;

    % account for delays between layers
    if layer > j_trial 
      continue;
    endif
    i_trial = j_trial + (layer - 1);
    
				% Read spike events
    [act_time(layer, j_trial), activity, ave_activity(layer, j_trial), pvp_layer_header{layer, j_trial}] = ...
	pvp_readActivity(layer, i_trial, pvp_order);
    disp([ layerID{layer}, ...
	  ': ave_activity(', num2str(layer), ',', num2str(j_trial), ') = ', ...
	  num2str(ave_activity(layer, j_trial))]);
    if isempty(activity)
      continue;
    endif

    write_activity_flag = 1;
    if write_activity_flag == 1
      activity_format = "%03.3i";
      activity_filename = ['V1_G', num2str(layer-1),'_', ...
			   num2str(j_trial, activity_format), '.mat.z']
      activity_filename = [output_path, activity_filename]
      save("-z", "-mat", activity_filename, "activity" );
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
      fig_tmp = pvp_reconstruct(activity, ...
				[layerID{layer}, '(', ...
				 int2str(layer), ',' , ...
				 num2str(j_trial), ')'], [], ...
				size_activity);
      fig_list = [fig_list; fig_tmp];
    endif

  endfor % layer
 
endfor % j_trial


%% plot connections
global N_CONNECTIONS
global NXP NYP NFP
[connID, connIndex] = pvp_connectionID();
if TRAINING_FLAG
  plot_weights = []; %1 : N_CONNECTIONS;
else
  plot_weights = []; %( N_CONNECTIONS - 1 ) : ( N_CONNECTIONS+~TRAINING_FLAG );
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
    patch_size = [1 NFEATURES NCOLS NROWS];
    fig_tmp = pvp_reconstruct(weights{i_conn}{i_patch}*weight_invert(i_conn), ...
		    [connID{i_conn}, '(', ...
		     int2str(i_conn), ',', ...
		     int2str(i_patch), ')' ], ...
		    [], patch_size);
    fig_list = [fig_list; fig_tmp];
  endfor % i_patch
endfor % i_conn



num_figs = length(fig_list);
for i_fig = 1 : num_figs
  fig_hndl = fig_list(i_fig);
  fig_filename = get(fig_hndl, 'Name');
  fig_filename = [output_path, fig_filename, '.png'];
  print(fig_hndl, fig_filename, '-dpng');
endfor