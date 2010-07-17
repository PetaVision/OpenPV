%%
close all
clear all
expNum = 1;
				% set paths, may not be applicable to all octave installations
%%pvp_matlabPath;

%%if ( uioctave )
setenv('GNUTERM', 'x11');
%%endif

				% Make the following global parameters available to all functions for convenience.
global N_image NROWS_image NCOLS_image
global N NROWS NCOLS % for the current layer
global NFEATURES  % for the current layer
global NO NK dK % for the current layer
global ROTATE_FLAG % orientation axis rotated by DTH / 2

global num_trials first_trial last_trial skip_trial
global output_path spiking_path twoAFC_path spiking_path activity_path

global MIN_INTENSITY
MIN_INTENSITY = 0;

global NUM2STR_FORMAT
NUM2STR_FORMAT = '%03.3i';

global FLAT_ARCH_FLAG
FLAT_ARCH_FLAG = 1;

global TRAINING_FLAG
TRAINING_FLAG = -3;

global FC_STR
				%FC_STR = ['_', num2str(4), 'fc'];
FC_STR = [num2str(4), 'fc'];

num_trials =    ( TRAINING_FLAG <= 0 ) * 999; % 9; %0;
first_trial =1;
last_trial = num_trials;
skip_trial = 1;

global G_STR
if abs(TRAINING_FLAG) == 1
  G_STR = '_G1';
elseif abs(TRAINING_FLAG) == 2
  G_STR = '_G2';
elseif abs(TRAINING_FLAG) == 3
  G_STR = '_G3';
elseif abs(TRAINING_FLAG) == 4
  G_STR = '_G4';
endif

machine_path = '/Users/gkenyon/Documents/eclipse-workspace/';
				%machine_path = '/nh/home/gkenyon/workspace/';

target_path = [];
target_path = [machine_path 'kernel/input/128/test_amoeba40K_target']; %, FC_STR];
if ~isempty(target_path)
  target_path = [target_path, G_STR, '/'];
  target_path = [target_path, FC_STR, '/'];
endif % ~isempty(target_path)

if num_trials > 10
  distractor_path = [machine_path, ...
		     'kernel/input/128/test_amoeba40K_distractor']; %, FC_STR];
else
  distractor_path = [];
endif
if ~isempty(distractor_path)
  distractor_path = [distractor_path, G_STR, '/'];
  distractor_path = [distractor_path, FC_STR, '/'];
endif % ~isempty(distractor_path)

twoAFC_path = target_path;
spiking_path = target_path; %[machine_path, 'kernel/input/spiking_target10K', FC_STR];
				%twoAFC_path = [twoAFC_path, G_STR, '/'];
				%spiking_path = [spiking_path, G_STR, '/'];
activity_path = {target_path; distractor_path};

min_target_flag = 2 - ~isempty(target_path);
max_target_flag = 1 + ~isempty(distractor_path);

pvp_order = 1;
ROTATE_FLAG = 1;
				% initialize to size of image (if known), these should be overwritten by each layer
NROWS_image=128; %256;
NCOLS_image=128; %256;
NROWS = NROWS_image;
NCOLS = NCOLS_image;
NFEATURES = 8;

NO = NFEATURES; % number of orientations
NK = 1; % number of curvatures
dK = 0; % spacing between curvatures (1/radius)

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

read_activity = 2:N_LAYERS;  % list of nonspiking layers whose activity is to be analyzed
num_layers = N_LAYERS;

if num_trials - first_trial + 1 > 10 %%|| TRAINING_FLAG > 0
  reconstruct_activity = [];
else
  reconstruct_activity = read_activity;
endif

				%acivity_array = cell(num_layers, num_trials);
ave_activity = zeros(2, num_layers, num_trials);
sum_activity = zeros(2, num_layers, num_trials);
global num_hist_activity_bins
num_hist_activity_bins = 100;
hist_activity_bins = cell(num_layers, 1);
for layer = 1 : num_layers
  hist_activity_bins{layer} = [];
endfor
hist_activity = zeros(2, num_hist_activity_bins, num_layers);
act_time = zeros(num_layers, num_trials);
twoAFC = zeros(2, num_layers, num_trials);

num_rows = ones(num_layers, num_trials);
num_cols = ones(num_layers, num_trials);
num_features = ones(num_layers, num_trials);
pvp_layer_header = cell(N_LAYERS, num_trials);

for j_trial = first_trial : skip_trial : last_trial
    
  close all;
  fig_list = [];
  
  %% Analyze activity layer by layer
  for layer = read_activity;
        
    %% account for delays between layers
    i_trial = j_trial + (layer - 1);
        
    for target_flag = min_target_flag : max_target_flag
      
      output_path = activity_path{target_flag};
      
      %% Read spike events
      hist_bins_tmp = ...
          hist_activity_bins{layer};
      [act_time(layer, j_trial),...
       activity, ...
       ave_activity(target_flag, layer, j_trial), ...
       sum_activity(target_flag, layer, j_trial), ...
       hist_activity_tmp, ...
       hist_activity_bins{layer}, ...
       pvp_layer_header{layer, j_trial}] = ...
          pvp_readActivity(layer, i_trial, hist_bins_tmp, pvp_order);
      disp([ layerID{layer}, ...
            ': ave_activity(', num2str(layer), ',', num2str(j_trial), ') = ', ...
            num2str(ave_activity(target_flag, layer, j_trial))]);
      if isempty(activity)
        continue;
      endif
      if layer == 1
        max_activity = max(activity(:));
        min_activity = min(activity(:));
        if any( ( activity(:) > min_activity ) && ...
               ( activity(:) < max_activity ) )
          disp( 'activity between min and max' );
        endif
      endif
      hist_activity(target_flag, :, layer) = ...
          hist_activity(target_flag, :, layer) + ...
          hist_activity_tmp;
      
      twoAFC(target_flag, layer, j_trial) = ...
          ave_activity(target_flag, layer, j_trial);
      
      write_activity_flag = 0;
      zip_activity_flag = 1;
      if write_activity_flag == 1
        if (zip_activity_flag == 1)
          activity_filename = ['V1_G', num2str(layer-1),'_', ...
			       num2str(j_trial, NUM2STR_FORMAT), '.mat.z']
          activity_filename = [output_path, activity_filename]
          %%save("-z", "-mat", activity_filename, "activity" );
          save('-mat', activity_filename, 'activity' );
        else
          activity_filename = ['V1_G', num2str(layer-1), '_', ...
			       num2str(j_trial, NUM2STR_FORMAT), '.mat']
          activity_filename = [output_path, activity_filename]
          save('-mat', activity_filename, 'activity' );
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
        plot_recon_flag = 1;
        fig_tmp = pvp_reconstruct(activity, ...
				  activity_filename, [], ...
				  size_activity, ...
				  plot_recon_flag);
        fig_list = [fig_list; fig_tmp];
      endif
            
    endfor % target_flag
    
    close all;
    fig_list = [];
	  
  endfor % layer
  
  pvp_saveFigList( fig_list, output_path, 'png');
  fig_list = [];
  
endfor % j_trial
	

%% plot connections
global N_CONNECTIONS
global NXP NYP NFP
[connID, connIndex] = pvp_connectionID();
if TRAINING_FLAG > 0
  plot_weights = N_CONNECTIONS;
else
  plot_weights = ( N_CONNECTIONS - 1 ) : ( N_CONNECTIONS+(TRAINING_FLAG<=0) );
endif
weights = cell(N_CONNECTIONS+(TRAINING_FLAG<=0), 1);
weight_invert = ones(N_CONNECTIONS+(TRAINING_FLAG<=0), 1);
weight_invert(6) = -1;
weight_invert(9) = -1;
weight_invert(12) = -1;
pvp_conn_header = cell(N_CONNECTIONS+(TRAINING_FLAG<=0), 1);
nxp = cell(N_CONNECTIONS+(TRAINING_FLAG<=0), 1);
nyp = cell(N_CONNECTIONS+(TRAINING_FLAG<=0), 1);
for i_conn = plot_weights
    weight_min = 10000000.;
    weight_max = -10000000.;
    weight_ave = 0;
  if i_conn < N_CONNECTIONS+1
    [weights{i_conn}, nxp{i_conn}, nyp{i_conn}, pvp_conn_header{i_conn}, pvp_index ] ...
	= pvp_readWeights(i_conn);
    pvp_conn_header_tmp = pvp_conn_header{i_conn};
    NXP = pvp_conn_header_tmp(pvp_index.WGT_NXP);
    NYP = pvp_conn_header_tmp(pvp_index.WGT_NYP);
    NFP = pvp_conn_header_tmp(pvp_index.WGT_NFP);
    if NXP <= 1 || NYP <= 1
      continue;
    endif
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
      N = NROWS * NCOLS * NFEATURES;
      weights_size = [ NFP, NXP, NYP];
      pvp_writeKernel( weights{i_conn}, weights_size, 'geisler_clean' );
      geisler_weights = weights{N_CONNECTIONS+(TRAINING_FLAG<=0)};
      geisler_weights_filename = ...
	  ['geisler_clean', num2str(expNum), '.mat']
      geisler_weights_filename = [twoAFC_path, geisler_weights_filename]
      %%save("-z", "-mat", geisler_weights_filename, "geisler_weights");
      save('-mat', geisler_weights_filename, 'geisler_weights');
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

pvp_saveFigList( fig_list, output_path, 'png');
fig_list = [];


%% 2AFC analysis

plot_hist_activity_flag = 0;
plot_2AFC_flag = num_trials > 10;
if max_target_flag > min_target_flag
  tot_trials = length( first_trial : skip_trial : num_trials );

  if plot_hist_activity_flag
    
    subplot_index = 0;
    num_subplots = length(read_activity);
    hist_name = 'Cum Pixel Dist';
    fig_tmp = figure('Name', hist_name);
    fig_list = [fig_list; fig_tmp];
    for layer = read_activity
      subplot_index = subplot_index + 1;
      subplot(num_subplots, 1, subplot_index);
      hist_activity_tmp = ...
	  hist_activity(1, :, layer) / ...
	  sum( squeeze( hist_activity(1, :, layer) ) );
      cum_activity_target = ...
	  1 - cumsum( hist_activity_tmp );
      hist_activity_tmp = ...
	  hist_activity(2, :, layer) / ...
	  sum( squeeze( hist_activity(2, :, layer) ) );
      cum_activity_distractor = ...
	  1 - cumsum( hist_activity_tmp );
      twoAFC_correct = 0.5 + 0.5 * ...
	  ( cum_activity_target - cum_activity_distractor );
      bar( hist_activity_bins{layer}, twoAFC_correct );  
    endfor
    hist_filename = ...
	['hist_activity', '.mat.z']
    hist_filename = [output_path, hist_filename]
    %%save("-z", "-mat", hist_filename, "hist_activity" );
            save('-mat', hist_filename, 'hist_activity');

  endif

  
  if plot_2AFC_flag

    %% pvp_calc2AFC();
    
    twoAFC_hist = cell(2, num_layers);
    twoAFC_bins = cell(num_layers, 1);
    subplot_index = 0;
    num_subplots = length(read_activity);
    twoAFC_hist_name = '2AFC hist';
    fig_tmp = figure('Name', twoAFC_hist_name);
    fig_list = [fig_list; fig_tmp];
    for layer = read_activity
      subplot_index = subplot_index + 1;
      subplot(num_subplots, 1, subplot_index);
      twoAFC_tmp = squeeze( twoAFC(:, layer, :) );
      [ twoAFC_hist_tmp, twoAFC_bins{layer} ] = ...
	  hist( twoAFC_tmp(:), num_hist_activity_bins );            
      for target_flag = 1 : 2;
	twoAFC_hist{target_flag, layer} = ...
	    hist( squeeze( twoAFC(target_flag, layer, :) ), twoAFC_bins{layer} );
	twoAFC_hist{target_flag, layer} = ...
	    twoAFC_hist{target_flag, layer} / ...
	    sum( twoAFC_hist{target_flag, layer} );
	if target_flag == 1
	  red_hist = 1;
	  blue_hist = 0;
	  bar_width = 0.8;
	else
	  red_hist = 0;
	  blue_hist = 1;
	  bar_width = 0.6;
	endif
	bh = bar( twoAFC_bins{layer}, ...
		 twoAFC_hist{target_flag, layer}, ...
		 bar_width);
	set( bh, 'EdgeColor', [red_hist 0 blue_hist] );
	set( bh, 'FaceColor', [red_hist 0 blue_hist] );
	hold on
      endfor  % target_flag
    endfor  % layer
    
    twoAFC_cumsum = cell(2, num_layers);
    twoAFC_ideal = cell(num_layers,1);
    subplot_index = 0;
    num_subplots = length(read_activity);
    twoAFC_ideal_name = '2AFC ideal observer';
    fig_tmp = figure('Name', twoAFC_ideal_name);
    fig_list = [fig_list; fig_tmp];
    for layer = read_activity
      for target_flag = 1 : 2;
	twoAFC_cumsum{target_flag, layer} = ...
	    1 - cumsum( twoAFC_hist{target_flag, layer} );
      endfor
      subplot_index = subplot_index + 1;
      subplot(num_subplots, 1, subplot_index);
      twoAFC_ideal{layer} = ...
	  0.5 + 0.5 * ...
	  ( twoAFC_cumsum{1, layer} - twoAFC_cumsum{2, layer} );
      bh = bar( twoAFC_bins{layer}, twoAFC_ideal{layer} );  
      set( bh, 'EdgeColor', [0 1 0] );
      set( bh, 'FaceColor', [0 1 0] );
      set(gca, 'YLim', [0 1]);
      hold on;
    endfor
    
    subplot_index = 0;
    num_subplots = length(read_activity);
    twoAFC_ROC_name = '2AFC ROC';
    fig_tmp = figure('Name', twoAFC_ROC_name);
    fig_list = [fig_list; fig_tmp];
    for layer = read_activity
      subplot_index = subplot_index + 1;
      subplot(num_subplots, 1, subplot_index);
      axis([0 1 0 1]);
      hold on;
      lh = plot( [0, fliplr( twoAFC_cumsum{2, layer} ), 1], ...
		 [0, fliplr( twoAFC_cumsum{1, layer} ), 1 ], ...
		'-k');  
      set( lh, 'LineWidth', 2 );
    endfor
    
    for layer = read_activity
      twoAFC_correct(layer) = ...
	  sum( squeeze( twoAFC(1,layer,:) > twoAFC(2,layer,:) ) ) / ...
	  ( tot_trials + (tot_trials == 0) );
      disp( ['twoAVC_correct(', num2str(layer), ') = ', ...
	     num2str(twoAFC_correct(layer)) ] );
    endfor
    
    twoAFC_filename = ...
	['twoAFC', num2str(expNum), '.mat.z']
    twoAFC_filename = [twoAFC_path, twoAFC_filename]
    save('-mat', twoAFC_filename, 'twoAFC', 'twoAFC_hist', ...
	 'twoAFC_cumsum', 'twoAFC_ideal', 'tot_trials', ...
	 'ave_activity', 'sum_activity');
    
  endif

  pvp_saveFigList( fig_list, twoAFC_path, 'png');
  fig_list = [];
  
endif




