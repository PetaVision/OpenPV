function [fig_list, ROC_struct] = ...
      pvp_ROC(exp_struct, layer_ndx, target_ID_ndx)

  %% confidence values for target and distractor images
  %% exp_stuct = stucture storing experimental data
  %% layer_ndx = indices of layers to analyze
  %% target_ID_ndx = indices of target_IDs to analyze

  [max_target_flag, num_layers, num_trials, num_target_IDs] = ...
      size(exp_struct.twoAFC_data);
  exp_ID = exp_struct.SOS_ID;

  if ( nargin < 2 ||  ~exist("layer_ndx") || isempty(layer_ndx) )
    layer_ndx = 1:num_layers;
  endif
  if ( nargin < 3 ||  ~exist("target_ID_ndx") || isempty(target_ID_ndx) )
    target_ID_ndx = 1:num_target_IDs;
  endif
  
  global num_hist_activity_bins

  fig_list = [];
  ROC_struct = struct;

  mean_2AFC = squeeze( mean( exp_struct.twoAFC_data, 3 ) );
  std_2AFC = squeeze( std( exp_struct.twoAFC_data, 0, 3 ) );

  %% data structures
  twoAFC_hist = cell(max_target_flag, num_layers, num_target_IDs);
  twoAFC_cumsum = cell(max_target_flag, num_layers, num_target_IDs);
  twoAFC_bins = cell(num_layers, num_target_IDs);
  twoAFC_ideal = cell(num_layers, num_target_IDs);

  subplot_index = 0;
  len_layer_ndx = length(layer_ndx);
  len_target_ID_ndx = length(target_ID_ndx);  
  num_subplots = len_layer_ndx * len_target_ID_ndx;
  twoAFC_hist_name = ['2AFC hist ', num2str(exp_struct.SOS_ID)];
  fig_tmp = figure('Name', twoAFC_hist_name);
  fig_list = [fig_list; fig_tmp];
  for layer = layer_ndx
    for target_ID = target_ID_ndx
      subplot_index = subplot_index + 1;
      subplot(len_layer_ndx, len_target_ID_ndx, subplot_index);
      twoAFC_tmp = squeeze( exp_struct.twoAFC_data(:, layer, :, target_ID) );
      [ twoAFC_hist_tmp, twoAFC_bins_tmp ] = ...
	  hist( twoAFC_tmp(:), num_hist_activity_bins );
      twoAFC_bins{layer, target_ID} = twoAFC_bins_tmp;
      for target_flag = 1 : max_target_flag;
	twoAFC_tmp = squeeze( exp_struct.twoAFC_data(target_flag, layer, :, target_ID) );
	twoAFC_bins_tmp = squeeze(twoAFC_bins{layer, target_ID});
	twoAFC_hist_tmp = ...
	    hist( twoAFC_tmp, twoAFC_bins_tmp );
	twoAFC_hist{target_flag, layer, target_ID} = ...
	    twoAFC_hist_tmp ./ ...
	    ( sum( twoAFC_hist_tmp ) + (sum( twoAFC_hist_tmp ) == 0) );
	if target_flag == 1
	  red_hist = 1;
	  blue_hist = 0;
	  bar_width = 0.8;
	else
	  red_hist = 0;
	  blue_hist = 1;
	  bar_width = 0.6;
	endif
	bh = bar( twoAFC_bins{layer, target_ID}, ...
		 twoAFC_hist{target_flag, layer, target_ID}, ...
		 bar_width);
	set( bh, 'EdgeColor', [red_hist 0 blue_hist] );
	set( bh, 'FaceColor', [red_hist 0 blue_hist] );
	th = title(["SOA = ", num2str(exp_struct.SOA_vals(layer)), ...
		    ", K = ", num2str(exp_struct.target_ID_vals(target_ID))]);
	axis "nolabel"
	hold on
      endfor  % target_flag
    endfor % target_ID
  endfor  % layer

  for layer = layer_ndx
    for target_ID = target_ID_ndx
      for target_flag = 1 : 2;
	twoAFC_cumsum{target_flag, layer, target_ID} = ...
	    1 - cumsum( twoAFC_hist{target_flag, layer, target_ID} );
      endfor
    endfor % target_ID
  endfor % layer

  plot_twoAFC_ideal = 0;
  if plot_twoAFC_ideal 
    subplot_index = 0;
    twoAFC_ideal_name = ['2AFC ideal observer ', num2str(exp_struct.SOS_ID)];
    fig_tmp = figure('Name', twoAFC_ideal_name);
    fig_list = [fig_list; fig_tmp];
    for layer = layer_ndx
      for target_ID = target_ID_ndx
	twoAFC_ideal{layer, target_ID} = ...
	    0.5 + 0.5 * ...
	    ( twoAFC_cumsum{1, layer, target_ID} - twoAFC_cumsum{2, layer, target_ID} );
	subplot_index = subplot_index + 1;
	subplot(len_layer_ndx, len_target_ID_ndx, subplot_index);
	bh = bar( twoAFC_bins{layer, target_ID}, twoAFC_ideal{layer, target_ID} );  
	set( bh, 'EdgeColor', [0 1 0] );
	set( bh, 'FaceColor', [0 1 0] );
	set(gca, 'YLim', [0 1]);
	th = title(["SOA = ", num2str(exp_struct.SOA_vals(layer)), ...
		    ", K = ", num2str(exp_struct.target_ID_vals(target_ID))]);
	axis "nolabel"
	hold on;
      endfor % target_ID
    endfor % layer
  endif % plot_twoAFC_ideal

  plot_2AFC_ROC = 0;
  if plot_2AFC_ROC
    subplot_index = 0;
    twoAFC_ROC_name = ['2AFC ROC ', num2str(exp_struct.SOS_ID)];
    fig_tmp = figure('Name', twoAFC_ROC_name);
    fig_list = [fig_list; fig_tmp];
    for layer = layer_ndx
      for target_ID = target_ID_ndx
	subplot_index = subplot_index + 1;
	subplot(len_layer_ndx, len_target_ID_ndx, subplot_index);
	axis([0 1 0 1]);
	th = title(["SOA = ", num2str(exp_struct.SOA_vals(layer)), ...
		    ", K = ", num2str(exp_struct.target_ID_vals(target_ID))]);
	axis "nolabel"
	axis "square";
	hold on;
	lh = plot( [0, fliplr( twoAFC_cumsum{2, layer, target_ID} ), 1], ...
		  [0, fliplr( twoAFC_cumsum{1, layer, target_ID} ), 1 ], ...
		  '-k');  
	set( lh, 'LineWidth', 2 );
      endfor % target_ID
    endfor % layer
  endif % plot_2AFC_ROC

  disp(num2str(exp_struct.SOS_ID));
  print_mean_2AFC = 0;
  if print_mean_2AFC
    for target_ID = target_ID_ndx
      for layer = layer_ndx
	for target_flag = 1 : max_target_flag
	  disp( ['mean_2AFC(', num2str(target_flag), ',', ...
			    num2str(layer), ',', num2str(target_ID), ',', ...
			    num2str(exp_struct.SOS_ID), ') = ', ...
		 num2str(mean_2AFC(target_flag, layer, target_ID)), ...
		 '+/- ', 
		 num2str(std_2AFC(target_flag, layer, target_ID)), ...
		 ] );
	endfor
      endfor % layer
    endfor % target_ID
  endif

  for target_ID = target_ID_ndx
    for layer = layer_ndx
      twoAFC_correct(layer, target_ID) = ...
	  sum( squeeze(exp_struct.twoAFC_data(1,layer, :, target_ID) >
		       exp_struct.twoAFC_data(2,layer, :, target_ID) ) ) / ...
	  ( num_trials + (num_trials == 0) );
      disp( ["twoAFC_correct(", ...
	     num2str(layer), ",",  ...
	     num2str(target_ID), ") ", ...
	     "[ID = ", num2str(exp_struct.SOS_ID), "] = ", ...
	     num2str(twoAFC_correct(layer, target_ID)) ] );
    endfor % layer
  endfor % target_ID

  ROC_struct.max_target_flag = max_target_flag;
  ROC_struct.num_layers = num_layers;
  ROC_struct.num_target_IDs = num_target_IDs;
  ROC_struct.num_trials = num_trials;
  ROC_struct.layer_ndx = layer_ndx;
  ROC_struct.target_ID_ndx = target_ID_ndx;
  ROC_struct.twoAFC_hist = twoAFC_hist;
  ROC_struct.twoAFC_cumsum = twoAFC_cumsum;
  if plot_twoAFC_ideal 
    ROC_struct.twoAFC_ideal = twoAFC_ideal;
  endif
  ROC_struct.twoAFC_correct = twoAFC_correct;

  
  twoAFC_ROC = cell(num_layers, num_target_IDs);
  twoAFC_AUC = zeros(num_layers, num_target_IDs);
  twoAFC_errorbars = zeros(num_layers, num_target_IDs);
  for layer = layer_ndx
    for target_ID = target_ID_ndx      
      twoAFC_ROC{layer, target_ID} = ...
	  [[0, fliplr( twoAFC_cumsum{2, layer, target_ID} ), 1]', ...
	   [0, fliplr( twoAFC_cumsum{1, layer, target_ID} ), 1 ]'];
      twoAFC_AUC(layer, target_ID) = ...
	  trapz(twoAFC_ROC{layer, target_ID}(:,1), ...
		twoAFC_ROC{layer, target_ID}(:,2));
      if ( twoAFC_correct(layer, target_ID) * num_trials ) ~= 0
	twoAFC_errorbar(layer, target_ID) = ...
	    sqrt( 1 - twoAFC_correct(layer, target_ID) ) / ...
	    sqrt( twoAFC_correct(layer, target_ID) * num_trials );
      else
	twoAFC_errorbar{i_fc, i_expNum}(layer, target_ID) = ...
	    0;
      endif      
    endfor % layer
  endfor % target_ID
  ROC_struct.twoAFC_ROC = twoAFC_ROC;
  ROC_struct.twoAFC_AUC = twoAFC_AUC;
  ROC_struct.twoAFC_errorbars = twoAFC_errorbars;
  
  
  for target_ID = target_ID_ndx
    disp(['target_ID = ', num2str(target_ID)]);
    for layer = layer_ndx
      disp( ['twoAFC_AUC(', num2str(layer), ...
			 ',', num2str(target_ID), ') = ', ...
	     num2str(twoAFC_AUC(layer, target_ID)), ...
	     ' +/- ', ...
	     num2str(twoAFC_errorbar(layer, target_ID)) ] );
    endfor % layer
  endfor % target_ID

  [fig_tmp] = pvp_plotROC(ROC_struct, exp_struct);
  fig_list = [fig_list; fig_tmp];

  [fig_tmp] = pvp_plotAUC(ROC_struct, exp_struct);
  fig_list = [fig_list; fig_tmp];


