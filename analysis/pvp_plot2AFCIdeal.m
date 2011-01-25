function [fig_list_tmp] = ...
      pvp_plot2AFCIdeal(twoAFC_ideal, ...
			twoAFC_bins, ...
			layer_ndx, ...
			target_ID_ndx, ...
			twoAFC_test_str)
  
  [max_target_flag, num_layers, num_target_IDs] = size(twoAFC_cumsum);
  if ~exist("layer_ndx") || isempty(layer_ndx) || nargin < 3
    layer_ndx = 1:num_layers;
  endif
  if ~exist("taret_ID_ndx") || isempty(target_ID_ndx) || nargin < 4
    target_ID_ndx = 1:num_target_IDs;
  endif
  if ~exist("twoAFC_str") || isempty(twoAFC_str) || nargin < 5
    twoAFC_str = [];
  endif
  
  num_subplots = length(layer_ndx);
  twoAFC_ideal_name = ['2AFC ideal ', twoAFC_str];
  fig_list = [];
  for target_ID = target_ID_ndx
    fig_tmp = figure('Name', twoAFC_ideal_name);
    if isempty(fig_list)
      fig_list = fig_tmp;
    else
      fig_list = [fig_list; fig_tmp];
    endif
    subplot_index = 0;
    for layer = layer_ndx
      subplot_index = subplot_index + 1;
      subplot(num_subplots, 1, subplot_index);
      axis "nolabel"
      twoAFC_ideal_tmp = ...
	  twoAFC_ideal{layer, target_ID};
      twoAFC_bins_tmp = ...
	  twoAFC_bins{layer, target_ID};
      bh = bar( twoAFC_bins_tmp, twoAFC_ideal_tmp );  
      set( bh, 'EdgeColor', [0 1 0] );
      set( bh, 'FaceColor', [0 1 0] );
      set(gca, 'YLim', [0 1]);
      hold on;
    endfor % layer
  endfor % target_ID

