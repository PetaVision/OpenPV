function [fig_list_tmp] = ...
      pvp_plot2AFCROC(twoAFC_ROC, ...
		      twoAFC_bins, ...
		      layer_ndx, ...
		      target_ID_ndx, ...
		      twoAFC_test_str)
  
  [num_layers, num_target_IDs] = size(twoAFC_ROC);
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
  twoAFC_ROC_name = ['2AFC ROC ', twoAFC_str];
  fig_list = [];
  for target_ID = target_ID_ndx
    fig_tmp = figure('Name', twoAFC_ROC_name);
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
      twoAFC_ROC_tmp = ...
	  twoAFC_ROC{layer, target_ID};
      twoAFC_bins_tmp = ...
	  twoAFC_bins{layer, target_ID};
      lh = plot(twoAFC_ROC_tmp(1,:), ...
		twoAFC_ROC_tmp(2,:), ...
		'-k');  
      set( lh, 'LineWidth', 2 );
    endfor % layer
  endfor % target_ID

  