
function [fig_list] = ...
      pvp_plot2AFCHist(twoAFC_hist, ...
		       twoAFC_bins, ...
		       layer_ndx, ...
		       target_ID_ndx, ...
		       twoAFC_str)

  global num_hist_bins

  [max_target_flag, num_layers, num_target_IDs] = size(twoAFC_hist);
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
  twoAFC_hist_name = ['2AFC hist ', twoAFC_str];
  fig_list = [];
  for target_ID = target_ID_ndx
    fig_tmp = figure('Name', twoAFC_hist_name);
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
      twoAFC_bins_tmp = ...
	  twoAFC_bins{layer, target_ID};
      for target_flag = 1 : max_target_flag;
	twoAFC_hist_tmp = ...
	    twoAFC_hist{target_flag, layer, target_ID};
	if target_flag == 1
	  red_hist = 1;
	  blue_hist = 0;
	  bar_width = 0.8;
	else
	  red_hist = 0;
	  blue_hist = 1;
	  bar_width = 0.6;
	endif
	bh = bar(twoAFC_bins_tmp, ...
		 twoAFC_hist_tmp, ...
		 bar_width);
	set( bh, 'EdgeColor', [red_hist 0 blue_hist] );
	set( bh, 'FaceColor', [red_hist 0 blue_hist] );
	hold on
      endfor  % target_flag
    endfor  % layer
  endfor  % target_ID
