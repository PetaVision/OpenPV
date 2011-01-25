
function [twoAFC_hist, twoAFC_bins] = ...
      pvp_calc2AFCHist(twoAFC, ...
		       layer_ndx, ...
		       target_ID_nx, ...
		       twoAFC_str)
  
  global num_hist_bins
  
  [max_target_flag, num_layers, num_trials, num_target_IDs] = ...
      size(twoAFC);
  if ~exist("layer_ndx") || isempty(layer_ndx) || nargin < 4
    layer_ndx = 1:num_layers;
  endif
  if ~exist("taret_ID_ndx") || isempty(target_ID_ndx) || nargin < 5
    target_ID_ndx = 1:num_target_IDs;
  endif
  if ~exist("twoAFC_str") || isempty(twoAFC_str) || nargin < 6
    twoAFC_str = [];
  endif
  
  twoAFC_hist = cell(max_target_flag, num_layers, num_target_IDs);
  twoAFC_bins = cell(num_layers, num_target_IDs);
  
  for layer = layer_ndx
    for target_ID = target_ID_ndx
      twoAFC_tmp = squeeze( twoAFC(:, layer, :, target_ID) );
      [ twoAFC_hist_tmp, twoAFC_bins_tmp ] = ...
	  hist( twoAFC_tmp(:), num_hist_bins );
      twoAFC_bins{layer, target_ID} = twoAFC_bins_tmp;
      for target_flag = 1 : max_target_flag;
	twoAFC_tmp = squeeze( twoAFC(target_flag, layer, :, target_ID) );
	twoAFC_hist_tmp = ...
	    hist(twoAFC_tmp(:) , ...
		 twoAFC_bins_tmp );
	twoAFC_hist_tmp = ...
	    twoAFC_hist_tmp / ...
	    ( sum( twoAFC_hist_tmp(:) )+ (sum( twoAFC_hist_tmp(:) == 0) ) );
	twoAFC_hist{target_flag, layer, target_ID} = ...
	    twoAFC_hist_tmp;
      endfor  % target_flag
    endfor  % target_ID
  endfor  % layer
  