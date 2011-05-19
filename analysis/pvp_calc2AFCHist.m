
function [twoAFC_hist, twoAFC_bins] = ...
      pvp_calc2AFCHist(twoAFC, ...
		       layer_ndx, ...
		       target_ID_ndx, ...
		       twoAFC_str, ...
		       baseline_layer)
  
  global num_hist_bins
  
  [max_target_flag, num_layers, num_trials, num_target_IDs] = ...
      size(twoAFC);
  if ~exist("layer_ndx") || isempty(layer_ndx) || nargin < 2
    layer_ndx = 1:num_layers;
  endif
  if ~exist("taret_ID_ndx") || isempty(target_ID_ndx) || nargin < 3
    target_ID_ndx = 1:num_target_IDs;
  endif
  if ~exist("twoAFC_str") || isempty(twoAFC_str) || nargin < 4
    twoAFC_str = [];
  endif
  if ~exist("baseline_layer") || isempty(baseline_layer) || nargin < 5
    baseline_layer = 0;
  endif

  twoAFC_hist = cell(max_target_flag, num_layers, num_target_IDs);
  twoAFC_bins = cell(num_layers, num_target_IDs);
  
  for layer = layer_ndx
    for target_ID = target_ID_ndx
      twoAFC_tmp = squeeze( twoAFC(:, layer, :, target_ID) );
      if (baseline_layer > 0) && (layer > baseline_layer)
	twoAFC_baseline = ...
	    squeeze(twoAFC(:,baseline_layer,:,target_ID));
	twoAFC_tmp = twoAFC_baseline - twoAFC_tmp;
	%%twoAFC_tmp = ...
	%%    twoAFC_tmp ./ ...
	%%    (twoAFC_baseline + (twoAFC_baseline == 0));
      elseif (baseline_layer < 0) && (layer > abs(baseline_layer))
	twoAFC_baseline = ...
	    squeeze(twoAFC(:,abs(baseline_layer),:,target_ID));
	twoAFC_tmp = twoAFC_tmp - twoAFC_baseline;
	%%twoAFC_tmp = ...
	%%    twoAFC_tmp ./ ...
	%%    (twoAFC_baseline + (twoAFC_baseline == 0));
      endif
      [ twoAFC_hist_tmp, twoAFC_bins_tmp ] = ...
	  hist( twoAFC_tmp(:), num_hist_bins );
      twoAFC_bins{layer, target_ID} = twoAFC_bins_tmp;
      for target_flag = 1 : max_target_flag;
	twoAFC_tmp2 = squeeze( twoAFC_tmp(target_flag, :) );
	twoAFC_hist_tmp = ...
	    hist(twoAFC_tmp2(:) , ...
		 twoAFC_bins_tmp );
	twoAFC_hist_tmp = ...
	    twoAFC_hist_tmp / ...
	    ( sum( twoAFC_hist_tmp(:) )+ (sum( twoAFC_hist_tmp(:) ) == 0) );
	if num_target_IDs > 1
	  twoAFC_hist{target_flag, layer, target_ID} = ...
	      twoAFC_hist_tmp;
	else
	  twoAFC_hist{target_flag, layer} = ...
	      twoAFC_hist_tmp;
	endif
      endfor  % target_flag
    endfor  % target_ID
  endfor  % layer
  