
function [twoAFC_hist, twoAFC_bins, twoAFC_calc] = ...
      pvp_calc2AFCHist(twoAFC, ...
		       layer_ndx, ...
		       target_ID_ndx, ...
		       twoAFC_str, ...
		       baseline_layer, ...
		       percent_change_flag, ...
		       cum_change_flag)
  
  global num_hist_bins
  
  [max_target_flag, num_layers, num_trials, num_target_IDs] = ...
      size(twoAFC);
  if  nargin < 2 || ~exist("layer_ndx") || isempty(layer_ndx)
    layer_ndx = 1:num_layers;
  endif
  if nargin < 3 || ~exist("taret_ID_ndx") || isempty(target_ID_ndx)
    target_ID_ndx = 1:num_target_IDs;
  endif
  if nargin < 4 || ~exist("twoAFC_str") || isempty(twoAFC_str) 
    twoAFC_str = [];
  endif
  if nargin < 5 || ~exist("baseline_layer") || isempty(baseline_layer) 
    baseline_layer = 0;
  endif
  if nargin < 6 || ~exist("percent_change_flag") || isempty(percent_change_flag)
    percent_change_flag = 1;
  endif
  if nargin < 7 || ~exist("cum_change_flag") || isempty(cum_change_flag)
    cum_change_flag = 1;
  endif

  twoAFC_hist = cell(max_target_flag, num_layers, num_target_IDs);
  twoAFC_bins = cell(num_layers, num_target_IDs);
  
  twoAFC_calc = zeros(max_target_flag, num_layers, num_trials, num_target_IDs);
  for i_layer = 1:length(layer_ndx)
    layer = layer_ndx(i_layer);
    for target_ID = target_ID_ndx
      twoAFC_tmp = squeeze( twoAFC(:, layer, :, target_ID) );
      twoAFC_calc(:, layer, :, target_ID) = ...
	  twoAFC_tmp;
      if (abs(baseline_layer) > 0) && (layer > abs(baseline_layer))
	if cum_change_flag 
	  twoAFC_baseline = ...
	      squeeze(twoAFC(:,layer_ndx(i_layer-1),:,target_ID));
	else
	  twoAFC_baseline = ...
	      squeeze(twoAFC(:,baseline_layer,:,target_ID));
	endif %% sum_change_flag
	if baseline_layer > 0
	  twoAFC_diff = twoAFC_tmp - twoAFC_baseline;
	else
	  twoAFC_diff = twoAFC_baseline - twoAFC_tmp;
	endif %% baseline_layer > 0
	if percent_change_flag
	  twoAFC_percent = ...
	      twoAFC_diff ./ ...
	      (twoAFC_baseline + (twoAFC_baseline == 0));
	  if cum_change_flag 
	    twoAFC_calc(:, layer, :, target_ID) = ...
		squeeze(twoAFC_calc(:, layer_ndx(i_layer-1), :, target_ID)) + ...
		twoAFC_percent;
	  else
	    twoAFC_calc(:, layer, :, target_ID) = ...
		twoAFC_percent;
	  endif %% cum_change_flag
	else
	  if cum_change_flag 
	    twoAFC_calc(:, layer, :, target_ID) = ...
		squeeze(twoAFC_calc(:, layer_ndx(i_layer-1), :, target_ID)) + ...
		twoAFC_diff;
	  else
	    twoAFC_calc(:, layer, :, target_ID) = ...
		twoAFC_diff;
	  endif %% cum_change_flag
 	endif %% percent_change_flag
      endif %% (abs(baseline_layer) > 0) && (layer > abs(baseline_layer))
      twoAFC_tmp = ...
	  squeeze(twoAFC_calc(:, layer, :, target_ID));
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
  