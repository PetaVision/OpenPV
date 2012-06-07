
function [twoAFC_cumsum, twoAFC_ideal] = ...
      pvp_calc2AFCCumsum(twoAFC_hist, ...
			 layer_ndx, ...
			 target_ID_ndx, ...
			 twoAFC_str)
  
  [max_target_flag, num_layers, num_target_IDs] = ...
      size(twoAFC_hist);
  if ~exist("layer_ndx") || isempty(layer_ndx) || nargin < 2
    layer_ndx = 1:num_layers;
  endif
  if ~exist("taret_ID_ndx") || isempty(target_ID_ndx) || nargin < 3
    target_ID_ndx = 1:num_target_IDs;
  endif
  if ~exist("twoAFC_str") || isempty(twoAFC_str) || nargin < 4
    twoAFC_str = [];
  endif
  
  twoAFC_cumsum = cell(max_target_flag, num_layers, num_target_IDs);
  twoAFC_ideal = cell(num_layers, num_target_IDs);
  
  for layer = layer_ndx
    for target_ID = target_ID_ndx
      for target_flag = 1 : max_target_flag
	twoAFC_hist_tmp = ...
	    twoAFC_hist{target_flag, layer, target_ID};
	twoAFC_cumsum_tmp = ...
	    1 - cumsum( twoAFC_hist_tmp );
	twoAFC_cumsum{target_flag, layer, target_ID} = ...
	    twoAFC_cumsum_tmp;
      endfor
      twoAFC_cumsum_tmp1 = twoAFC_cumsum{1, layer, target_ID};
      twoAFC_cumsum_tmp2 = twoAFC_cumsum{2, layer, target_ID};
      twoAFC_ideal_tmp = ...
	  0.5 + 0.5 * ...
	  ( twoAFC_cumsum_tmp1- twoAFC_cumsum_tmp2 );
      twoAFC_ideal{layer, target_ID} = ...
	  twoAFC_ideal_tmp;
    endfor  % target_ID
  endfor  % layer
  