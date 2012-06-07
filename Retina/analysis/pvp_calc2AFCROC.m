
function [twoAFC_ROC] = ...
      pvp_calc2AFCROC(twoAFC_cumsum, ...
		      layer_ndx, ...
		      target_ID_ndx, ...
		      twoAFC_str)
  
  [max_target_flag, num_layers, num_target_IDs] = ...
      size(twoAFC_cumsum);
  if ~exist("layer_ndx") || isempty(layer_ndx) || nargin < 2
    layer_ndx = 1:num_layers;
  endif
  if ~exist("taret_ID_ndx") || isempty(target_ID_ndx) || nargin < 3
    target_ID_ndx = 1:num_target_IDs;
  endif
  if ~exist("twoAFC_str") || isempty(twoAFC_str) || nargin < 4
    twoAFC_str = [];
  endif
  
  twoAFC_ROC = cell(num_layers, num_target_IDs);
  
  for layer = layer_ndx
    for target_ID = target_ID_ndx
      twoAFC_cumsum_tmp1 = twoAFC_cumsum{1, layer, target_ID};
      twoAFC_cumsum_tmp2 = twoAFC_cumsum{2, layer, target_ID};
      num_ROC_bins = length(twoAFC_cumsum_tmp1)+2;
      twoAFC_ROC_tmp = zeros(2, num_ROC_bins);
      twoAFC_ROC_tmp(1,:) = [0, fliplr(twoAFC_cumsum_tmp2), 1];
      twoAFC_ROC_tmp(2,:) = [0, fliplr(twoAFC_cumsum_tmp1), 1];
      twoAFC_ROC{layer, target_ID} = ...
	  twoAFC_ROC_tmp;
    endfor  % target_ID
  endfor  % layer
