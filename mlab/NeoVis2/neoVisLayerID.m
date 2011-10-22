function [layerID] = neoVisLayerID(max_layer)

  if nargin < 1 || ~exist("max_layer") || isempty(max_layer)
    max_layer = 7;
  endif
  
  i_layer = 0;
  layerID = cell(1, 1);
  
  i_layer = i_layer + 1;
  layerID{ 1, i_layer } =  "image";
  
  i_layer = i_layer + 1;
  layerID{ 1, i_layer } =  "retina";
  
  i_layer = i_layer + 1;
  layerID{ 1, i_layer } =  "V1";
  
  while i_layer < max_layer
    
    i_layer = i_layer + 1;
    layerID{ 1, i_layer } =  ["V1_ODD_", num2str(i_layer)];
    
  endwhile