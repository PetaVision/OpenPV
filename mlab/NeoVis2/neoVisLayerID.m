function [layerID] = neoVisLayerID(max_layer)

  if nargin < 1 || ~exist("max_layer") || isempty(max_layer)
    max_layer = 9;
  endif
  
  i_layer = 0;
  layerID = cell(1, 1);
  
  i_layer = i_layer + 1;
  layerID{ 1, i_layer } =  "image";
  
  i_layer = i_layer + 1;
  layerID{ 1, i_layer } =  "retina";
  
  i_layer = i_layer + 1;
  layerID{ 1, i_layer } =  "V1";
  
  i_ODD = 0;
  while i_layer < 7
    
    i_ODD = i_ODD + 1;
    i_layer = i_layer + 1;
    layerID{ 1, i_layer } =  ["ODD_", num2str(i_ODD)];
    
    i_layer = i_layer + 1;
    layerID{ 1, i_layer } =  ["Pool_", num2str(i_ODD)];

  endwhile

  while i_layer < 9
    
    i_ODD = i_ODD + 1;
    i_layer = i_layer + 1;
    layerID{ 1, i_layer } =  ["ODD_", num2str(i_ODD)];
    
  endwhile
