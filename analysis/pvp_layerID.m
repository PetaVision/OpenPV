function [layerID, layerIndex] = pvp_layerID()

  layerIndex = struct;
  
  global N_LAYERS
  N_LAYERS = 7;
  
  i_layer = 0;
  layerID = cell(1, N_LAYERS);

  i_layer = i_layer + 1;
  layerIndex.retina = i_layer;
  layerID{ 1, i_layer } =  'retina';

  i_layer = i_layer + 1;
  layerIndex.lgn = i_layer;
  layerID{ 1, i_layer } =  'LGN';

  i_layer = i_layer + 1;
  layerIndex.lgninhff = i_layer;
  layerID{ 1, i_layer } =  'LGNInhFF';

  i_layer = i_layer + 1;
  layerIndex.lgninh = i_layer;
  layerID{ 1, i_layer } =  'LGNInh';

  i_layer = i_layer + 1;
  layerIndex.l1 = i_layer;
  layerID{ 1, i_layer } =  'L1';

  i_layer = i_layer + 1;
  layerIndex.l1inhff = i_layer;
  layerID{ 1, i_layer } =  'L1InhFF';

  i_layer = i_layer + 1;
  layerIndex.l1inh = i_layer;
  layerID{ 1, i_layer } =  'L1Inh';

