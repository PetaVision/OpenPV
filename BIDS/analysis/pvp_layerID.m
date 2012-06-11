function [layerID, layerIndex] = pvp_layerID()

layerIndex = struct;
global N_LAYERS
global SPIKING_FLAG
i_layer = 0;

    N_LAYERS = 4;
    layerID = cell(1, N_LAYERS);
    
	i_layer = i_layer + 1;
	layerIndex.image = i_layer;
	layerID{ 1, i_layer } =  'image';

	i_layer = i_layer + 1;
	layerIndex.cone = i_layer;
	layerID{ 1, i_layer } =  'cone';

	i_layer = i_layer + 1;
	layerIndex.ganglion = i_layer;
	layerID{ 1, i_layer } =  'ganglion';

	i_layer = i_layer + 1;
	layerIndex.retina = i_layer;
	layerID{ 1, i_layer } =  'retina';

   
