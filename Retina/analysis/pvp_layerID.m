function [layerID, layerIndex] = pvp_layerID()

layerIndex = struct;
global N_LAYERS

    N_LAYERS = 31;
    
    i_layer =  1;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'Image';

    i_layer =  2;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'Cone';
 
    i_layer =  3;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'ConeSigmoidON';

    i_layer =  4;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'ConsigmoidOFF';

    i_layer =  5;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'BipolarON';

    i_layer =  6;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'BipoloarSigmoidON';

    i_layer =  7;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'Horizontal';

    i_layer =  8;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'HoriGap';


    i_layer =  9;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'HoriSigmoid';
 

    i_layer =  10;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'WFAmacrineON';
 

    i_layer =  11;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'WFAmacrineGapON';
 
    i_layer =  12;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'WFAmacrineSigmoidON';
    
    i_layer =  13;
    layerIndex.biploar = i_layer;
    layerID{ 1, i_layer } =  'GanglionON';
    
    i_layer =  14;
    layerIndex.ganglion = i_layer;
    layerID{ 1, i_layer } =  'GangliGapON';
    
    i_layer =  15;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'PAAmacrineON';

    i_layer =  16;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'PAAmaGapON';

    i_layer =  17;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'SynchronicityON';

    i_layer =  18;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'RetinaON';


    i_layer =  19;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'BipolarOFF';

    i_layer =  20;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'BipolarSigmoidOFF';

    i_layer =  21;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'WFAmacrineOFF';
 

    i_layer =  22;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'WFAmacrineGapOFF';
 

    i_layer =  23;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'WFAmacrineSigmoidOFF';
 
    i_layer =  24;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'GanglionOFF';
    
    i_layer =  25;
    layerIndex.biploar = i_layer;
    layerID{ 1, i_layer } =  'GangliGapOFF';
    
    i_layer =  26;
    layerIndex.ganglion = i_layer;
    layerID{ 1, i_layer } =  'PAAmacrineOFF';
    
    i_layer =  27;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'PAAmaGapOFF';

    i_layer =  28;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'SynchronicityOFF';

    i_layer =  29;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'RetinaOFF';

    i_layer =  30;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'SFAmacrine';

    i_layer =  31;
    layerIndex.image = i_layer;
    layerID{ 1, i_layer } =  'SFAmacrineSigmoid';
