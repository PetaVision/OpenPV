function [layerID, layerIndex] = pvp_layerID()

layerIndex = struct;
global N_LAYERS
global SPIKING_FLAG
global TRAINING_FLAG
global TOPDOWN_FLAG
global G2_FLAG G4_FLAG G6_FLAG 
i_layer = 0;

if ( SPIKING_FLAG == 1 )
    
    N_LAYERS = 11;
    layerID = cell(1, N_LAYERS);
    
    i_layer = i_layer + 1;
    layerIndex.retina = i_layer;
    layerID{ 1, i_layer } =  'image';
    
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
    layerIndex.s1 = i_layer;
    layerID{ 1, i_layer } =  'S1';
    
    i_layer = i_layer + 1;
    layerIndex.s1inh = i_layer;
    layerID{ 1, i_layer } =  'S1Inh';
    
    i_layer = i_layer + 1;
    layerIndex.s1inhgap = i_layer;
    layerID{ 1, i_layer } =  'S1InhGap';
    
    i_layer = i_layer + 1;
    layerIndex.c1 = i_layer;
    layerID{ 1, i_layer } =  'C1';
    
    i_layer = i_layer + 1;
    layerIndex.c1inh = i_layer;
    layerID{ 1, i_layer } =  'C1Inh';

    i_layer = i_layer + 1;
    layerIndex.c1inhgap = i_layer;
    layerID{ 1, i_layer } =  'C1InhGap';
    N_LAYERS = i_layer;

    i_layer = i_layer + 1;
    layerIndex.h1 = i_layer;
    layerID{ 1, i_layer } =  'H1';
    
    i_layer = i_layer + 1;
    layerIndex.h1inh = i_layer;
    layerID{ 1, i_layer } =  'H1Inh';

    i_layer = i_layer + 1;
    layerIndex.h1inhgap = i_layer;
    layerID{ 1, i_layer } =  'H1InhGap';
    N_LAYERS = i_layer;

    
else  % NON_SPIKING
    
    N_LAYERS = 4;
    layerID = cell(1, N_LAYERS);
    
    i_layer = i_layer + 1;
    layerIndex.retina = i_layer;
    layerID{ 1, i_layer } =  'image';
    
    i_layer = i_layer + 1;
    layerIndex.retina = i_layer;
    layerID{ 1, i_layer } =  'retina';
    
    i_layer = i_layer + 1;
    layerIndex.l1 = i_layer;
    layerID{ 1, i_layer } =  'L1';
    
    i_layer = i_layer + 1;
    layerIndex.l1Pooling1X1 = i_layer;
    layerID{ 1, i_layer } =  'L1Pooling1X1';
    
    N_LAYERS = N_LAYERS + 1;
    layerID = [layerID, cell(1, 1)];
        
    i_layer = i_layer + 1;
    layerIndex.l1Clique = i_layer;
    layerID{ 1, i_layer } =  'L1Clique';
    
    if G2_FLAG
      N_LAYERS = N_LAYERS + 1;
      layerID = [layerID, cell(1, 1)];
        
      i_layer = i_layer + 1;
      layerIndex.l2Pooling2X2 = i_layer;
      layerID{ 1, i_layer } =  'L2Pooling2X2';
    endif
    
    N_LAYERS = N_LAYERS + 1;
    layerID = [layerID, cell(1, 1)];
    
    i_layer = i_layer + 1;
    layerIndex.l2Clique = i_layer;
    layerID{ 1, i_layer } =  'L2Clique';
      
    if G4_FLAG
      N_LAYERS = N_LAYERS + 1;
      layerID = [layerID, cell(1, 1)];
        
      i_layer = i_layer + 1;
      layerIndex.l3Pooling4X4 = i_layer;
      layerID{ 1, i_layer } =  'L3Pooling4X4';
    endif
    
    N_LAYERS = N_LAYERS + 1;
    layerID = [layerID, cell(1, 1)];
      
    i_layer = i_layer + 1;
    layerIndex.l3Clique = i_layer;
    layerID{ 1, i_layer } =  'L3Clique';
      
    N_LAYERS = N_LAYERS + 1;
    layerID = [layerID, cell(1, 1)];
      
    i_layer = i_layer + 1;
    layerIndex.l4Clique = i_layer;
    layerID{ 1, i_layer } =  'L4Clique';
      
    
        
end%%if % SPIKING_FLAG