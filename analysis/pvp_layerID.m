function [layerID, layerIndex] = pvp_layerID()

layerIndex = struct;
global N_LAYERS
global SPIKING_FLAG
global TRAINING_FLAG
global TOPDOWN_FLAG
global G4_FLAG
i_layer = 0;

if ( SPIKING_FLAG == 1 )
    
    N_LAYERS = 8;
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
    layerIndex.l1 = i_layer;
    layerID{ 1, i_layer } =  'L1';
    
    i_layer = i_layer + 1;
    layerIndex.l1inhff = i_layer;
    layerID{ 1, i_layer } =  'L1InhFF';
    
    i_layer = i_layer + 1;
    layerIndex.l1inh = i_layer;
    layerID{ 1, i_layer } =  'L1Inh';

    i_layer = i_layer + 1;
    layerIndex.l1inh = i_layer;
    layerID{ 1, i_layer } =  'L1InhGap';
    
else  % NON_SPIKING
    
    N_LAYERS = 3;
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
    
    N_LAYERS = N_LAYERS + 2;
    layerID = [layerID, cell(1, 2)];
        
    i_layer = i_layer + 1;
    layerIndex.l1_geisler = i_layer;
    layerID{ 1, i_layer } =  'L1G';
        
    i_layer = i_layer + 1;
    layerIndex.l2_geisler = i_layer;
    layerID{ 1, i_layer } =  'L2G';
        
    N_LAYERS = N_LAYERS + 1;
    layerID = [layerID, cell(1, 1)];
            
    i_layer = i_layer + 1;
    layerIndex.l3_geisler = i_layer;
    layerID{ 1, i_layer } =  'L3G';
            
    if G4_FLAG 
    N_LAYERS = N_LAYERS + 1;
    layerID = [layerID, cell(1, 1)];
            
    i_layer = i_layer + 1;
    layerIndex.l4_geisler = i_layer;
    layerID{ 1, i_layer } =  'L4G';
    end%%if

    if TOPDOWN_FLAG
      
      N_LAYERS = N_LAYERS + 3 + G4_FLAG;
      layerID = [layerID, cell(1, 3 + G4_FLAG)];
        
      i_layer = i_layer + 1;
      layerIndex.l1_topdown = i_layer;
      layerID{ 1, i_layer } =  'L1TD';
        
      i_layer = i_layer + 1;
      layerIndex.l2_topdpwn = i_layer;
      layerID{ 1, i_layer } =  'L2TD';
        
      i_layer = i_layer + 1;
      layerIndex.l3_topdpwn = i_layer;
      layerID{ 1, i_layer } =  'L3TD';

      if G4_FLAG 
      i_layer = i_layer + 1;
      layerIndex.l4_topdpwn = i_layer;
      layerID{ 1, i_layer } =  'L4TD';
      end%%if
        
    end%%if  % TOPDOWN_FlAG
            
end%%if % SPIKING_FLAG