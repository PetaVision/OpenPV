function [layerID, layerIndex] = pvp_layerID()

layerIndex = struct;
global N_LAYERS
global SPIKING_FLAG
global TRAINING_FLAG
global TOPDOWN_FLAG
global G4_FLAG G6_FLAG
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
    layerIndex.l1_ODD = i_layer;
    layerID{ 1, i_layer } =  'L1_ODD';
        
    i_layer = i_layer + 1;
    layerIndex.l2_ODD = i_layer;
    layerID{ 1, i_layer } =  'L2_ODD';
        
    N_LAYERS = N_LAYERS + 1;
    layerID = [layerID, cell(1, 1)];
            
    i_layer = i_layer + 1;
    layerIndex.l3_ODD = i_layer;
    layerID{ 1, i_layer } =  'L3_ODD';
            
    if G4_FLAG 
      N_LAYERS = N_LAYERS + 1;
      layerID = [layerID, cell(1, 1)];
            
      i_layer = i_layer + 1;
      layerIndex.l4_ODD = i_layer;
      layerID{ 1, i_layer } =  'L4_ODD';
      if G6_FLAG 
	N_LAYERS = N_LAYERS + 1;
	layerID = [layerID, cell(1, 1)];
            
	i_layer = i_layer + 1;
	layerIndex.l5_ODD = i_layer;
	layerID{ 1, i_layer } =  'L5_ODD';
 
	N_LAYERS = N_LAYERS + 1;
	layerID = [layerID, cell(1, 1)];
            
	i_layer = i_layer + 1;
	layerIndex.l6_ODD = i_layer;
	layerID{ 1, i_layer } =  'L6_ODD';
      endif
    endif


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