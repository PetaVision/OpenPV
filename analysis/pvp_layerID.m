function [layerID, layerIndex] = pvp_layerID()

layerIndex = struct;
global N_LAYERS
global SPIKING_FLAG
global TRAINING_FLAG
i_layer = 0;

if ( SPIKING_FLAG == 1 )
    
    N_LAYERS = 7;
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
    
    
%%    if abs(TRAINING_FLAG) > 1
        
        N_LAYERS = N_LAYERS + 2;
        layerID = [layerID, cell(1, 2)];
        
        i_layer = i_layer + 1;
        layerIndex.l1_geisler = i_layer;
        layerID{ 1, i_layer } =  'L1G';
        
        i_layer = i_layer + 1;
        layerIndex.l1_geisler2 = i_layer;
        layerID{ 1, i_layer } =  'L1G2';
        
        
%%        if abs(TRAINING_FLAG) > 2
            
            N_LAYERS = N_LAYERS + 1;
            layerID = [layerID, cell(1, 1)];
            
            i_layer = i_layer + 1;
            layerIndex.l1_geisler2 = i_layer;
            layerID{ 1, i_layer } =  'L1G3';
            
%%        end%%if % TRAINING_FLAG
        
%%    end%%if  % TRAINING_FLAG
    
end%%if % SPIKING_FLAG