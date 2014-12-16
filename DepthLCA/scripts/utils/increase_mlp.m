addpath('~/workspace/PetaVision/mlab/util/');

checkpointDir = "/nh/compneuro/Data/Depth/LCA/Checkpoints/saved_stack_mlp/";
outDir = "/nh/compneuro/Data/Depth/LCA/Checkpoints/saved_stack_mlp_double/";

weights_list = ...
    { ...
     ["V1S2ToLeftError_W"]; ...
     ["V1S4ToLeftError_W"]; ... 
     ["V1S8ToLeftError_W"]; ... 
     ["V1S2ToRightError_W"]; ...
     ["V1S4ToRightError_W"]; ...
     ["V1S8ToRightError_W"]; ...
     };

new_nf_list = ...
   [64; 128; 256; 64; 128; 256;]

wMinInit = -1.0;
wMaxInit = 1.0;
sparseFraction  = .9;

for Wi = 1:length(weights_list)
   inFilename = [checkpointDir, weights_list{Wi}, '.pvp']
   outFilename = [outDir, weights_list{Wi}, '.pvp']
   newNf = new_nf_list(Wi);
   increaseNumFeatInFile(inFilename, newNf, wMinInit, wMaxInit, sparseFraction, outFilename);
end
   

