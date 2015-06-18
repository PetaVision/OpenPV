function printCheckpointedSecondLevelWeights(input_pvps, weights_1_pvps, weights_2_pvps, numProcs)
%% printCheckpointedSecondLevelWeights.m - An analysis tool to convolve 2nd layer dictionaries (stacked HyPerLayers in PetaVision) for visualization in image space.
%%  -Wesley Chavez 5/21/15
%%
%% -----Hints-----
%% BEFORE RUNNING THIS SCRIPT FOR THE FIRST TIME:
%% Add the PetaVision mlab/util directory to your .octaverc or .matlabrc file, which is probably in your home (~) directory.
%% Example:    addpath("~/_/_/PetaVision/mlab/util")
%%
%% IMPORTANT: 
%% Run this script in a PetaVision checkpoint (checkpointWriteDir/Checkpoint"#").  checkpointWriteDir is specified in your pv.params file.
%% Change "input_pvps" to the .pvp files that are input(s) to your first-level Error layer(s).
%% weights_1_pvps are your first-level weights.
%% weights_2_pvps are your second-level weights.
%% Figures will be written to checkpointWriteDir/Checkpoint"#"/Weights
%% 
%% Example:
%% input_pvps = {'Ganglion_A.pvp' 'Ganglion_A.pvp' 'Ganglion_A.pvp' 'Ganglion_A.pvp'};
%% weights_1_pvps = {'V1ToError_W.pvp' 'V1ToError_W.pvp' 'V1ToError_W.pvp' 'V1ToError_W.pvp'};
%% weights_2_pvps = {'V1BToErrorDelay0_W.pvp' 'V1BToErrorDelay100_W.pvp' 'V1BToErrorDelay200_W.pvp' 'V1BToErrorDelay300_W.pvp'};
%% printCheckpointedSecondLevelWeights(input_pvps, weights_1_pvps, weights_2_pvps, 4);

   if !(size(input_pvps) == size(weights_1_pvps) == size(weights_2_pvps))
      disp("Make sure your pvp cell arrays are the same size.");
      exit;
   end

   system('mkdir Weights');

   parcellfun(numProcs,@conv2weights,input_pvps,weights_1_pvps,weights_2_pvps,'UniformOutput',0);
end
