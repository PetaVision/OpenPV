function printCheckpointedFirstLevelWeights(weightspvpfile)
   % printCheckpointedFirstLevelWeights.m - Prints visualization of weights (dictionary) from a checkpointed weights pvp file.
   %
   % -Wesley Chavez 5/21/15
   %
   % -----Hints-----
   % BEFORE RUNNING THIS SCRIPT FOR THE FIRST TIME:
   % Add the PetaVision mlab/util directory to your .octaverc or .matlabrc file, which is probably in your home (~) directory.
   % Example:    addpath("~/_/_/PetaVision/mlab/util")
   %
   % IMPORTANT: 
   % Run this script in a PetaVision checkpoint (checkpointWriteDir/Checkpoint"#").  checkpointWriteDir is specified in your pv.params file.
   % Figures will be written to checkpointWriteDir/Checkpoint"#"/Weights
   % 
   % Example: 
   % weightspvpfile = 'V1ToError_W.pvp';
   % printCheckpointedFirstLevelWeights(weightspvpfile);
   
   system('mkdir Weights');
   weightsdata = readpvpfile(weightspvpfile);
   t = weightsdata{1}.time;
   weightsnumpatches = size(weightsdata{1}.values{1})(4)
   subplot_x = ceil(sqrt(weightsnumpatches));
   subplot_y = ceil(weightsnumpatches/subplot_x);
   h_weightsbyindex = figure;
   
   for j = 1:weightsnumpatches  % Normalize and plot weights by weight index
      weightspatch{j} = weightsdata{1}.values{1}(:,:,:,j);
      weightspatch{j} = weightspatch{j}-min(weightspatch{j}(:));
      weightspatch{j} = weightspatch{j}*255/max(weightspatch{j}(:));
      weightspatch{j} = uint8(permute(weightspatch{j},[2 1 3]));
      subplot(subplot_y,subplot_x,j);
      imshow(weightspatch{j});
   end

   suffix='_W.pvp';
   [startSuffix,endSuffix] = regexp(weightspvpfile,suffix);
   outFile = ['Weights/' weightspvpfile(1:startSuffix-1) '_WeightsByFeatureIndex_' sprintf('%.08d',t) '.png']
   print(h_weightsbyindex,outFile);

end
