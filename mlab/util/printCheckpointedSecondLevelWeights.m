function printCheckpointedSecondLevelWeights(input_pvps, weights_1_pvps, weights_2_pvps)
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
%% printCheckpointedSecondLevelWeights(input_pvps, weights_1_pvps, weights_2_pvps);

   if !(size(input_pvps) == size(weights_1_pvps) == size(weights_2_pvps))
      disp("Make sure your pvp cell arrays are the same size.");
      exit;
   end

   numProcs = size(input_pvps,2);
   system('mkdir Weights');

   function convweights (input_pvps,weights_1_pvps,weights_2_pvps) % Nested function for parallelism 
      if (isempty(weights_2_pvps) || isempty(weights_1_pvps) || isempty(input_pvps))
         continue;
      end
      fid2 = fopen(input_pvps,'r');
      input_header = readpvpheader(fid2);
      fclose(fid2);

      [weights_2_data weights_2_header] = readpvpfile(weights_2_pvps,1);
      [weights_1_data weights_1_header] = readpvpfile(weights_1_pvps,1);
      t = weights_2_data{1}.time;
      xstride = input_header.nx/weights_1_header.nx;
      ystride = input_header.ny/weights_1_header.ny;
      nxp_global = weights_1_header.nxp + xstride * (weights_2_header.nxp - 1);
      nyp_global = weights_1_header.nyp + ystride * (weights_2_header.nyp - 1);
      nx_conv_2 = xstride * (weights_2_header.nxp - 1) + 1; 
      ny_conv_2 = ystride * (weights_2_header.nyp - 1) + 1; 
      subplot_x = ceil(sqrt(weights_2_header.numPatches));
      subplot_y = ceil(weights_2_header.numPatches/subplot_x);
      h_weightsbyindex = figure;

      for j = 1:weights_2_header.numPatches 
         j
         fflush(1);
         globalpatch = zeros(nyp_global,nxp_global,weights_1_header.nfp);
         for k = 1:weights_2_header.nfp % Convolve weights 
            patch_2 = weights_2_data{1}.values{1}(:,:,k,j);
            temp_conv_2 = zeros(ny_conv_2,nx_conv_2);
            temp_conv_2(1:ystride:end,1:xstride:end) = patch_2;
            globalpatch = globalpatch + convn(weights_1_data{1}.values{1}(:,:,:,k),temp_conv_2,'full');
         end
         globalpatch = globalpatch-min(globalpatch(:)); % Normalize weights individually 
         globalpatch = globalpatch*255/max(globalpatch(:));
         globalpatch = uint8(permute(globalpatch,[2 1 3]));
         subplot(subplot_y,subplot_x,j);
         imshow(globalpatch);
      end
      suffix = '_W.pvp';
      [startSuffix1,endSuffix] = regexp(weights_1_pvps,suffix);
      [startSuffix2,endSuffix] = regexp(weights_2_pvps,suffix);
      outFile = ['Weights/' weights_2_pvps(1:startSuffix2-1) '_' weights_1_pvps(1:startSuffix1-1) '_WeightsByFeatureIndex_' sprintf('%.08d',t) '.png'];
      print(h_weightsbyindex,outFile);
   end

   parcellfun(numProcs,@convweights,input_pvps,weights_1_pvps,weights_2_pvps,'UniformOutput',0);
end
