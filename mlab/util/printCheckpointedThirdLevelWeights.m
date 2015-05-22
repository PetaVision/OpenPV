function printCheckpointedThirdLevelWeights(input_pvps, weights_1_pvps, weights_2_pvps, weights_3_pvps)
%% printCheckpointedThirdLevelWeights.m - An analysis tool to convolve 3rd layer dictionaries (stacked HyPerLayers in PetaVision) for visualization in image space.
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
%% weights_3_pvps are your third-level weights.
%% Figures will be written to checkpointWriteDir/Checkpoint"#"/Weights
%%
%% Example:
%% input_pvps = {'Ganglion_A.pvp' 'Ganglion_A.pvp' 'Ganglion_A.pvp' 'Ganglion_A.pvp'};
%% weights_1_pvps = {'V1ToError_W.pvp' 'V1ToError_W.pvp' 'V1ToError_W.pvp' 'V1ToError_W.pvp'};
%% weights_2_pvps = {'V1BToErrorDelay0_W.pvp' 'V1BToErrorDelay100_W.pvp' 'V1BToErrorDelay200_W.pvp' 'V1BToErrorDelay300_W.pvp'};
%% weights_3_pvps = {'V14BToV14BError_W.pvp' 'V14BToV14BError_W.pvp' 'V14BToV14BError_W.pvp' 'V14BToV14BError_W.pvp'};
%% printCheckpointedThirdLevelWeights(input_pvps, weights_1_pvps, weights_2_pvps, weights_3_pvps);

   if !(size(input_pvps) == size(weights_1_pvps) == size(weights_2_pvps) == size(weights_3_pvps))
      disp("Make sure your pvp cell arrays are the same size.");
      exit;
   end

   numProcs = size(input_pvps,2);
   system('mkdir Weights');

   function convweights (input_pvps,weights_1_pvps,weights_2_pvps,weights_3_pvps) % Nested function for parallelism
      if (isempty(weights_3_pvps) || isempty(weights_2_pvps) || isempty(weights_1_pvps) || isempty(input_pvps))
         continue;
      end
      fid = fopen(input_pvps,'r');
      input_header = readpvpheader(fid);
      fclose(fid);
      
      [weights_1_data weights_1_header] = readpvpfile(weights_1_pvps,1);
      [weights_2_data weights_2_header] = readpvpfile(weights_2_pvps,1);
      [weights_3_data weights_3_header] = readpvpfile(weights_3_pvps,1);
      xstride1_0 = input_header.nx/weights_1_header.nx
      ystride1_0 = input_header.ny/weights_1_header.ny
      xstride2_0 = input_header.nx/weights_2_header.nx
      ystride2_0 = input_header.ny/weights_2_header.ny
      nxp2_0 = weights_1_header.nxp + xstride1_0 * (weights_2_header.nxp - 1)
      nyp2_0 = weights_1_header.nyp + ystride1_0 * (weights_2_header.nyp - 1)
      nxp3_0 = nyp2_0 + xstride2_0 * (weights_3_header.nxp - 1)
      nyp3_0 = nyp2_0 + ystride2_0 * (weights_3_header.nyp - 1)
      nx_conv_2 = xstride1_0 * (weights_2_header.nxp - 1) + 1 
      ny_conv_2 = ystride1_0 * (weights_2_header.nyp - 1) + 1 
      nx_conv_3 = xstride2_0 * (weights_3_header.nxp - 1) + 1 
      ny_conv_3 = ystride2_0 * (weights_3_header.nyp - 1) + 1 

      for j = 1:weights_2_header.numPatches % Convolve 2nd level weights with 1st level weights
         j
         fflush(1);
         globalpatch_2{j} = zeros(nyp2_0,nxp2_0,weights_1_header.nfp);
         for k = 1:weights_2_header.nfp
            patch_2 = weights_2_data{1}.values{1}(:,:,k,j);
            temp_conv_2 = zeros(ny_conv_2,nx_conv_2);
            temp_conv_2(1:ystride1_0:end,1:xstride1_0:end) = patch_2;
            globalpatch_2{j} = globalpatch_2{j} + convn(weights_1_data{1}.values{1}(:,:,:,k),temp_conv_2,'full');
         end      
      end
     
      h_weightsbyindex = figure;
      subplot_x = ceil(sqrt(weights_3_header.numPatches));
      subplot_y = ceil(weights_3_header.numPatches/subplot_x);

      for j = 1:weights_3_header.numPatches
         j
         fflush(1);
         globalpatch_3 = zeros(nyp3_0,nxp3_0,weights_1_header.nfp);
         for k = 1:weights_3_header.nfp % Convolve 3rd level weights with 2nd level weights
            patch_3 = weights_3_data{1}.values{1}(:,:,k,j);
            temp_conv_3 = zeros(ny_conv_3,nx_conv_3);
            temp_conv_3(1:ystride2_0:end,1:xstride2_0:end) = patch_3;
            globalpatch_3 = globalpatch_3 + convn(globalpatch_2{k},temp_conv_3,'full');
         end      
         globalpatch_3 = globalpatch_3-min(globalpatch_3(:)); % Normalize weights individually
         globalpatch_3 = globalpatch_3*255/max(globalpatch_3(:));
         globalpatch_3 = uint8(permute(globalpatch_3,[2 1 3]));
         subplot(subplot_y,subplot_x,j);
         imshow(globalpatch_3);
      end
      suffix='_W.pvp';
      t = weights_3_data{1}.time;
      [startSuffix1,endSuffix] = regexp(weights_1_pvps,suffix);
      [startSuffix2,endSuffix] = regexp(weights_2_pvps,suffix);
      [startSuffix3,endSuffix] = regexp(weights_3_pvps,suffix);
      outFile = ['Weights/' weights_3_pvps(1:startSuffix3-1) '_' weights_2_pvps(1:startSuffix2-1) '_' weights_1_pvps(1:startSuffix1-1) '_WeightsByFeatureIndex_' sprintf('%.08d',t) '.png']; 
      print(h_weightsbyindex,outFile);
   end

   parcellfun(numProcs,@convweights,input_pvps,weights_1_pvps,weights_2_pvps,weights_3_pvps,'UniformOutput',0);
end
