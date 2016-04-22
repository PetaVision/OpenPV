%% This function was taken out of printCheckpointedThirdLevelWeights.m because some versions of octave do not support nested function handles.
%%
%% -Wesley Chavez, 6/18/15

function conv3weights (input_pvps,weights_1_pvps,weights_2_pvps,weights_3_pvps) 
   fid = fopen(input_pvps,'r');
   input_header = readpvpheader(fid);
   fclose(fid);
   
   [weights_1_data weights_1_header] = readpvpfile(weights_1_pvps);
   [weights_2_data weights_2_header] = readpvpfile(weights_2_pvps);
   [weights_3_data weights_3_header] = readpvpfile(weights_3_pvps);
   xstride1_0 = input_header.nx/weights_1_header.nx;
   ystride1_0 = input_header.ny/weights_1_header.ny;
   xstride2_0 = input_header.nx/weights_2_header.nx;
   ystride2_0 = input_header.ny/weights_2_header.ny;
   nxp2_0 = weights_1_header.nxp + xstride1_0 * (weights_2_header.nxp - 1);
   nyp2_0 = weights_1_header.nyp + ystride1_0 * (weights_2_header.nyp - 1);
   nxp3_0 = nyp2_0 + xstride2_0 * (weights_3_header.nxp - 1);
   nyp3_0 = nyp2_0 + ystride2_0 * (weights_3_header.nyp - 1);
   nx_conv_2 = xstride1_0 * (weights_2_header.nxp - 1) + 1;
   ny_conv_2 = ystride1_0 * (weights_2_header.nyp - 1) + 1;
   nx_conv_3 = xstride2_0 * (weights_3_header.nxp - 1) + 1;
   ny_conv_3 = ystride2_0 * (weights_3_header.nyp - 1) + 1;

   reverseStr = '';
   
   for j = 1:weights_2_header.numPatches % Convolve 2nd level weights with 1st level weights
      percentDone = 100*j/weights_2_header.numPatches;
      msg = sprintf('Convolving 2nd level. Percent done: %3.0f', percentDone);
      if(j!=1)
         fprintf([reverseStr, msg]);
         fflush(1);
      end
      reverseStr = repmat(sprintf('\b'), 1, length(msg));
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
      percentDone = 100*j/weights_3_header.numPatches;
      msg = sprintf('Convolving 3rd level. Percent done: %3.0f', percentDone);
         fprintf([reverseStr, msg]);
         fflush(1);
      reverseStr = repmat(sprintf('\b'), 1, length(msg));
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
