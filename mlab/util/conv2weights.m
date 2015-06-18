%% This function was taken out of printCheckpointedSecondLevelWeights.m because some versions of octave do not support nested function handles.
%%
%% -Wesley Chavez, 6/18/15

function conv2weights (input_pvps,weights_1_pvps,weights_2_pvps) 
      fid = fopen(input_pvps,'r');
      input_header = readpvpheader(fid);
      fclose(fid);

      [weights_2_data weights_2_header] = readpvpfile(weights_2_pvps);
      [weights_1_data weights_1_header] = readpvpfile(weights_1_pvps);
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
      reverseStr = '';
      for j = 1:weights_2_header.numPatches 
         percentDone = 100*j/weights_2_header.numPatches;
         msg = sprintf('Convolving. Percent done: %3.0f', percentDone);
         if(j!=1)
            fprintf([reverseStr, msg]);
            fflush(1);
         end
         reverseStr = repmat(sprintf('\b'), 1, length(msg));
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
