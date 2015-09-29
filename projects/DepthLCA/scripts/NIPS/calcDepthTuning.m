function [outVals, kurtVals, peakMeanVals, peakAreaVals, activationFreq] = calcDepthTuning(v1ActFile, depthFile, sampleDim, numDepthBins, skipNum, numAreaPeak, numEpochs)
   addpath('~/workspace/PetaVision/mlab/util');

   %[left_w_data, left_hdr] = readpvpfile(dictPvpFiles{1});
   %[right_w_data, right_hdr] = readpvpfile(dictPvpFiles{2});
   %[v1_data, v1_hdr] = readpvpfile(v1ActFile, 0, 20, 1);
   %[depth_data, depth_hdr] = readpvpfile(depthFile, 0, 20, 1);

   disp(['Reading pvp files']);
   [v1_data, v1_hdr] = readpvpfile(v1ActFile);
   [depth_data, depth_hdr] = readpvpfile(depthFile);

   %sampleDim must be odd
   assert(mod(sampleDim, 2) == 1);

   %if(v1_data{1}.time ~= depth_data{1}.time)
      %v1's first write is at time 0, remove
      v1_data(1) = [];
   %end

   %assert(v1_data{1}.time == depth_data{1}.time);
   %assert(v1_data{2}.time == depth_data{2}.time);
   assert(v1_hdr.nxGlobal == depth_hdr.nxGlobal);
   assert(v1_hdr.nyGlobal == depth_hdr.nyGlobal);
   assert(depth_hdr.nf == 1);

   numTimes = length(depth_data);
   assert(numTimes * skipNum <= length(v1_data));

   nx = v1_hdr.nxGlobal;
   ny = v1_hdr.nyGlobal;
   nf = v1_hdr.nf;

   %Splitting nf up into different epochs
   %TODO make this take arbitrary epoch numbers
   assert(mod(nf, numEpochs) == 0);
   nfEpoch = nf/numEpochs;

   %Data structure to hold stat values
   outVals = zeros(nf, numDepthBins - 1, sampleDim * sampleDim);
   peakMeanVals = zeros(nf, 1);
   peakAreaVals= zeros(nf, 1);
   kurtVals = zeros(nf, 1);
   activationFreq = zeros(nf, 1);

   for iepoch = 1:numEpochs
      startNf = ((iepoch - 1) * nfEpoch) + 1;
      endNf = iepoch * nfEpoch;

      disp(['Reshaping pvp files']);

      v1_act = zeros(numTimes, nfEpoch, nx, ny); 

      act_idx = 1;
      depth_act = zeros(numTimes, nx, ny);

      for(t = skipNum:skipNum:numTimes*skipNum)
         if(v1_hdr.filetype == 4)
            v1_vals = permute(v1_data{t}.values, [3, 1, 2]); 
            v1_act(act_idx, :, :, :) = v1_vals(startNf:endNf, :, :);
         elseif(v1_hdr.filetype == 6)
            %TODO check if layer is sparse
            N = nx * ny * nf;
            active_ndx = v1_data{t}.values(:, 1);
            active_vals = v1_data{t}.values(:, 2);
            tmp_v1 = full(sparse(active_ndx+1, 1, active_vals, N, 1, N));
            %pv does [nf, nx, ny] ordering
            tmp_v1 = reshape(tmp_v1, [nf, nx, ny]);
            v1_act(act_idx, :, :, :) = tmp_v1(startNf:endNf, :, :);
         else
            disp(['Error: filetype ', v1_hdr.filetype, ' not supported']);
            assert(-1);
         endif
         depth_act(act_idx, :, :) = depth_data{act_idx}.values;
         act_idx += 1;
      endfor

      %Take out unnessessary depth_data and v1_data for memory
      %clear depth_data v1_data tmp_v1 active_ndx active_vals;

      disp(['Histograming Depth']);
      %bin depth_act into 64 bins, and use the index as the new matrix
      %Note that 1 is now the DNC region
      [drop, depth_act] = histc(depth_act, 0:(1/numDepthBins):1);

      %Given a depth and neuron, we want to find the family of the maximum activity
      %Final out matrix will be [neuronIdx, depth, sample patch]
      edgeMag = floor(sampleDim / 2);

      for(ni = 1:nfEpoch)
         currOutVals = zeros(numDepthBins - 1, sampleDim * sampleDim);
         disp(['Calculating neuron ', num2str(ni+startNf-1), ' out of ' , num2str(nf)]);
         %Take a slice and squeeze out singleton dimension
         v1_slice = reshape(v1_act(:, ni, :, :), numTimes, nx, ny);
         assert(size(v1_slice) == size(depth_act));

         %Grab num activations for this neuron
         activationFreq(startNf+ni-1) = nnz(v1_slice)/(numTimes * nx * ny);

         for(di = 2:numDepthBins)
            %Find locations of a given depth
            targetDepthIdx = find(depth_act == di);
            %If depth doesn't exist, skip
            if(isempty(targetDepthIdx))
               continue
            end
            %t, x, and y are subscripts for locations where depth == di
            [t, x, y] = ind2sub([numTimes nx ny], targetDepthIdx);
            %Variable to index into outVals
            sampleDimIdx = 1;
            for(diffx = -edgeMag:edgeMag)
               %Calculate the difference and set to new x
               newx = x + diffx;
               %Check for border conditions
               newx(find(newx <= 0)) = 1;
               newx(find(newx > nx)) = nx;
               for(diffy = -edgeMag:edgeMag)
                  newy = y + diffy;
                  newy(find(newy <= 0)) = 1;
                  newy(find(newy > ny)) = ny;
                  %Convert back to linear index
                  linIdxs = sub2ind([numTimes, nx, ny], t, newx, newy);
                  %Find maximum activity of v1 activity slice
                  maxVal = max(abs(v1_slice(linIdxs)(:)));
                  if(length(maxVal) ~= 1)
                     disp('maxVal not size of 1');
                     keyboard
                  end
                  %Set to currOutVals
                  currOutVals(di-1, sampleDimIdx) = maxVal;
                  %Increment sampleDimIdx
                  sampleDimIdx += 1;
               end
            end
            %Make sure the idx matches up and reset
            assert(sampleDimIdx == (sampleDim * sampleDim)+1);
            sampleDimIdx = 1;
         end

         %currOutVals is a matrix with [Depth, # of lines per position]

         %Normalize peak to 1
         %outVals(ni, :, :) = currOutVals ./ max(currOutVals(:));

         outVals(startNf+ni-1, :, :) = currOutVals;

         %Calculate kurtosis, finding a kurtosis per sampleDim 
         kurtVals(startNf+ni-1) = mean(kurtosis(currOutVals, 1, 1));
         %Calculate peak - mean
         meanPosVals = mean(currOutVals, 2);
         %Scale values
         normMeanPosVals = meanPosVals/max(meanPosVals);
         peakMeanVals(startNf+ni-1) = 1-mean(normMeanPosVals);

         %Calc area under peak vals
         [drop, idx] = max(normMeanPosVals);
         startIdx = idx - numAreaPeak;
         endIdx = idx + numAreaPeak;
         %Edge cases
         if(startIdx < 1)
            startIdx = 1;
         end
         if(endIdx > numDepthBins - 1)
            endIdx = numDepthBins - 1;
         end
         peakAreaVals(startNf+ni-1) = mean(normMeanPosVals(startIdx:endIdx)) - mean(normMeanPosVals);
      end
   end
end

