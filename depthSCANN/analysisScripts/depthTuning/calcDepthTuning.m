function [outVals, kurtVals, peakMeanVals] = calcDepthTuning(v1ActFile, depthFile, sampleDim, numDepthBins, skipNum)
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
   %   %v1's first write is at time 0, remove
   %   v1_data(1) = [];
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

   disp(['Reshaping pvp files']);

   splitIdx = 256;
   nf1 = splitIdx;
   nf2 = nf-splitIdx;

   %Use 2 arrays for storing v1 act
   v1_act_1 = zeros(numTimes, nf1, nx, ny); 
   v1_act_2 = zeros(numTimes, nf2, nx, ny); 

   keyboard

   %%Reshape cell structs to be one big array
   %try
   %   v1_act = zeros(numTimes, nf, nx, ny);
   %catch e
   %   keyboard;
   %end

   act_idx = 1;
   depth_act = zeros(numTimes, nx, ny);
   for(t = skipNum:skipNum:numTimes*skipNum)
      if(v1_hdr.filetype == 4)
         v1_vals = permute(v1_data{t}.values, [3, 1, 2]); 
         v1_act_1(act_idx, :, :, :) = v1_vals(1:splitIdx, :, :);
         v1_act_2(act_idx, :, :, :) = v1_vals(splitIdx+1:end, :, :);
      elseif(v1_hdr.filetype == 6)
         %TODO check if layer is sparse
         N = nx * ny * nf;
         active_ndx = v1_data{t}.values(:, 1);
         active_vals = v1_data{t}.values(:, 2);
         tmp_v1 = full(sparse(active_ndx+1, 1, active_vals, N, 1, N));
         %pv does [nf, nx, ny] ordering
         tmp_v1 = reshape(tmp_v1, [nf, nx, ny]);
         v1_act_1(act_idx, :, :, :) = tmp_v1(1:splitIdx, :, :);
         v1_act_2(act_idx, :, :, :) = tmp_v1(splitIdx+1:end, :, :);
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

   %Data structure to hold stat values
   peakMeanVals = zeros(nf, 1);
   kurtVals = zeros(nf, 1);

   %Given a depth and neuron, we want to find the family of the maximum activity
   %Final out matrix will be [neuronIdx, depth, sample patch]
   edgeMag = floor(sampleDim / 2);
   outVals = zeros(nf, numDepthBins - 1, sampleDim * sampleDim);

   for(ni = 1:nf)
      currOutVals = zeros(numDepthBins - 1, sampleDim * sampleDim);
      disp(['Calculating neuron ', num2str(ni), ' out of ' , num2str(nf)]);
      %Take a slice and squeeze out singleton dimension
      if(ni <= splitIdx)
         v1_slice = reshape(v1_act_1(:, ni, :, :), numTimes, nx, ny);
      else
         v1_slice = reshape(v1_act_2(:, ni-splitIdx, :, :), numTimes, nx, ny);
      end
      assert(size(v1_slice) == size(depth_act));
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

      outVals(ni, :, :) = currOutVals;

      %Calculate kurtosis, finding a kurtosis per sampleDim 
      kurtVals(ni) = mean(kurtosis(currOutVals, 1, 1));
      %Calculate peak - mean
      meanPosVals = mean(currOutVals, 2);
      %Scale values
      normMeanPosVals = meanPosVals/max(meanPosVals);
      peakMeanVals(ni) = 1-mean(normMeanPosVals);

      %handle = figure;
      %if(size(left_w_data{1}.values{1}, 3) == 1)
      %   colormap(gray)
      %end

      %subplot(2, 2, 1);
      %imagesc(permute(left_w_data{1}.values{1}(:, :, :, ni), [2, 1, 3]));
      %subplot(2, 2, 2);
      %imagesc(permute(right_w_data{1}.values{1}(:, :, :, ni), [2, 1, 3]));
      %subplot(2, 2, [3, 4]);
      %hold on;
      %for(plotIdx = 1:sampleDim * sampleDim)
      %   plot(outVals(:, plotIdx));
      %end
      %hold off;

      %title(['Depth Vs Activation   Kurtosis: ', num2str(kurtVals(ni)), '  PeakMean: ', num2str(peakMeanVals(ni))]);
      %xlabel('Far Depth                 Near Depth');
      %ylabel('Activation');
      %print(handle, [plotOutDir, num2str(ni), '.png']);
      %close(handle)
   end
   %%Return all useful variables
   %%return [outVals, kurtVals, peakMeanVals]


   %%%Kurtosis
   %%Write mean and std of kurtosis in file
   %kurtFile = fopen([plotOutDir, 'kurtosis.txt'], 'w');
   %fprintf(kurtFile, 'kurtosis: %f +- %f\n', mean(kurtVals(:)), std(kurtVals(:)));
   %[sortedKurt, sortedKurtIdxs] = sort(kurtVals, 'descend');

   %%Write ranking by kurtosis
   %for(ni = 1:nf)
   %   fprintf(kurtFile, '%d: %f\n', sortedKurtIdxs(ni), sortedKurt(ni));
   %end
   %fclose(kurtFile);

   %%Create histogram of kurtosis elements
   %handle = figure;
   %hist(kurtVals, [0:.5:10]);
   %title('Hist of kurtosis values');
   %print(handle, [plotOutDir, 'kurtHist.png']);

   %%%PeakMean
   %%Write mean and std of peak mean in file
   %peakMeanFile = fopen([plotOutDir, 'peakmean.txt'], 'w');
   %fprintf(peakMeanFile, 'peakmean: %f +- %f\n', mean(peakMeanVals(:)), std(peakMeanVals(:)));
   %[sortedPeakMean , sortedPeakMeanIdxs] = sort(peakMeanVals, 'descend');

   %%Write ranking by peakMean
   %for(ni = 1:nf)
   %   fprintf(peakMeanFile, '%d: %f\n', sortedPeakMeanIdxs(ni), sortedPeakMean(ni));
   %end
   %fclose(peakMeanFile);

   %%Create histogram of peak mean elements
   %handle = figure;
   %%hist(peakMeanVals, [0:.5:10]);
   %hist(peakMeanVals, [0.4:.05:.85]);
   %title('Hist of peak mean values');
   %print(handle, [plotOutDir, 'peakMeanHist.png']);

end

