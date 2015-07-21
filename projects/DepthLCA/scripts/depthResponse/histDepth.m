function histDepth(depthFile, numDepthBins)
   addpath('~/workspace/PetaVision/mlab/util');
   [depth_data, depth_hdr] = readpvpfile(depthFile);

   disp(['Reshaping pvp files']);
   %Reshape cell structs to be one big array

   numTimes = length(depth_data);
   nx = depth_hdr.nxGlobal;
   ny = depth_hdr.nyGlobal;

   depth_act = zeros(numTimes, nx, ny);
   for(t = 1:numTimes)
      depth_act(t, :, :) = depth_data{t}.values;
   endfor

   histVals = depth_act(:);
   %remove dnc areas
   histVals(find(histVals == 0)) = [];

   %Plot histogram
   figure;
   hist(histVals(:), numDepthBins);
   keyboard
end
