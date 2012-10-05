%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Image reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [outMat] = reconstruct(activityIndex, onWeightValues, offWeightValues, arborId)
   [procsX procsY numArbors] = size(onWeightValues);
   [patchSizeX patchSizeY numFeatures temp] = size(onWeightValues{procsX, procsY, numArbors});
   global sizeX sizeY;
   global columnSizeX columnSizeY;
   global marginIndex;
   outMat = zeros(columnSizeY, columnSizeX);
   for activityi = 1:length(activityIndex)
      %Calculate what proc activity is in
      %Since this is being calculated as row first, use X Y instead of Y X
      [aIx aIy] = ind2sub([columnSizeX columnSizeY], activityIndex(activityi) + 1); 

      %Calculate what process this is in
      procXi = floor((aIx - 1)/sizeX) + 1;
      procYi = floor((aIy - 1)/sizeY) + 1;

      %If the spiking activity is not in the allowed area
      if isempty(find(marginIndex == (activityIndex(activityi) + 1)))
         continue;   %Skip
      end

      %Get bounds for out
      yStart = aIy - floor(patchSizeX/2);
      yEnd = aIy + floor(patchSizeY/2);
      xStart = aIx - floor(patchSizeX/2);
      xEnd = aIx + floor(patchSizeY/2);
      %get index based on what quaderant
      newIndX = mod((aIx - 1), sizeX) + 1;
      newIndY = mod((aIy - 1), sizeY) + 1;
      newInd = sub2ind([sizeX sizeY], newIndX, newIndY);

      for nfi = 1:numFeatures %Number of features
         %Set out
         outMat(yStart:yEnd, xStart:xEnd) += ... 
         onWeightValues{procXi, procYi, arborId}(:, :, nfi, newInd)' - ...
         offWeightValues{procXi, procYi, arborId}(:, :, nfi, newInd)';
      end
   end
end
