%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Weight Patch
%% Note: It is assumed that the values in cells are within the margins
%% This is checked in the calling script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [outMat] = cellMap(onWeightValues, offWeightValues, arborId, patchLoc)
   [procsX procsY numArbors] = size(onWeightValues);
   [patchLocSizeX patchLocSizeY numFeatures temp] = size(onWeightValues{procsX, procsY, numArbors});
   global sizeX sizeY;
   outMat = zeros(patchLocSizeY, patchLocSizeX);
   patchLocX = patchLoc(1);
   patchLocY = patchLoc(2);
   %Calculate what proc activity is in
   procXi = floor((patchLocX - 1)/sizeX) + 1;
   procYi = floor((patchLocY - 1)/sizeY) + 1;
   %Calculate index of that proc
   newIndX = mod((patchLocX - 1), sizeX) + 1;
   newIndY = mod((patchLocY - 1), sizeY) + 1;
   %Since this is being calculated as row first, use X Y instead of Y X
   newInd = sub2ind([sizeX sizeY], newIndX, newIndY);

   for nfi = 1:numFeatures  %For multiple features
      outMat +=...
      onWeightValues{procXi, procYi, arborId}(:, :, nfi, newInd)' - ...
      offWeightValues{procXi, procYi, arborId}(:, :, nfi, newInd)';
   end
end
