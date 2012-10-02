%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Weight Map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [outMat] = weightMap(onWeightValues, offWeightValues, arborId)
   [procsX procsY numArbors] = size(onWeightValues);
   [patchSizeX patchSizeY numFeatures temp] = size(onWeightValues{procsX, procsY, numArbors});
   global GRID_FLAG;
   global columnSizeX columnSizeY;
   global sizeX sizeY;
   global marginIndex;
   if(GRID_FLAG)
      outMat = zeros(columnSizeY * patchSizeY + columnSizeY, columnSizeX * patchSizeX + columnSizeX);
   else
      outMat = zeros(columnSizeY * patchSizeY, columnSizeX * patchSizeX);
   end
   for margini = 1:length(marginIndex)   %Iterate through allowed margin index
      %Calculate what proc activity is in
      %Since this is being calculated as row first, use X Y instead of Y X
      [mIx mIy] = ind2sub([columnSizeX columnSizeY], marginIndex(margini)); 
      procXi = floor((mIx - 1)/sizeX) + 1;
      procYi = floor((mIy - 1)/sizeY) + 1;

      %Convert index to weight out mat
      if(GRID_FLAG)
         %Added 1 pixel grid between patches
         startIndexX = (mIx - 1) * (patchSizeX + 1) + 1; 
         startIndexY = (mIy - 1) * (patchSizeY + 1) + 1; 
      else
         startIndexX = (mIx - 1) * patchSizeX + 1; 
         startIndexY = (mIy - 1) * patchSizeY + 1; 
      end
      endIndexX = startIndexX + patchSizeX - 1;
      endIndexY = startIndexY + patchSizeY - 1;

      %get index based on what quaderant
      newIndX = mod((mIx - 1), sizeX) + 1;
      newIndY = mod((mIy - 1), sizeY) + 1;
      newInd = sub2ind([sizeX sizeY], newIndX, newIndY);

      for nfi = 1:numFeatures  %For multiple features
         %Set out
         outMat(startIndexY:endIndexY, startIndexX:endIndexX) +=...
         onWeightValues{procXi, procYi, arborId}(:, :, nfi, newInd)' - ...
         offWeightValues{procXi, procYi, arborId}(:, :, nfi, newInd)';
      end
   end
   %Draw grid
   %TODO find better way to do this
   if(GRID_FLAG)
      marginVal = min(outMat(:));
      outMat(:, patchSizeX:patchSizeX:end) = marginVal;
      outMat(patchSizeY:patchSizeY:end, :) = marginVal;
   end
end
