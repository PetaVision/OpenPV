addpath("~/workspace/PetaVision/mlab/util");

checkpointDir = "/nh/compneuro/Data/Depth/LCA/Checkpoints/saved_stack_slp/"
outputDir = "weightImg/";

leftWeightFiles = {"V1S2ToLeftError_W";"V1S4ToLeftError_W";"V1S8ToLeftError_W"};
rightWeightFiles = {"V1S2ToRightError_W";"V1S4ToRightError_W";"V1S8ToRightError_W"};
arborIdx = 0;
debugMode = 0;

for(iWeight = 1:length(leftWeightFiles))
   leftWeightFile = leftWeightFiles{iWeight}
   rightWeightFile = rightWeightFiles{iWeight}

   [lData, lHdr] = readpvpfile([checkpointDir, leftWeightFile, ".pvp"]);
   [rData, rHdr] = readpvpfile([checkpointDir, rightWeightFile, ".pvp"]);

   assert(lHdr.nxp == rHdr.nxp);
   assert(lHdr.nyp == rHdr.nyp);
   assert(lHdr.nfp == rHdr.nfp);
   assert(lHdr.numPatches == rHdr.numPatches);

   for (patchIdx = 0:lHdr.numPatches-1)
      close all

      patchIdx
      leftOutImg = zeros(lHdr.nyp*2, lHdr.nxp*2, rHdr.nfp);
      rightOutImg = zeros(lHdr.nyp*2, lHdr.nxp*2, rHdr.nfp);
      marginX = lHdr.nxp/2;
      marginY = lHdr.nyp/2;
      
      %Put img in center
      leftOutImg(marginY+1:marginY+lHdr.nyp, marginX+1:marginX+lHdr.nxp) = lData{1}.values{arborIdx+1}(:, :, :, patchIdx+1)';
      rightOutImg(marginY+1:marginY+lHdr.nyp, marginX+1:marginX+lHdr.nxp) = rData{1}.values{arborIdx+1}(:, :, :, patchIdx+1)';


      if(debugMode)
         %Scale image to be between 0 and 1
         viewLeftOutImg = 128+((leftOutImg / max(abs(leftOutImg(:))))*127);
         viewLeftOutImg = viewLeftOutImg/256;
         figure;
         imshow(viewLeftOutImg)
      end

      %Mirroring
      %Top
      leftOutImg(:, 1:marginY) = fliplr(leftOutImg(:, marginY+1:2*marginY));
      %Bot
      leftOutImg(:, marginY+lHdr.nyp+1:end) = fliplr(leftOutImg(:, lHdr.nyp+1:lHdr.nyp+marginY));
      %Left
      leftOutImg(1:marginX,:) = flipud(leftOutImg(marginX+1:2*marginX, :));
      %Right
      leftOutImg(marginX+lHdr.nxp+1:end, :) = flipud(leftOutImg(lHdr.nxp+1:lHdr.nxp+marginX,:));

      %Top
      rightOutImg(:, 1:marginY) = fliplr(rightOutImg(:, marginY+1:2*marginY));
      %Bot
      rightOutImg(:, marginY+lHdr.nyp+1:end) = fliplr(rightOutImg(:, lHdr.nyp+1:lHdr.nyp+marginY));
      %Left
      rightOutImg(1:marginX,:) = flipud(rightOutImg(marginX+1:2*marginX, :));
      %Right
      rightOutImg(marginX+lHdr.nxp+1:end, :) = flipud(rightOutImg(lHdr.nxp+1:lHdr.nxp+marginX,:));

      if(debugMode)
         %Scale image to be between 0 and 1
         viewLeftOutImg = 128+((leftOutImg / max(abs(leftOutImg(:))))*127);
         viewLeftOutImg = viewLeftOutImg/256;
         figure
         imshow(viewLeftOutImg)
      end


      %Gaussian filtering
      mask = fspecial('gaussian', [lHdr.nyp*2, lHdr.nxp*2], (marginX/sqrt(2)));
      %Normalize mask
      mask = mask/max(abs(mask(:)));
      %Filter patch
      leftOutImg = leftOutImg .* mask;
      rightOutImg = rightOutImg .* mask;

      %Scale image to be between 0 and 1
      leftOutImg = 128+((leftOutImg / max(abs(leftOutImg(:))))*127);
      rightOutImg = 128+((rightOutImg / max(abs(rightOutImg(:))))*127);

      leftOutImg = leftOutImg/256;
      rightOutImg = rightOutImg/256;

      if(debugMode)
         figure
         imshow(leftOutImg)
         keyboard
      else
         imwrite(leftOutImg, [outputDir, leftWeightFile, "_patch_", num2str(patchIdx), ".png"]);
         imwrite(rightOutImg, [outputDir, rightWeightFile, "_patch_", num2str(patchIdx), ".png"]);
      end
   end
end



