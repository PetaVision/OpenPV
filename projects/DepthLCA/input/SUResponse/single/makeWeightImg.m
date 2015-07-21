addpath("~/workspace/PetaVision/mlab/util");

checkpointDir = "/nh/compneuro/Data/Depth/LCA/Checkpoints/saved_single/"
outputDir = "weightImg/";

leftWeightFiles = {"V1ToLeftError_W"};
rightWeightFiles = {"V1ToRightError_W"};
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
      leftOutImg(marginY+1:marginY+lHdr.nyp, marginX+1:marginX+lHdr.nxp, :) = permute(lData{1}.values{arborIdx+1}(:, :, :, patchIdx+1), [2,1,3]);
      rightOutImg(marginY+1:marginY+lHdr.nyp, marginX+1:marginX+lHdr.nxp, :) = permute(rData{1}.values{arborIdx+1}(:, :, :, patchIdx+1), [2,1,3]);


      if(debugMode)
         %Scale image to be between 0 and 1
         viewLeftOutImg = 128+((leftOutImg / max(abs(leftOutImg(:))))*127);
         viewLeftOutImg = viewLeftOutImg/256;
         figure;
         imshow(viewLeftOutImg)
      end

      %Mirroring
      for colorIdx = 1:3
         %Top
         leftOutImg(:, 1:marginY, colorIdx) = fliplr(leftOutImg(:, marginY+1:2*marginY, colorIdx));
         %Bot
         leftOutImg(:, marginY+lHdr.nyp+1:end, colorIdx) = fliplr(leftOutImg(:, lHdr.nyp+1:lHdr.nyp+marginY, colorIdx));
         %Left
         leftOutImg(1:marginX,:,colorIdx) = flipud(leftOutImg(marginX+1:2*marginX, :, colorIdx));
         %Right
         leftOutImg(marginX+lHdr.nxp+1:end, :,colorIdx) = flipud(leftOutImg(lHdr.nxp+1:lHdr.nxp+marginX,:,colorIdx));

         %Top
         rightOutImg(:, 1:marginY, colorIdx) = fliplr(rightOutImg(:, marginY+1:2*marginY, colorIdx));
         %Bot
         rightOutImg(:, marginY+lHdr.nyp+1:end, colorIdx) = fliplr(rightOutImg(:, lHdr.nyp+1:lHdr.nyp+marginY, colorIdx));
         %Left
         rightOutImg(1:marginX,:, colorIdx) = flipud(rightOutImg(marginX+1:2*marginX, :, colorIdx));
         %Right
         rightOutImg(marginX+lHdr.nxp+1:end, :, colorIdx) = flipud(rightOutImg(lHdr.nxp+1:lHdr.nxp+marginX,:,colorIdx));
      end

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
      %repmat across features
      mask = repmat(mask, [1, 1, lHdr.nfp]);
      

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



