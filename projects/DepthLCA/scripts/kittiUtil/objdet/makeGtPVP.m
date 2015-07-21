labelDir = '/nh/compneuro/Data/KITTI/objdet/training/label_2';
outDir = '/nh/compneuro/Data/KITTI/objdet/training/';

numData = 7480; %Zero indexed

imageSizeX = 1242;
imageSizeY = 375;

function colorVec = getColorVec(bbType)
   if(strcmp(bbType, 'Car'))
      %Red
      colorVec = [1, 0, 0];
   elseif(strcmp(bbType, 'Van'))
      %Light pink
      colorVec = [1, .5, .5];
   elseif(strcmp(bbType, 'Truck'))
      %Purple
      colorVec = [1, 0, 1];
   elseif(strcmp(bbType, 'Pedestrian'))
      %Blue
      colorVec = [0, 0, 1];
   elseif(strcmp(bbType, 'Person_sitting'))
      %Light blue
      colorVec = [.5, .5, 1];
   elseif(strcmp(bbType, 'Cyclist'))
      %Green
      colorVec = [0, 1, 0];
   elseif(strcmp(bbType, 'Tram'))
      %Yellow
      colorVec = [1, 1, 0];
   elseif(strcmp(bbType, 'Misc'))
      %White
      colorVec = [1, 1, 1];
   elseif(strcmp(bbType, 'DontCare'))
      %Gray
      colorVec = [.5, .5, .5];
   else
      assert(0);
   end
end

function featureIdx = getFeatureIdx(bbType)
   if(strcmp(bbType, 'Car'))
      featureIdx = 1;
   elseif(strcmp(bbType, 'Van'))
      featureIdx = 2;
   elseif(strcmp(bbType, 'Truck'))
      featureIdx = 3;
   elseif(strcmp(bbType, 'Pedestrian'))
      featureIdx = 4;
   elseif(strcmp(bbType, 'Person_sitting'))
      featureIdx = 5;
   elseif(strcmp(bbType, 'Cyclist'))
      featureIdx = 6;
   elseif(strcmp(bbType, 'Tram'))
      featureIdx = 7;
   elseif(strcmp(bbType, 'Misc'))
      featureIdx = 8;
   elseif(strcmp(bbType, 'DontCare'))
      featureIdx = 0;
   else
      assert(0);
   end
end

function viewBB(bbObj, xSize, ySize)
   outImg = zeros(ySize, xSize, 3);
   for(objIdx = 1:length(bbObj))
      colorVec = getColorVec(bbObj(objIdx).type);
      %Expand colorVec to size of bb with 3 colors
      x1 = round(bbObj(objIdx).x1) + 1;
      x2 = round(bbObj(objIdx).x2) + 1;
      y1 = round(bbObj(objIdx).y1) + 1;
      y2 = round(bbObj(objIdx).y2) + 1;
      bbXSize = x2 - x1 + 1;
      bbYSize = y2 - y1 + 1;
      colorMat = permute(repmat(colorVec, [bbYSize, 1, bbXSize]), [1, 3, 2]);
      try
         outImg(y1 : y2, x1:x2, :) = colorMat;
      catch err
         keyboard
      end

   end
   imagesc(outImg);
   keyboard
end

function [outIdxs, dncIdxs] = buildIdxList(bbObj, xSize, ySize)
   outImg = zeros(8, xSize, ySize); %In petavision order, 8 classes
   dncImg = zeros(1, xSize, ySize); %DNC class seperate
   for(objIdx = 1:length(bbObj))
      featureIdx = getFeatureIdx(bbObj(objIdx).type);
      %+1 to convert to 1 index
      x1 = round(bbObj(objIdx).x1) + 1;
      x2 = round(bbObj(objIdx).x2) + 1;
      y1 = round(bbObj(objIdx).y1) + 1;
      y2 = round(bbObj(objIdx).y2) + 1;
      if(featureIdx == 0)
         dncImg(1, x1:x2, y1:y2) = 1;
      else
         outImg(featureIdx, x1:x2, y1:y2) = 1;
      end
   end
   %Get non-zero values for outImg and dncImg, convert to zero index
   outIdxs = find(outImg) - 1;
   dncIdxs = find(dncImg) - 1;
end



addpath('devkit/matlab/');


gtData = {};
dncData = {};

for imgIdx = 0:numData
   disp([num2str(imgIdx), ' out of ', num2str(numData)])
   bbObjs = readLabels(labelDir, imgIdx);
   [outIdxs, dncIdxs] = buildIdxList(bbObjs, imageSizeX, imageSizeY);
   gtData{imgIdx+1}.time = imgIdx;
   gtData{imgIdx+1}.values = outIdxs;
   dncData{imgIdx+1}.time = imgIdx;
   dncData{imgIdx+1}.values = dncIdxs;
end

%Write to file
writepvpsparseactivityfile([outDir, '/kittiGT.pvp'], gtData, imageSizeX, imageSizeY, 8);
writepvpsparseactivityfile([outDir, '/dncData.pvp'], dncData, imageSizeX, imageSizeY, 1);





