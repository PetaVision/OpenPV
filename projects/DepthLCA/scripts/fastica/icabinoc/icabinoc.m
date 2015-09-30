addpath('../FastICA_25/')
addpath('~/workspace/OpenPV/pv-core/mlab/util');

outputdir = 'output/'
savepath = [outputdir, 'icabinoc.mat'];
clobber = false;

leftlistpath = '/nh/compneuro/Data/KITTI/list/image_02.txt';
rightlistpath = '/nh/compneuro/Data/KITTI/list/image_03.txt';

patchSize = 66;
numExamples = 10; %Num examples per image
numImages = 5000;
maxLineRead = 100000;
numElements = 512;

rectified = false;

if(clobber)
   %Read all lines
   l = 1;
   leftImgs = {};
   rightImgs = {};

   leftfid = fopen(leftlistpath);
   rightfid = fopen(rightlistpath);

   disp('Reading file');
   while ~feof(leftfid) && ~feof(rightfid) && l <= maxLineRead 
      leftImgs{l} = fgetl(leftfid);
      rightImgs{l} = fgetl(rightfid);
      l = l + 1;
   end
   fclose(leftfid);
   fclose(rightfid);

   %Random frames within the numImages 
   p = randperm(l-1);
   randIdxs = p(1:numImages);

   patches = zeros(numExamples*numImages, patchSize*patchSize*2);
   exampleIdx = 1;
   for(i = 1:numImages)
      j = randIdxs(i);
      leftImgFile = leftImgs{j};
      rightImgFile = rightImgs{j};

      disp([num2str(i), ' out of ', num2str(numImages), ': ', leftImgFile]);

      leftimg = imread(leftImgFile);
      rightimg = imread(rightImgFile);

      assert(max(leftimg(:)) <= 255);
      assert(max(rightimg(:)) <= 255);

      leftimg = double(leftimg)/255;
      rightimg = double(rightimg)/255;

      %Make b/w
      if(ndims(leftimg) == 3)
         leftimg = 0.21 * leftimg(:, :, 1) + 0.72 * leftimg(:, :, 2) + 0.07 * leftimg(:, :, 3);
      end
      if(ndims(rightimg) == 3)
         rightimg = 0.21 * rightimg(:, :, 1) + 0.72 * rightimg(:, :, 2) + 0.07 * rightimg(:, :, 3);
      end

      %Downsample
      leftimg = leftimg(1:2:end, 1:2:end);
      rightimg = rightimg(1:2:end, 1:2:end);

      [ny, nx] = size(leftimg);
      [ny2, nx2] = size(rightimg);
      assert(ny == ny2 && nx == nx2);

      %Random spot for top left corner
      xstart = round((nx-patchSize)*rand(numExamples)) + 1;
      ystart = round((ny-patchSize)*rand(numExamples)) + 1;

      for(i = 1:numExamples)
         leftPatch = leftimg(ystart(i):ystart(i)+patchSize-1, xstart(i):xstart(i)+patchSize-1);
         rightPatch = rightimg(ystart(i):ystart(i)+patchSize-1, xstart(i):xstart(i)+patchSize-1);

         patches(exampleIdx, 1:patchSize*patchSize) = leftPatch(:);
         patches(exampleIdx, patchSize*patchSize+1:end) = rightPatch(:);
         exampleIdx = exampleIdx + 1;
      end
   end

   if(rectified)
      assert(mod(numElements, 2) == 0);
      inNumElements = numElements/2;
   else
      inNumElements = numElements;
   end

   [icasig, A, W] = fastica(patches', 'approach', 'symm', 'g', 'tanh', 'lastEig', inNumElements);
   save('icabinoc.mat', 'A');
else
   load('icabinoc.mat');
end

A = A';
[numIC, ~] = size(A);
ALeft = A(:, 1:patchSize*patchSize);
ARight = A(:, patchSize*patchSize+1:end);


leftIcaPatches = reshape(ALeft, [numIC, patchSize, patchSize]);
rightIcaPatches = reshape(ARight, [numIC, patchSize, patchSize]);

%If rectified, double and invert
if(rectified)
   assert(numIC == numElements/2);
   tmpLeft = zeros(numElements, patchSize, patchSize);
   tmpRight = zeros(numElements, patchSize, patchSize);
   tmpLeft(1:numIC, :, :) = leftIcaPatches;
   tmpRight(1:numIC, :, :) = rightIcaPatches;
   tmpLeft(numIC+1:end, :, :) = -1 * leftIcaPatches;
   tmpRight(numIC+1:end, :, :) = -1 * rightIcaPatches;
   leftIcaPatches = tmpLeft;
   rightIcaPatches = tmpRight;
   numIC = numElements;
end

%Make tablou to see ICA patches
numCols = floor(sqrt(numIC));
numRows = ceil(numIC/numCols);

leftOutImg = zeros(numRows * patchSize, numCols * patchSize);
rightOutImg = zeros(numRows * patchSize, numCols * patchSize);
for(i = 1:numIC)
   colIdx = mod((i-1), numCols) * patchSize + 1;  %zero idxd
   rowIdx = floor((i-1)/numCols) * patchSize + 1;
   leftPatch = leftIcaPatches(i, :, :);
   leftNormPatch = (leftPatch - min(leftPatch(:)))/(max(leftPatch(:))-min(leftPatch(:)));
   leftOutImg(rowIdx:rowIdx+patchSize-1, colIdx:colIdx+patchSize-1) = leftNormPatch;
   rightPatch = rightIcaPatches(i, :, :);
   rightNormPatch = (rightPatch - min(rightPatch(:)))/(max(rightPatch(:))-min(rightPatch(:)));
   rightOutImg(rowIdx:rowIdx+patchSize-1, colIdx:colIdx+patchSize-1) = rightNormPatch;
end

imwrite(leftOutImg, [outputdir, 'leftICA.png']);
imwrite(rightOutImg, [outputdir, 'rightICA.png']);

%Write to pvp file
ldata = zeros(patchSize, patchSize, 1, numIC);
rdata = zeros(patchSize, patchSize, 1, numIC);
ldata(:, :, 1, :) = permute(leftIcaPatches, [3, 2, 1]);
rdata(:, :, 1, :) = permute(rightIcaPatches, [3, 2, 1]);

leftData{1}.time = 0;
rightData{1}.time = 0;
leftData{1}.values{1} = ldata;
rightData{1}.values{1} = rdata;

writepvpsharedweightfile([outputdir, 'leftDict.pvp'], leftData);
writepvpsharedweightfile([outputdir, 'rightDict.pvp'], rightData);


%figure
%colormap(gray);
%imagesc(leftOutImg);
%figure
%imagesc(rightOutImg);
