addpath('../FastICA_25/')

outputdir = 'output/'
savepath = [outputdir, 'icabinoc.mat'];
clobber = false;

leftlistpath = '/nh/compneuro/Data/KITTI/list/image_02.txt';
rightlistpath = '/nh/compneuro/Data/KITTI/list/image_03.txt';

leftfid = fopen(leftlistpath);
rightfid = fopen(rightlistpath);

patchSize = 66;
numExamples = 10; %Num examples per image
numImages = 5000;
numElements = 512;

if(clobber)
   patches = zeros(numExamples*numImages, patchSize*patchSize*2);
   exampleIdx = 1;
   for(j = 1:numImages)
      leftImgFile = fgetl(leftfid);
      rightImgFile = fgetl(rightfid);

      disp([num2str(j), ' out of ', num2str(numImages), ': ', leftImgFile]);
      %printf('%d out of %d: %s\n', j, numImages, leftImgFile);
      %fflush(stdout);
      leftimg = imread(leftImgFile);
      rightimg = imread(rightImgFile);

      leftimg = leftimg(1:2:end, 1:2:end);
      rightimg = rightimg(1:2:end, 1:2:end);

      %leftimg = imresize(leftimg, .5, 'bilinear');
      %rightimg = imresize(rightimg, .5, 'bilinear');

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

   fclose(leftfid)

   [icasig, A, W] = fastica(patches', 'approach', 'symm', 'g', 'tanh', 'lastEig', numElements);
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
