checkptDir = '/nh/compneuro/Data/Depth/LCA/dictLearn/white_LCA_dictLearn/'

leftDict = [checkptDir, 'V1ToLeftError_W.pvp'];
rightDict = [checkptDir, 'V1ToRightError_W.pvp'];

[leftData, leftHdr] = readpvpfile(leftDict);
[rightData, rightHdr] = readpvpfile(rightDict);

leftVals = leftData{1}.values{1};
rightVals = rightData{1}.values{1};

[nxp, nyp, nfp, nkernel] = size(leftVals);

binocScores = zeros(nkernel, 1);

%Normalize left/rightVals

minVal = min([leftVals(:), rightVals(:)](:));
maxVal = max([leftVals(:), rightVals(:)](:));

maxScale = max(abs(minVal), abs(maxVal));

leftVals = (leftVals/maxScale + 1)/2; 
rightVals = (rightVals/maxScale + 1)/2; 

for k = 1:nkernel
   %difference betwen sum of sq
   leftElement = leftVals(:, :, :, k);
   rightElement = rightVals(:, :, :, k);

   binocScores(k) = sum((rightElement .^ 2)(:)) - sum((leftElement .^ 2)(:));
end

figure;
hist(binocScores, 21);


[drop, minIdx] = min(binocScores);
[drop, maxIdx] = max(binocScores);

figure;
subplot(1, 2, 1);
imshow(leftVals(:, :, 1, minIdx));
subplot(1, 2, 2);
imshow(rightVals(:, :, 1, minIdx));
title('Left');
colormap(gray);

figure;
subplot(1, 2, 1);
imshow(leftVals(:, :, 1, maxIdx));
subplot(1, 2, 2);
imshow(rightVals(:, :, 1, maxIdx));
title('Right');
colormap(gray);

keyboard
