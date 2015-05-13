function outimage = heatMapMontage(imagePvpFile, resultPvpFile, pv_dir, imageFrameNumber, resultFrameNumber, montagePath, displayCommand)
% outimage = heatMapMontage(imagePvpFile, resultPvpFile, pv_dir, imageFrameNumber, resultFrameNumber, montagePath)
% Takes frames from two input pvp files, imagePvpFile and resultPvpFile and creates a montage compositing
% the image pvp file with each of the features of the result pvp file.
%
% imagePvpFile: the path to a pvp file containing the base image.
% resultPvpFile: the path to a pvp file containing the results.
% pv_dir: the path containing the function m-file readpvpfile.m (usually in <PV_DIR>/mlab/util).
%    If empty, readpvpfile must be a recognized command after initializing octave.
%    If nonempty, readpvpfile must be a recognized command after calling addpath(pv_dir);
% imageFrameNumber: the index of the specific frame from imagePvpFile to use.  The beginning frame has index 1.
% resultFrameNumber: the index of the specific frame from resultPvpFile to use.
% montagePath: The path to write the output image to.  The output image has the same dimensions as the frame of imagePvpFile.
%    If resultPvpFile has different dimensions, it will be rescaled using upsamplefill.
%
% outimage: an ny-by-nx-by-3 array giving the output image.  ny and nx are the same as the frame of imagePvpFile.
%

if exist('pv_dir', 'var') && ~isempty(pv_dir)
   addpath(pv_dir);
end%if
if isempty(which('readpvpfile'))
   error("heatMapMontage:readpvpfilemissing","heatMapMontage error: missing command readpvpfile");
end%if

if ~exist('tmp','dir')
    mkdir tmp
    ownstmpdir = 1;
else
   ownstmpdir = 0;
end%if

fprintf(1,'heatMapMontage: input image file \"%s\", frame %d\n', imagePvpFile, imageFrameNumber);

[imagePvp,imgHdr] = readpvpfile(imagePvpFile, [], imageFrameNumber, imageFrameNumber);
if (imgHdr.filetype != 4)
   error("heatMapMontage:expectingnonsparse","heatMapMontage expects %s to be a nonsparse layer",imagePvpFile);
end%if
%imageData = permute(imagePvp{1}.values,[2 1 3]); % keep image as color
imageData = mean(imagePvp{1}.values,3)'; % convert image to gray

[resultPvp,resultHdr] = readpvpfile(resultPvpFile, [], resultFrameNumber, resultFrameNumber);
if (resultHdr.filetype != 4)
   error("heatMapMontage:expectingnonsparse","heatMapMontage expects %s to be a nonsparse layer",resultPvpFile);
end%if
resultData = permute(resultPvp{1}.values,[2 1 3]);
resultData = max(resultData,0);

classes={'background'; 'aeroplane'; 'bicycle'; 'bird'; 'boat'; 'bottle'; 'bus'; 'car'; 'cat'; 'chair'; 'cow'; 'diningtable'; 'dog'; 'horse'; 'motorbike'; 'person'; 'pottedplant'; 'sheep'; 'sofa'; 'train'; 'tvmonitor'};
categoryindices=2:21; % Which categories to display.  background=1, aeroplane=2, etc.
numCategories=numel(categoryindices);

numColumns = 1:numCategories;
numRows = ceil(numCategories./numColumns);
totalSizeY = (size(imageData,1)+64+10)*numRows;
totalSizeX = (size(imageData,2)+64+10)*numColumns;
aspectRatio = totalSizeX./totalSizeY;
ldfgr = abs(log(aspectRatio) - log((1+sqrt(5))/2));
numColumns = find(ldfgr==min(ldfgr),1);
numRows = numRows(numColumns);
while (numColumns-1)*numRows >= numCategories
    numColumns = numColumns-1;
end%while
while (numRows-1)*numColumns >= numCategories
    numRows = numRows-1;
end%while
assert(numRows*numColumns >= numCategories);

if(numel(classes)!=resultHdr.nf)
   error("heatMapMontage:wrongnf","number of classes is %d but %s has %d features.",numel(classes),resultPvpFile,resultHdr.nf);
end%if

montagerows = floor(sqrt(numCategories));
montagecols = ceil(numCategories/montagerows);
assert(montagerows*montagecols>=numCategories);
thresholdLevel = 1.00; % the confidence that will be mapped to minimum brightness
saturationLevel = 1.50; % max(resultData(:)); % the confidence that will be mapped to maximum brightness
upsampleNx = size(imageData,2)/size(resultData,2);
upsampleNy = size(imageData,1)/size(resultData,1);
assert(upsampleNx==round(upsampleNx));
assert(upsampleNy==round(upsampleNy));

%imagePngFilename = sprintf('tmp/image%04d.png', imageFrameNumber);
%imwrite(imageData,imagePngFilename);

zeroconfcolor = [0.5 0.5 0.5];
maxconfcolor = [0 1 0];
imageblendcoeff = 0.3;
% heatmap image will be imageblendcoeff * imagedata plus (1-imageblendcoeff) * heatmap data, where
% the heatmap is converted to color using zeroconfcolor and maxconfcolor
montageImage = zeros((size(imageData,1)+64+10)*numRows, (size(imageData,2)+10)*numColumns,3);
% The +10 creates a border around each tile in the montage

imageDataBlend = imageblendcoeff*imageData;
for k=1:numCategories
    category = categoryindices(k);
    categorycolumn = mod(k-1,numColumns)+1;
    categoryrow = (k-categorycolumn)/numColumns+1;
    assert(categoryrow==round(categoryrow));
    thisclass = classes{category};
    maxConfidence = max(max(resultData(:,:,category)));
    isWinner = maxConfidence==max(resultData(:));
    resultDataTrunc = max(resultData(:,:,category),thresholdLevel)-thresholdLevel;
    if saturationLevel-thresholdLevel != 0
        resultDataTrunc = min(resultDataTrunc/(saturationLevel-thresholdLevel),1);
    end%if
    assert(all(resultDataTrunc(:)>=0) && all(resultDataTrunc(:)<=1));
    resultUpsampledY = upsamplefill(resultDataTrunc,upsampleNx-1,'COPY');
    resultUpsampled = upsamplefill(resultUpsampledY',upsampleNy-1,'COPY')';
    %resultPngFilename = sprintf('tmp/result-frame%04d-category%02d.png',resultFrameNumber, category);
    tileImage = zeros([size(imageData),3]);
    for b=1:3
       tileImage(:,:,b) = zeroconfcolor(b) + (maxconfcolor(b)-zeroconfcolor(b))*resultUpsampled;
       tileImage(:,:,b) = imageblendcoeff * imageData + (1-imageblendcoeff) * tileImage(:,:,b);
    end%for
    if (size(tileImage,3)==1), tileImage=repmat(tileImage,[1 1 3]); end;

    if isWinner
        captionColor = 'blue';
    else
        captionColor = 'gray';
    end%if

    file = sprintf('tmp/label%s.png', classes{category});
    makeLabelCommand = sprintf('convert -background white -fill %s -size %dx32 -pointsize 24 -gravity center label:%s %s', captionColor, size(imageData,2), classes{category}, file);
    system(makeLabelCommand);
    img = readImageMagickFile(file);
    delete(file);

    valueFile = sprintf('tmp/value%s.png', classes{category});
    makeValueCommand = sprintf('convert -background white -fill %s -size %dx32 -pointsize 24 -gravity center label:%f %s', captionColor, size(imageData,2), maxConfidence, valueFile);
    system(makeValueCommand);
    valueImage = readImageMagickFile(valueFile);
    delete(valueFile);

    xstart = (size(imageData,2)+10)*(categorycolumn-1)+5;
    ystart = (size(imageData,1)+64+10)*(categoryrow-1)+5;
    % The +10 provides a 10-pixel border around each image.
    % The +5 places the tile in the middle of the region with 10-pixel border.
    % The +64 is because each tile includes the caption, which is 64 pixels high.
    montageImage(ystart+(1:32),xstart+(1:size(imageData,2)),:) = img;
    montageImage(ystart+(33:64),xstart+(1:size(imageData,2)),:) = valueImage;
    montageImage(ystart+64+(1:size(imageData,1)),xstart+(1:size(imageData,2)),:) = tileImage;
end%for
imwrite(montageImage, montagePath);

if nargout>0
   outimage = imread(montagePath);
end%if

fprintf(1,'heatMapMontage: output heat map image \"%s\"\n', montagePath);

if exist('displayCommand','var') && ~isempty(displayCommand)
   displayCommandWithArg = [displayCommand ' ' montagePath];
   fprintf(1, "displayCommand \"%s\"", displayCommandWithArg);
   displayStatus = system(displayCommandWithArg);
   fprintf(1,'displayCommand returned %d\n', displayStatus);
end%if

%for k=1:numCategories
%    category = categoryindices(k);
%    resultPngFilename = sprintf('tmp/result-frame%04d-category%02d.png',resultFrameNumber, category);
%    delete(resultPngFilename);
%    compositedFilename = sprintf('tmp/composited-frame%04d-category%02d.png',imageFrameNumber, category);
%    delete(compositedFilename);
%end%for
%delete(imagePngFilename);
%
if ownstmpdir
   rmdir tmp;
end%if

end%function

function img = readImageMagickFile(file)
[img,cmap] = imread(file);
if isempty(cmap)
   if isequal(class(img),'uint8')
      img = double(img)/255;
   end%if
   if size(img,3)==1
      img=repmat(img,[1 1 3]);
   end%if
else
   assert(numel(size(img))==2);
   v = img;
   img = zeros([size(v) 3]);
   for n=1:size(v,2)
      for m=1:size(v,1)
         img(m,n,:) = cmap(v(m,n)+1,:);
      end%for
   end%for
end%if

end%function
