function outimage = heatMapMontage(imagePvpFile, resultPvpFile, pv_dir, imageFrameNumber, resultFrameNumber, confidenceTable, classNameFile, evalCategoryIndices, displayCategoryIndices, montagePath, displayCommand)
% outimage = heatMapMontage(imagePvpFile, resultPvpFile, pv_dir, imageFrameNumber, resultFrameNumber, confidenceTable, montagePath, displayCommand)
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
% confidenceTable: a matrix with nf+1 columns, where nf is the number of features in resultPvpFile.
%    The last column should be an increasing vector where all values of the given frame of resultPvpFile
%    lie between the first entry of the column and the last.
%    confidenceTable(k,f) is the confidence value when a neuron in feature f has the value confidenceTable(k,nf+1).
%    Generally speaking, each of the first nf columns of confidenceTable should therefore be increasing between 0 and 1
%    (or between 0 and 100) but the program doesn't check for this.
% classNameFile: a character string that indicates a file containing the names of the character classes.
%    The file contains one class name per line.  The number of features in resultPvpFile should agree with
%    the number of lines in the file.
% evalCategoryIndices: a vector of the feature numbers to consider.  If empty, use all indices.
% displayCategoryIndices: a vector of the feature numbers to display.  If empty, use all indices.
% montagePath: The path to write the output image to.  The output image has the same dimensions as the frame of imagePvpFile.
%    If resultPvpFile has different dimensions, it will be rescaled using upsamplefill.
% displayCommand: If nonempty, run this command on montagePath after it has been written.  Uses the system command,
%    so control does not return to octave until the command completes.
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

numCategories=numel(displayCategoryIndices);

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

if isempty(classNameFile)
   classes=cell(resultHdr.nf,1)
   for k=1:resultHdr.nf
      classes{k}=num2str(k);
   end%for
else
   classnameFID = fopen(classNameFile,'r');
   % TODO allow blank lines and comments
   for k=1:resultHdr.nf
       classes{k} = fgets(classnameFID);
       if ~isequal(class(classes{k}),'char')
          error("heatMapMontage:classnameeof","classNameFile \"%s\" has EOF before all %d features have been named.\n",classNameFile, resultHdr.nf);
       end%if
       while(~isempty(classes{k}) && classes{k}(end)==char(10)), classes{k}(end)=[]; end
       fprintf(1,'class %d is \"%s\"\n', k, classes{k});
   end%for
   fclose(classnameFID);
end%if

if numel(classes)!=resultHdr.nf
   error("heatMapMontage:wrongnf","number of classes is %d but %s has %d features.",numel(classes),resultPvpFile,resultHdr.nf);
end%if

if isempty(evalCategoryIndices)
   evalCategoryIndices=1:resultHdr.nf;
end%if

if isempty(displayCategoryIndices)
   displayCategoryIndices=1:resultHdr.nf;
end%if

upsampleNx = size(imageData,2)/size(resultData,2);
upsampleNy = size(imageData,1)/size(resultData,1);
assert(upsampleNx==round(upsampleNx));
assert(upsampleNy==round(upsampleNy));

thresholdConfidence = 0.5;
thresholdConfColor = [0.5 0.5 0.5];
maxConfColor = [0 1 0];
imageBlendCoeff = 0.3;
% heatmap image will be imageBlendCoeff * imagedata plus (1-imageBlendCoeff) * heatmap data, where
% the heatmap is converted to color using thresholdConfColor and maxConfColor
montageImage = zeros((size(imageData,1)+64+10)*numRows, (size(imageData,2)+10)*numColumns,3);
% The +10 creates a border around each tile in the montage

confData = zeros(size(resultData));
for k=1:resultHdr.nf
    if any(evalCategoryIndices==k)
       confData(:,:,k) = interp1(confidenceTable(:,end), confidenceTable(:,k), resultData(:,:,k));
    end%if
end%for
maxConfidence = max(confData(:));
thresholdConfData = max(confData-thresholdConfidence,0)/(1.0-thresholdConfidence);
winningIndex = find(confData==maxConfidence);
[~,~,winningFeature] = ind2sub(size(confData),winningIndex);
for k=1:numel(winningFeature)
   fprintf(1,'winning feature is %s, with confidence %.1f%%\n', classes{winningFeature(k)}, 100*maxConfidence);
end%for

imageDataBlend = imageBlendCoeff*imageData;
for k=1:numCategories
    category = displayCategoryIndices(k);
    categorycolumn = mod(k-1,numColumns)+1;
    categoryrow = (k-categorycolumn)/numColumns+1;
    assert(categoryrow==round(categoryrow));
    thisclass = classes{category};
    resultUpsampledY = upsamplefill(thresholdConfData(:,:,category),upsampleNx-1,'COPY');
    resultUpsampled = upsamplefill(resultUpsampledY',upsampleNy-1,'COPY')';
    %resultPngFilename = sprintf('tmp/result-frame%04d-category%02d.png',resultFrameNumber, category);
    tileImage = zeros([size(imageData),3]);
    for b=1:3
       tileImage(:,:,b) = thresholdConfColor(b) + (maxConfColor(b)-thresholdConfColor(b))*resultUpsampled;
       tileImage(:,:,b) = imageBlendCoeff * imageData + (1-imageBlendCoeff) * tileImage(:,:,b);
    end%for
    if (size(tileImage,3)==1), tileImage=repmat(tileImage,[1 1 3]); end;

    maxConfCategory = max(max(confData(:,:,category)));
    if any(winningFeature==category)
    %if maxConfCategory==maxConfidence
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
    makeValueCommand = sprintf('convert -background white -fill %s -size %dx32 -pointsize 24 -gravity center label:%.1f%% %s', captionColor, size(imageData,2), 100*maxConfCategory, valueFile);
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
   if isequal(class(img),'uint16')
      img = double(img)/65535;
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
