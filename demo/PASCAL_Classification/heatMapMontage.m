function outimage = heatMapMontage(...
    imagePvpFile,...
    resultPvpFile,...
    reconPvpFile,...
    pv_dir,...
    imageFrameNumber,...
    resultFrameNumber,...
    reconFrameNumber,...
    confidenceTable,...
    classNameFile,...
    resultsTextFile,...
    evalCategoryIndices,...
    displayCategoryIndices,...
    highlightThreshold,...
    heatMapThreshold,...
    heatMapMaximum,...
    drawBoundingBoxes,...
    boundingBoxThickness,...
    dbscanEps,...
    dbscanDensity,...
    montagePath,...
    displayCommand)
% outimage = heatMapMontage(imagePvpFile, resultPvpFile, reconPvpFile, pv_dir, imageFrameNumber, resultFrameNumber,
% reconFrameNumber, confidenceTable, classNameFile, resultsTextFile, evalCategoryIndices, displayCategoryIndices, highlightThreshold, heatMapThreshold, heatMapMaximum, drawBoundingBoxes, boundingBoxThickness, dbscanEps, dbscanDensity, montagePath, displayCommand)
% Takes frames from two input pvp files, imagePvpFile and resultPvpFile and creates a montage compositing
% the image pvp file with each of the features of the result pvp file.
%
% imagePvpFile: the path to a pvp file containing the base image.
% resultPvpFile: the path to a pvp file containing the results.
% reconPvpFile: the path to a pvp file containing the image reconstruction.
% pv_dir: the path containing the function m-file readpvpfile.m (usually in <PV_DIR>/mlab/util).
%    If empty, readpvpfile must be a recognized command after initializing octave.
%    If nonempty, readpvpfile must be a recognized command after calling addpath(pv_dir);
% imageFrameNumber: the index of the specific frame from imagePvpFile to use.  The beginning frame has index 1.
% resultFrameNumber: the index of the specific frame from resultPvpFile to use.
% reconFrameNumber: the index of the specific frame from reconPvpFile to use.
% confidenceTable: a matrix with nf+1 columns, where nf is the number of features in resultPvpFile.
%    The last column should be an increasing vector where all values of the given frame of resultPvpFile
%    lie between the first entry of the column and the last.
%    confidenceTable(k,f) is the confidence value when a neuron in feature f has the value confidenceTable(k,nf+1).
%    Generally speaking, each of the first nf columns of confidenceTable should therefore be increasing between 0 and 1
%    (or between 0 and 100) but the program doesn't check for this.
% classNameFile: a character string that indicates a file containing the names of the character classes.
%    The file contains one class name per line.  The number of features in resultPvpFile should agree with
%    the number of lines in the file.
% resultsTextFile: a character string that indicates a text file to write results to.
%    If the file exists, it will be clobbered, unless starting from a checkpoint, in which case it is appended to.
%    The text file contains all confidences above zero (strictly speaking, >= Octave's eps variable)
%    over all categories.
%    heatMapMontage calls the function defined in resultTextFile.m to produce the file.
% evalCategoryIndices: a vector of the feature numbers to consider.  If empty, use all indices.
% displayCategoryIndices: a vector of the feature numbers to display.  If empty, use all indices.
% highlightThreshold: the minimum confidence at which the highest confidence label and value are highlighted in the montage.
%    If empty, use 0.
% heatMapThreshold: The maximum value of confidence for which the heat map is zero.
%    If empty, use the same value as highlightThreshold.
% heatMapMaximum: The confidence value corresponding to maximum value of the threshold.  If empty, use 1.0
%    (on the assumption that confidences are between 0 and 1)
% drawBoundingBoxes: Determines whether to calculate clusters based on heatmap confidences, and draw bounding boxes.
%    If empty, defaults to 0 (Will not calculate or draw).
% boundingBoxThickness: the Thickness of the box (in pixels).
% dbscanEps: dbscan Eps parameter (neighborhood radius). If empty, dbscan will attempt to calculate this.
% dbscanDensity: dbscan density parameter (minimal number of objects considered as a cluster). If empty, uses 1.
% montagePath: The path to write the output image to.
% displayCommand: If nonempty, run this command on montagePath after it has been written.  Uses the system command,
%    so control does not return to octave until the command completes.
%
% outimage: a 3-dimensional array giving the output heat map montage as a color image.
%
load(confidenceTable);
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
imageData = permute(imagePvp{1}.values,[2 1 3]);
imageDataGrayscale = mean(imageData,3); % convert image to gray

[resultPvp,resultHdr] = readpvpfile(resultPvpFile, [], resultFrameNumber, resultFrameNumber);
if (resultHdr.filetype != 4)
   error("heatMapMontage:expectingnonsparse","heatMapMontage expects %s to be a nonsparse layer",resultPvpFile);
end%if
resultData = permute(resultPvp{1}.values,[2 1 3]);

if isempty(displayCategoryIndices)
   displayCategoryIndices=1:resultHdr.nf
end%if

numCategories=numel(displayCategoryIndices);

numColumns = 1:numCategories;
numRows = ceil(numCategories./numColumns);
totalSizeY = (size(imageDataGrayscale,1)+64+10)*numRows;
totalSizeX = (size(imageDataGrayscale,2)+10)*(numColumns+2);
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
numRows = max(numRows, 2);
assert(numRows*numColumns >= numCategories);

if isempty(classNameFile)
   classes=cell(resultHdr.nf,1)
   for k=1:resultHdr.nf
      classes{k}=num2str(k);
   end%for
else
   classnameFID = fopen(classNameFile,'r');
   if (classnameFID<0)
      error('heatMapMontage:badclassnamefile', 'Unable to open classNameFile "%s"', classNameFile);
   end%if
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

upsampleNx = size(imageDataGrayscale,2)/size(resultData,2);
upsampleNy = size(imageDataGrayscale,1)/size(resultData,1);
assert(upsampleNx==round(upsampleNx));
assert(upsampleNy==round(upsampleNy));

thresholdConfColor = [0.5 0.5 0.5];
maxConfColor = [0 1 0];
imageBlendCoeff = 0.3;
% heatmap image will be imageBlendCoeff * imagedata plus (1-imageBlendCoeff) * heatmap data, where
% the heatmap is converted to color using thresholdConfColor and maxConfColor
montageImage = zeros((size(imageDataGrayscale,1)+64+10)*numRows, (size(imageDataGrayscale,2)+10)*(numColumns+2), 3);
% The +64 in the y-dimension makes room for the category caption
% The +10 creates a border around each tile in the montage
% The two extra columns make room for the reconstructed image.

confData = zeros(size(resultData));
for k=1:resultHdr.nf
    if any(evalCategoryIndices==k)
       confData(:,:,k) = interp1(confidenceTable(:,end), confidenceTable(:,k), resultData(:,:,k), "extrap");
    end%if
end%for

% Print confidences to resultsTextFile, if that file was defined
if ~isempty(resultsTextFile)
    printResultsToFile(confData, sprintf('Image %d', resultFrameNumber), resultsTextFile, classes, evalCategoryIndices, eps, 1.0, true);
end%if

maxConfidence = max(confData(:));
scaledConfData = (confData-heatMapThreshold)/(heatMapMaximum-heatMapThreshold);
thresholdConfData = max(min(scaledConfData,1.0),0.0);
winningIndex = find(confData==maxConfidence);
[~,~,winningFeature] = ind2sub(size(confData),winningIndex);
for k=1:numel(winningFeature)
   fprintf(1,'winning feature is %s, with confidence %.1f%%\n', classes{winningFeature(k)}, 100*maxConfidence);
end%for

imageDataBlend = imageBlendCoeff*imageDataGrayscale;
for k=1:numCategories
    category = displayCategoryIndices(k);
    categorycolumn = mod(k-1,numColumns)+1;
    categoryrow = (k-categorycolumn)/numColumns+1;
    assert(categoryrow==round(categoryrow));
    thisclass = classes{category};
    resultUpsampledY = upsamplefill(thresholdConfData(:,:,category),upsampleNx-1,'COPY');
    resultUpsampled = upsamplefill(resultUpsampledY',upsampleNy-1,'COPY')';
    %resultPngFilename = sprintf('tmp/result-frame%04d-category%02d.png',resultFrameNumber, category);
    tileImage = zeros([size(imageDataGrayscale),3]);
    for b=1:3
       tileImage(:,:,b) = thresholdConfColor(b) + (maxConfColor(b)-thresholdConfColor(b))*resultUpsampled;
       tileImage(:,:,b) = imageBlendCoeff * imageDataGrayscale + (1-imageBlendCoeff) * tileImage(:,:,b);
    end%for
    if (size(tileImage,3)==1), tileImage=repmat(tileImage,[1 1 3]); end;

    %%%%% Bounding Box Logic
    if drawBoundingBoxes && any(resultUpsampled(:));
       [yy xx]       = find(resultUpsampled);
       nactive       = length(yy);
       class_vector      = zeros(nactive,2);
       class_vector(:,1) = xx; %class_vector must be [x y] coordinate pairs for dbscan
       class_vector(:,2) = yy;

       [cluster_vector, type_vector, dbscanEps] = dbscan(class_vector, dbscanDensity, dbscanEps); %dbscan clustering algorithm
       max_class_vector = max(cluster_vector); % how many clusters

       if max_class_vector < 0
          disp(['No clusters were made by dbscan!'])
       else
          disp(['Number of clusters = ',num2str(max_class_vector)])
          clusterpoints = cat(2,cluster_vector',class_vector);

          for l=1:max_class_vector
             classpoints = clusterpoints(clusterpoints(:,1)==l,2:3);
             bbox{l}(1) = min(classpoints(:,2)); % min y
             bbox{l}(2) = min(classpoints(:,1)); % min x
             bbox{l}(3) = max(classpoints(:,2)); % max y
             bbox{l}(4) = max(classpoints(:,1)); % max x
             %% TODO: make sure this doesn't step outside the image (make boxes encompass max/min points)
             tileImage(bbox{l}(1) : bbox{l}(1) + boundingBoxThickness, bbox{l}(2) :  bbox{l}(4), 1) = 1;
             tileImage(bbox{l}(3) - boundingBoxThickness : bbox{l}(3), bbox{l}(2) :  bbox{l}(4), 1) = 1;
             tileImage(bbox{l}(1) : bbox{l}(3), bbox{l}(2) : bbox{l}(2) + boundingBoxThickness, 1) = 1;
             tileImage(bbox{l}(1) : bbox{l}(3), bbox{l}(4) - boundingBoxThickness : bbox{l}(4), 1) = 1;
          end
       end
    end
    %%%%%

    maxConfCategory = max(max(confData(:,:,category)));
    if any(winningFeature==category) && maxConfCategory >= highlightThreshold;
        captionColor = 'blue';
    else
        captionColor = 'gray';
    end%if

    file = sprintf('tmp/label%s.png', classes{category});
    makeLabelCommand = sprintf('convert -background white -fill %s -size %dx32 -pointsize 24 -gravity center label:%s %s', captionColor, size(imageDataGrayscale,2), classes{category}, file);
    system(makeLabelCommand);
    img = readImageMagickFile(file);
    delete(file);

    valueFile = sprintf('tmp/value%s.png', classes{category});
    makeValueCommand = sprintf('convert -background white -fill %s -size %dx32 -pointsize 24 -gravity center label:%.1f%% %s', captionColor, size(imageDataGrayscale,2), 100*maxConfCategory, valueFile);
    system(makeValueCommand);
    valueImage = readImageMagickFile(valueFile);
    delete(valueFile);

    xstart = (size(imageDataGrayscale,2)+10)*(categorycolumn-1)+5;
    ystart = (size(imageDataGrayscale,1)+64+10)*(categoryrow-1)+5;
    % The +10 provides a 10-pixel border around each image.
    % The +5 places the tile in the middle of the region with 10-pixel border.
    % The +64 is because each tile includes the caption, which is 64 pixels high.
    montageImage(ystart+(1:32),xstart+(1:size(imageDataGrayscale,2)),:) = img;
    montageImage(ystart+(33:64),xstart+(1:size(imageDataGrayscale,2)),:) = valueImage;
    montageImage(ystart+64+(1:size(imageDataGrayscale,1)),xstart+(1:size(imageDataGrayscale,2)),:) = tileImage;
end%for

[reconPvp,reconHdr] = readpvpfile(reconPvpFile, [], reconFrameNumber, reconFrameNumber);
if (reconHdr.filetype != 4)
   error("heatMapMontage:expectingnonsparse","heatMapMontage expects %s to be a nonsparse layer",reconPvpFile);
end%if
reconData = permute(reconPvp{1}.values,[2 1 3]);
reconMax = max(reconData(:));
reconMin = min(reconData(:));
reconData = (reconData-reconMin)/(reconMax-reconMin+(reconMax==reconMin));
if size(reconData,3)==1
   reconData = repmat(reconData,[1 1 3]);
end%if


xstart = floor((numColumns+0.5)*(size(reconData,2)+10));
ystart = 5;
if (size(imageData,3)==1)
   montageImage(ystart+64+(1:size(imageData,1)), xstart+(1:size(imageData,2)), :) = repmat(imageData,[1 1 3]);
else
   montageImage(ystart+64+(1:size(imageData,1)), xstart+(1:size(imageData,2)), :) = imageData;
end%if

file = 'tmp/labelimage.png';
makeLabelCommand = sprintf('convert -background black -fill %s -size %dx32 -pointsize 24 -gravity center label:%s %s', "white", size(reconData,2), "\"original image\"", file);
system(makeLabelCommand);
img = readImageMagickFile(file);
montageImage(ystart+(33:64), xstart+(1:size(imageDataGrayscale,2)), :) = img;
delete(file);


%use the same xstart
ystart = size(reconData,1)+64+10+5;
montageImage(ystart+64+(1:size(reconData,1)), xstart+(1:size(reconData,2)), :) = reconData;

file = 'tmp/labelrecon.png';
makeLabelCommand = sprintf('convert -background black -fill %s -size %dx32 -pointsize 24 -gravity center label:%s %s', "white", size(reconData,2), "reconstruction", file);
system(makeLabelCommand);
img = readImageMagickFile(file);
montageImage(ystart+(33:64),xstart+(1:size(imageDataGrayscale,2)),:) = img;
delete(file);


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
