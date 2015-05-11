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
resultData = max(resultData-1,0);

classes={'background'; 'aeroplane'; 'bicycle'; 'bird'; 'boat'; 'bottle'; 'bus'; 'car'; 'cat'; 'chair'; 'cow'; 'diningtable'; 'dog'; 'horse'; 'motorbike'; 'person'; 'pottedplant'; 'sheep'; 'sofa'; 'train'; 'tvmonitor'};
categoryindices=[15 16]; %% to be determined :-)
numcategories=numel(categoryindices);

if(numel(classes)!=resultHdr.nf)
   error("heatMapMontage:wrongnf","number of classes is %d but %s has %d features.",numel(classes),resultPvpFile,resultHdr.nf);
end%if

montagerows = floor(sqrt(numcategories));
montagecols = ceil(numcategories/montagerows);
assert(montagerows*montagecols>=numcategories);
maxResultData = 0.50; % max(resultData(:)); % the confidence that will be mapped to maximum brightness
upsampleNx = size(imageData,2)/size(resultData,2);
upsampleNy = size(imageData,1)/size(resultData,1);
assert(upsampleNx==round(upsampleNx));
assert(upsampleNy==round(upsampleNy));

imagePngFilename = sprintf('tmp/image%04d.png', imageFrameNumber);
imwrite(imageData,imagePngFilename);

cmap = zeros(256,3);
zeroconfcolor = [0.5 0.5 0.5];
maxconfcolor = [0 1 0];
for k=1:256
   cmap(k,:) = (k-1)/255*maxconfcolor + (256-k)/255*zeroconfcolor;
end%for

for k=1:numcategories
    category = categoryindices(k);
    thisclass = classes{category};
    resultDataTrunc = max(resultData(:,:,category),0);
    if maxResultData != 0
        resultDataTrunc = min(resultDataTrunc/maxResultData,1);
    end%if
    resultUpsampledY = upsamplefill(resultDataTrunc,upsampleNx-1,'COPY');
    resultUpsampled = upsamplefill(resultUpsampledY',upsampleNy-1,'COPY')';
    %rgb = [0 1 0]; % hsv2rgb([(k-1)/numcategories, 1, 1]); % commented out since we're using the same cmap for each category
    %cmap = bsxfun(@times, rgb, linspace(0,1,256)');
    resultPngFilename = sprintf('tmp/result-frame%04d-category%02d.png',resultFrameNumber, category);
    %resultUpsampled = (1+resultUpsampled)/2;
    resultUpsampled = resultUpsampled*255;
    imwrite(resultUpsampled, cmap, resultPngFilename);
    compositedFilename = sprintf('tmp/composited-frame%04d-category%02d.png', imageFrameNumber, category);
    compositecommandstring = sprintf('composite -blend 30%% %s %s %s', imagePngFilename, resultPngFilename, compositedFilename);
    status = system(compositecommandstring); assert(status==0);
end%for

montagecommandstring = 'montage ';
for k=1:numcategories
    category = categoryindices(k);
    thisclass = classes{category};
    montagecommandstring = [montagecommandstring, 'labels/label', thisclass, '.png' ' '];
end%for
for k=1:numcategories
    category = categoryindices(k);
    compositedFilename = sprintf('tmp/composited-frame%04d-category%02d.png',imageFrameNumber, category);
    montagecommandstring = [montagecommandstring, compositedFilename, ' '];
end%for
montagecommandstring = [montagecommandstring, '-tile ', num2str(numcategories), 'x2 '];
montagecommandstring = [montagecommandstring, montagePath];

status = system(montagecommandstring); assert(status==0);

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

%for k=1:numcategories
%    category = categoryindices(k);
%    resultPngFilename = sprintf('tmp/result-frame%04d-category%02d.png',resultFrameNumber, category);
%    delete(resultPngFilename);
%    compositedFilename = sprintf('tmp/composited-frame%04d-category%02d.png',imageFrameNumber, category);
%    delete(compositedFilename);
%end%for
%delete(imagePngFilename);
%
%if ownstmpdir
%   rmdir tmp;
%end%if
