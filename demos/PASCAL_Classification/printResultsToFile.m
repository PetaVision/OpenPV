function printResultsToFile(confidenceData, imageLabel, outputTextFilename, classes, evalCategoryIndices, minConfidence, maxConfidence, appendFlag)

if ~exist('confidenceData','var') || isempty(confidenceData)
    error("confidenceData must be a nonempty array of numeric type");
end%if
if ~isnumeric(confidenceData)
    error("printResultsToFile:nonnumeric","confidenceData must be of numeric type, not %s", class(confidenceData));
end%if
if numel(size(confidenceData))>3
    error("printResultsToFile:tooManyDims","confidenceData must be a numeric array of no more than three dimensions");
end%if

nx=size(confidenceData,1);
ny=size(confidenceData,2);
nf=size(confidenceData,3);

if ~exist('imageLabel', 'var') || ~ischar(imageLabel) || ~isvector(imageLabel) || size(imageLabel,1)~=1
    error("printResultsToFile:badFrameNumber","printResultsToFile error: imageLabel must be a string");
end%if

if ~exist('outputTextFilename','var') || isempty(outputTextFilename)
   fid =  1; % standard output
elseif ~ischar(outputTextFilename) || ~isvector(outputTextFilename) || size(outputTextFilename,1)~=1
   error("printResultsToFile:badOutputTextFilename","printResultsToFile error: outputTextFilename must be a string");
elseif exist('appendFlag','var') && appendFlag
   fid = fopen(outputTextFilename, "a");
else
   fid = fopen(outputTextFilename, "w");
end%if
if fid<0
   error("printResultsToFile:cannotopen","printResultsToFile error: unable to open \"%s\"", outputTextFilename);
end%if

if isempty(classes)
   classes = cell(nf,1);
   for f=1:nf
      classes{f,1}=num2str(f);
   end%for
end%if
if nf==1 && ~iscell(classes)
   classes = {classes};
   assert(iscell(classes));
end%if
if ~isequal(class(classes),'cell')
   error("printResultsToFile:classescell","printResultsToFile error: more than one category (%d) in confidenceData but input variable classes is not a cell array", nf);
end%if
if numel(classes)<nf
   error("printResultsToFile:toofewclasses","printResultsToFile error: only %d class labels defined but confidenceData has %d categories", numel(classes), nf);
end%if
for f=1:nf
   c=classes{f};
   if ~ischar(c) || ~isvector(c) || size(c,1)~=1
      error("printResultsToFile:badclasslabel","printResultsToFile error: class label %d must be a string", f);
   end%if
end%for

if ~exist('evalCategoryIndices','var') || isempty(evalCategoryIndices)
   evalCategoryIndices=1:nf;
   printf("evalCategoryIndices is empty; setting to 1:%d\n", nf);
end%if
if ~isnumeric(evalCategoryIndices) || ~isvector(evalCategoryIndices) || ...
   ~isequal(round(evalCategoryIndices),evalCategoryIndices) || any(evalCategoryIndices<=0)
   error("printResultsToFile:badCategoryIndices","printResultsToFile error: evalCategoryIndices must be a vector of positive integers");
end%if
if max(evalCategoryIndices)>nf
   error("printResultsToFile:categoryIndexTooBig","printResultsToFile error: evalCategoryIndices must be a vector of integers between 1 and the number of categories %d (max of evalCategoryIndices is %d)", nf, max(evalCategoryIndices));
end%if

for y=1:ny
   for x=1:nx
      for f=1:numel(evalCategoryIndices)
         category=evalCategoryIndices(f);
         conf=confidenceData(x,y,category);
         if conf>=minConfidence && conf<=maxConfidence
            fprintf(fid,"%s, x=%d, y=%d, %-12s confidence=%f%%\n", imageLabel, x, y, classes{category}, 100*conf);
         end%if
      end%for
   end%for
end%for

if fid>=3
   fclose(fid);
end%if
