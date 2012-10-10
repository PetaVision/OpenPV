global lifInhPatchX; lifInhPatchX = 21;
global lifInhPatchY; lifInhPatchY = 21;
global deltaT; deltaT = 1;
global tLCA; tLCA = 20;
global columnSizeY; columnSizeY = 256;
global columnSizeX; columnSizeX = 256;
global FNUM_ALL; FNUM_ALL = 1;         %1 for all frames, 0 for FNUM_SPEC

global NUM_PROCS; NUM_PROCS = nproc();

rootDir                                    = '/Users/slundquist';
workspaceDir                               = [rootDir,'/Documents/workspace/iHouse'];
pvpDir                                     = [workspaceDir,'/denseOutput/'];
outputDir                                  = [workspaceDir,'/denseOutput/analysis/'];
postActivityFile                           = [pvpDir,'lif.pvp'];

if (exist(outputDir, 'dir') ~= 7)
   mkdir(outputDir);
end


function coorFunc(activityData)
   global lifInhPatchX lifInhPatchY;
   global columnSizeX columnSizeY;
   global tLCA;
   global deltaT;
   global NUM_PROCS;
   timeSteps = length(activityData);

   %Grab max distance
   maxDist = round(sqrt((lifInhPatchX/2)^2 + (lifInhPatchY/2)^2));

   mask = zeros(columnSizeY, columnSizeX);
   mask(1+maxDist:columnSizeY-maxDist, 1+maxDist:columnSizeX-maxDist) = 1;
   marginIndex = find(mask'(:));

   %Data structure
   pixDist = cell(maxDist, 1);
   %Make offset of x and y with respect to center of circle
   for d = 1:maxDist
      tempX = [];
      tempY = [];
      delta_theta = atan(1/d);
      for theta = 0:delta_theta:2*pi
         tempX = [tempX; round(d * cos(theta))];
         tempY = [tempY; round(d * sin(theta))];
      end
      pixDist{d}.x = tempX;
      pixDist{d}.y = tempY;
   end

   %Define output matrix of values
   outMat = zeros(maxDist, timeSteps);
   intSpike = zeros(columnSizeY, columnSizeX, timeSteps);

   if (mod(length(marginIndex), NUM_PROCS) == 0)
      procSize = floor(length(marginIndex) / NUM_PROCS);
      cellIndex = mat2cell(marginIndex, ones(1, NUM_PROCS) .* procSize, 1);
   else
      procSize = floor(length(marginIndex) / (NUM_PROCS - 1));
      lastSize = mod(length(marginIndex), NUM_PROCS - 1);
      cellIndex = mat2cell(marginIndex, [ones(1, NUM_PROCS - 1) .* procSize, lastSize], 1);
   end
   
   %Calculate intSpike for all time
   disp('Calculating intSpike');
   fflush(1);
   for ts = 1:(timeSteps - 1)
      intSpike(:, :, ts+1) = updateIntSpike(intSpike(:, :, ts), tLCA, deltaT, activityData{ts}.values, columnSizeX, columnSizeY);
   end
   disp('Done');
   fflush(1);

   %Put intSpike into cell array for cell fun
   cellIntSpike{1} = intSpike;
   cellPixDist{1} = pixDist;
   cellTimeSteps{1} = timeSteps;
   %Uniform output as false to store in cell arrays
%   [out] = parcellfun(NUM_PROCS, @parFindMean, cellIndex, cellIntSpike, cellPixDist, 'UniformOutput', 0);
   [out] = cellfun(@parFindMean, cellIndex, cellIntSpike, cellPixDist, cellTimeSteps, 'UniformOutput', 0);
   %Calculate average based on all pixels
   for i = 1:length(out)
      outMat += out{i};
   end
   
   outMat = outMat./length(marginIndex);

   figure;
   hold all;
   for d = 1:maxDist
      plot(outMat(d, :));
   end
   hold off;
end

function [out] = parFindMean(index, intSpike, pixDist, timeSteps) 
   %Put index into a cell array for cell fun
  % cellIntSpike{1} = intSpike;
  % cellMargIdx{1} = index;
   out = zeros(timeSteps, length(pixDist));
   %Put intSpike into a cell to avoid arrayfun iterating
   cIntSpike{1} = intSpike;

   for ts = 1:timeSteps
      for d=1:length(pixDist)
         %Put pix dist into a cell to avoid arrayfun iterating
         cPixDist{1} = pixDist{d};
         cTs{1} = ts;
         %Find the mean of all indicies of that specific distance
         aOut = arrayfun(@findMean, index, cPixDist, cIntSpike, cTs);
         %Add it up, and store based on distance
         out(ts, d) = sum(aOut);
      end
   end
  %out = parcellfun(4, @findMean, pixDist, cellMargIdx, cellIntSpike);
end

function [outMean] = findMean(idx, dist, intSpike, ts) 
   global columnSizeX columnSizeY;
   [mIx mIy] = ind2sub([columnSizeX columnSizeY], idx);
   %Using x and y offset, grab actual pixels based on center point
   xCoord = dist{1}.x + mIx;
   yCoord = dist{1}.y + mIy;
   outMean = mean(intSpike{1}(yCoord, xCoord, ts{1})(:));
end

function [outIntSpike] = updateIntSpike(inIntSpike, tLCA, deltaT, activityIndex, colX, colY)
   if isempty(activityIndex)
      outIntSpike = inIntSpike;
      return
   end
   %Change activity sparse matrix to full matrix
   sparse_size = colX * colY;
   activity = sparse(activityIndex + 1, 1, 1, sparse_size, 1, length(activityIndex));
   actMat = reshape(full(activity), [colX, colY]);
   actMat = flipud(rot90(actMat));
   
   %Update integrated spike
   outIntSpike = exp(-deltaT/tLCA) .* (inIntSpike + actMat); 
end

[data hdr] = readpvpfile(postActivityFile, outputDir, 0);
coorFunc(data);
