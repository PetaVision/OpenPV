clear all;
global lifInhPatchX; lifInhPatchX = 21;
global lifInhPatchY; lifInhPatchY = 21;
global deltaT; deltaT = 1;
global tLCA; tLCA = 20;
global columnSizeY; columnSizeY = 256;
global columnSizeX; columnSizeX = 256;

global NUM_PROCS; NUM_PROCS =  nproc();

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

   
   disp('Calculating intSpike');
   fflush(1);
   %Change activitydata into sparse matrix
   activity = activityData.spikeVec;
   time = activityData.frameVec;
   numactivity = columnSizeX * columnSizeY;
   timesteps = activityData.numframes;
   sparse_act = sparse(activity, time, 1, numactivity, timesteps);

   %Create decay kernel
   tau_kernel = exp(-[0:128]/20);
   tau_kernel = [zeros(1, 129), tau_kernel];

   %Create intSpikeCount matrix where it is indexed by (vectorized index, timestep)
   intSpike = conv2(sparse_act, tau_kernel, 'same');
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
   outMat = zeros(maxDist, timesteps);
%   intSpike = zeros(columnSizeY, columnSizeX, timesteps);

   %Split margin index into number of processes
   if (mod(length(marginIndex), NUM_PROCS) == 0)
      procSize = floor(length(marginIndex) / NUM_PROCS);
      cellIndex = mat2cell(marginIndex, ones(1, NUM_PROCS) .* procSize, 1);
   else
      procSize = floor(length(marginIndex) / (NUM_PROCS - 1));
      lastSize = mod(length(marginIndex), NUM_PROCS - 1);
      cellIndex = mat2cell(marginIndex, [ones(1, NUM_PROCS - 1) .* procSize, lastSize], 1);
   end
%
%   %Split intSpike matrix into row strips based on number of processes
%   if (mod(columnSizeY * columnSizeX, NUM_PROCS) == 0)
%      procSize = floor(columnSizeY / NUM_PROCS);
%      lastSize = 0;
%      cellIntSpike = mat2cell(intSpike, columnSizeX,  ones(1, NUM_PROCS) .* procSize);
%   else
%      procSize = floor(columnSizeY / (NUM_PROCS - 1));
%      lastSize = mod(columnSizeY , NUM_PROCS - 1);
%      cellIntSpike= mat2cell(intSpike, columnSizeX, [ones(1, NUM_PROCS - 1) .* procSize, lastSize]);
%   end
%
%
%   %Define starting and ending points based on row strips
%   startY = cell(NUM_PROCS);
%   endY = cell(NUM_PROCS);
%
%   for i = 1:(NUM_PROCS - 1)
%      startY{i} = (i-1) * procSize + 1;
%      endY{i} = i * procSize;
%   end
%   startY{NUM_PROCS} = columnSizeY - lastSize + 1;
%   endY{NUM_PROCS} = columnSizeY;


   %Calculate intSpike for all time

%   intSpikeOut = parcellfun(NUM_PROCS, @updateIntSpike, cellIntSpike, activityData{ts}.values, startY, endY); 
      %intSpike(:, :, ts+1) = updateIntSpike(intSpike(:, :, ts), activityData{ts}.values, columnSizeX, columnSizeY);

   disp('Done');
   fflush(1);

   %Put intSpike into cell array for cell fun
   cellIntSpike{1} = intSpike;
   cellPixDist{1} = pixDist;
   cellTimeSteps{1} = timesteps;
   %Uniform output as false to store in cell arrays
   if NUM_PROCS == 1
      [out] = cellfun(@parFindMean, cellIndex, cellIntSpike, cellPixDist, cellTimeSteps, 'UniformOutput', 0);
   else
      [out] = parcellfun(NUM_PROCS, @parFindMean, cellIndex, cellIntSpike, cellPixDist, cellTimeSteps, 'UniformOutput', 0);
   end
%   [out] = cellfun(@parFindMean, cellIndex, cellIntSpike, cellPixDist, cellTimeSteps, 'UniformOutput', 0);
   %Calculate average based on all pixels
   for i = 1:length(out)
      outMat += out{i};
   end
   
   %Divide by total number of idicies to find average
   outMat = outMat./length(marginIndex);

   %Divide by tau squared to make value a rate
   outMat = outMat ./ (tLCA * tLCA);

   for d = 1:maxDist
      figure;
      plot(outMat(d, :));
      print_filename = [outputDir, 'Coorfunc_', num2str(d), '.jpg'];
      print(print_filename);
   end
   
end

function [out] = parFindMean(index, intSpike, pixDist, timeSteps) 
   %Put index into a cell array for cell fun
  % cellIntSpike{1} = intSpike;
  % cellMargIdx{1} = index;
   out = zeros(length(pixDist), timeSteps);
   %Put intSpike into a cell to avoid arrayfun iterating
   cIntSpike{1} = intSpike;
   for d=1:length(pixDist)
      %Put pix dist into a cell to avoid arrayfun iterating
      cPixDist{1} = pixDist{d};
      %Find the mean of all indicies of that specific distance
      aOut = arrayfun(@findMean, index, cPixDist, cIntSpike, 'UniformOutput', 0);
      %Combine and reshape into matrix
      %Matrix is (sets of sumed indicies for given distance, timesteps)
      mOut = reshape([aOut{:}], length(aOut), timeSteps); 
      %Add it up and store by distance
      out(d, :) = sum(mOut);
   end
   %Out is now a [list of distance, timesteps] matrix
end

%Idx is specific center index
%Dist is a list of indicies that make up a circle for a specific distance
%intSpike is a matrix of integrated spike count that is defined as (pos, time)
function [outMean] = findMean(idx, dist, intSpike) 
   global columnSizeX columnSizeY;
   [mIx mIy] = ind2sub([columnSizeX columnSizeY], idx);
   %Using x and y offset, grab circle pixels based on center point
   xCoord = dist{1}.x + mIx;
   yCoord = dist{1}.y + mIy;
   %Change back into ind
   circIdx = sub2ind([columnSizeX columnSizeY], xCoord, yCoord);
   %A matrix that contains the int spike count as (Circle Positions, Time steps)
   circleIntSpike = intSpike{1}(circIdx, :);
   %A vector that contains the int spike count in time
   centerIntSpike = intSpike{1}(idx, :);
   %Repeate to make the same size as circleIntSpike
   centerIntSpike = repmat(centerIntSpike, length(circIdx), 1);
   %Multiply and sum
   prodMat = circleIntSpike .* centerIntSpike;
   outMean = sum(prodMat);
   outMean = outMean ./ length(circIdx);
end

%function [outIntSpike] = updateIntSpike(inIntSpike, tLCA, deltaT, activityIndex, begPosY, endPosY)
%   global tLCA;
%   global deltaT;
%   global columnSizeX columnSizeY;
%   if isempty(activityIndex)
%      outIntSpike = inIntSpike;
%      return
%   end
%
%   %Find range of values for activity index to be in
%   %Start index of 1
%   idxStart = begPosY * columnSizeX + 1;
%   idxEnd = (endPosY + 1) * columnSizeX;
%   
%%   %Change activity sparse matrix to full matrix
%%   sparse_size = colX * colY;
%%   activity = sparse(activityIndex + 1, 1, 1, sparse_size, 1, length(activityIndex));
%%   actMat = reshape(full(activity), [colX, colY]);
%%   actMat = flipud(rot90(actMat));
%   
%
%   
%   %Update integrated spike
%   for i = 1:length(activityIndex)
%      
%      outIntSpike = exp(-deltaT/tLCA) .* (inIntSpike + actMat); 
%   end
%   
%end

data = readactivitypvp(postActivityFile);
coorFunc(data);
