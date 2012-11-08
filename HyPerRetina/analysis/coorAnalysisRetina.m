clear all;
global lifInhPatchX; lifInhPatchX = 21;
global lifInhPatchY; lifInhPatchY = 21;
global deltaT; deltaT = 1;
global tLCA; tLCA = 33; %% use  fraction of framerate = 33
global columnSizeY; columnSizeY = 1080/2; %%256;
global columnSizeX; columnSizeX = 1920/2; %%256;

global PRINT_N; PRINT_N = 1;
global CALC_COOR; CALC_COOR = 0;

global FNUM_ALL; FNUM_ALL = 0;           %All farmes
num_frames = tLCA * 450;
num_frames_per_block = floor(tLCA * 5);
num_blocks = floor(num_frames / num_frames_per_block);
num_frames = num_frames_per_block * num_blocks;
global FNUM_SPEC; 
FNUM_SPEC    = cell(1, num_blocks);
for i_block = 1 : num_blocks
  FNUM_SPEC{1,i_block} = ...
      [(i_block-1)*num_frames + 1 : i_block*num_frames]; 
endfor
%%FNUM_SPEC = ...
%%{...    %start:int:end frames
%%   [34:33*4]...
%%   [33*4+1:33*8]...
%%   [33*8+1:33*12]...
%%   [33*12+1:33*16]...
%%   [33*16+1:33*20]...
%%};

global NUM_PROCS; NUM_PROCS = 1;%% nproc();

rootDir                                    = '/Users/garkenyon';
workspaceDir                               = [rootDir,'/workspace-sync-anterior/HyPerRetina'];
pvpDir                                     = [rootDir, '/NeoVision2/neovision-programs-petavision/Heli/Challenge/026/p0/ns14850/'];
outDir                                     = pvpDir;
global outputDir; outputDir                = [outDir, 'coor/'];
global outputMovieDir; outputMovieDir      = [outputDir, 'nMovieGanglionOFF/'];
postActivityFile                           = [pvpDir,'GanglionOFF.pvp'];

if (exist(outDir, 'dir') ~= 7)
   mkdir(outDir);
end
if (exist(outputDir, 'dir') ~= 7)
   mkdir(outputDir);
end
if (exist(outputMovieDir, 'dir') ~= 7)
   mkdir(outputMovieDir);
end


function coorFunc(activityData)
   global lifInhPatchX lifInhPatchY;
   global columnSizeX columnSizeY;
   global tLCA;
   global deltaT;
   global NUM_PROCS;
   global outputDir;
   global outputMovieDir;
   global FNUM_SPEC;
   global FNUM_ALL;
   global PRINT_N;
   global CALC_COOR;

   if FNUM_ALL > 0
      FNUM_SPEC = cell(1, 1);
      FNUM_SPEC{1} = 1;
   end
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   disp('Calculating range coordinates');
   fflush(1);
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %Grab max distance
   maxDist = round(sqrt((lifInhPatchX/2)^2 + (lifInhPatchY/2)^2));

   %Mask with non-margin values based on max distance on coor
   mask = zeros(columnSizeY, columnSizeX);
   mask(1+maxDist:columnSizeY-maxDist, 1+maxDist:columnSizeX-maxDist) = 1;
   marginIndex = find(mask'(:));

   %Calculate data structure of points based on the distance away
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

   for c = 1:length(FNUM_SPEC)
      %Change activitydata into sparse matrix
      activity = activityData{c}.spikeVec;
      time = activityData{c}.frameVec;
      numactivity = columnSizeX * columnSizeY;
      timesteps = activityData{c}.numframes;
      sparse_act = sparse(activity, time, 1, numactivity, timesteps);

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      disp('Calculating intSpike');
      fflush(1);
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %Create decay kernel
      tau_kernel = exp(-[0:(5*tLCA)]/tLCA);
      tau_kernel = [zeros(1, (5*tLCA)+1), tau_kernel];

      %Split activity into number of processes
      if (mod(numactivity, NUM_PROCS) == 0)
         procSize = floor(numactivity / NUM_PROCS);
         cellAct = mat2cell(sparse_act, ones(1, NUM_PROCS) .* procSize, timesteps);
      else
         procSize = floor(numactivity / (NUM_PROCS - 1));
         lastSize = mod(numactivity, NUM_PROCS - 1);
         cellAct = mat2cell(sparse_act, [ones(1, NUM_PROCS - 1) .* procSize, lastSize], timesteps);
      end

      %Set rest of variables as cells for parcellfun
      cTau_Kernel{1} = tau_kernel;
      cIdentity{1} = [1];
      cShape{1} = 'same';

      %Create intSpikeCount matrix where it is indexed by (vectorized index, timestep)
      %intSpike = conv2(sparse_act, tau_kernel, 'same');
      %%if NUM_PROCS == 1
      %%  cIntSpike = cellfun(@conv2, cTau_Kernel, cIdentity, cellAct, cShape, 'UniformOutput', false);
      %%else
      %%  cIntSpike = parcellfun(NUM_PROCS, @conv2, cTau_Kernel, cIdentity, cellAct, cShape, 'UniformOutput', false);
      %%end
      if NUM_PROCS == 1
        cIntSpike = cellfun(@conv2, cellAct, cTau_Kernel, cShape, 'UniformOutput', false);
      else
        cIntSpike = parcellfun(NUM_PROCS, @conv2, cellAct, cTau_Kernel, cShape, 'UniformOutput', false);
      end

      %Recombine from cells, needs to be rotated for collection of cell arrays
      cIntSpike = cellfun(@(x) x', cIntSpike, 'UniformOutput', false);
      intSpike = [cIntSpike{:}]';

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      disp('Printing integrated write steps');
      fflush(1);
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if(PRINT_N > 0)
         m = max(intSpike(:));

         for t = 1:timesteps
            nOutMat = zeros(columnSizeY, columnSizeX);
            i = 1:numactivity;
            nOutMat = nOutMat';
            nOutMat(i) = intSpike(:, t);
            nOutMat = nOutMat';
            nOutMat = nOutMat ./ m;
            print_filename = [outputMovieDir, 'n_', num2str(c) , '_' , num2str(t), '.jpg'];
            imwrite(nOutMat, print_filename);
         end
      end

      if (CALC_COOR > 0)
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      disp('Calculating coorlation function');
      fflush(1);
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %Define output matrix of values
      outMat = zeros(maxDist, timesteps);
      %Split margin index into number of processes
      if (mod(length(marginIndex), NUM_PROCS) == 0)
         procSize = floor(length(marginIndex) / NUM_PROCS);
         cellIndex = mat2cell(marginIndex, ones(1, NUM_PROCS) .* procSize, 1);
      else
         procSize = floor(length(marginIndex) / (NUM_PROCS - 1));
         lastSize = mod(length(marginIndex), NUM_PROCS - 1);
         cellIndex = mat2cell(marginIndex, [ones(1, NUM_PROCS - 1) .* procSize, lastSize], 1);
      end

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
      %Calculate average based on all pixels
      for i = 1:length(out)
         outMat += out{i};
      end
      %Divide by total number of idicies to find average
      outMat = outMat./length(marginIndex);


      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      disp('Plotting');
      fflush(1);
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %Plot
      legName = cell(maxDist, 1);
      colorStep = 1/maxDist;
      figure('Visible', 'off');
      hold all;
      for d = 1:maxDist
         blueColorVal = colorStep * d;
         redColorVal = 1 - blueColorVal;
         plot(FNUM_SPEC{c}, outMat(d, :), 'Color', [redColorVal, 128, blueColorVal]);
         legName{d} = ['Dist: ', num2str(d)];
         %plot(mean(intSpike));
      end
      hold off;
      legend(legName);
      print_filename = [outputDir, 'Coorfunc_', num2str(c), '.jpg'];
      print(print_filename);
    end%% CALC_COOR
    end%% FNUM_SPECS
 end %% function

%index is an array of indexes to calculate coor function
%intSpike is integrated spike count that is defined as (pos, time)
%pixDist is a cell array structure that contains the offset of x and y
%   coordinates to create a circle given by pixDist{radius}
function [out] = parFindMean(index, intSpike, pixDist, timeSteps) 
   %Allocate out matrix
   out = zeros(length(pixDist), timeSteps);
   %Put intSpike into a cell to avoid arrayfun iterating
   cIntSpike{1} = intSpike;
   %Iterate through distances
   for d=1:length(pixDist)
      %Put pix dist into a cell to avoid arrayfun iterating
      cPixDist{1} = pixDist{d};
      %Find the mean of all indicies of that specific distance
      aOut = arrayfun(@findMean, index, cPixDist, cIntSpike, 'UniformOutput', 0);
      %Combine and reshape into matrix
      %Matrix is (sets of sumed indicies for given distance, timesteps)
      %Recombine from cells, needs to be rotated for collection of cell arrays
      mOut = cellfun(@(x) x', aOut, 'UniformOutput', false);
      mOut = [mOut{:}]';
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
   global tLCA;
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
   prodMat = (circleIntSpike .* centerIntSpike)./(tLCA * tLCA);
   outMean = sum(prodMat);
   outMean = outMean ./ length(circIdx);
end


%Script
%Grab activity data
data = readactivitypvp(postActivityFile);
%Run coorFunc
coorFunc(data);
