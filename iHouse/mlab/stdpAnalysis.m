clear all; close all; more off; clc;
system("clear");

%Nessessary Petavision Variables 
global postNxScale; postNxScale = 4; 
global postNyScale; postNyScale = 4;
global lifInhPatchX; lifInhPatchX = 21;
global lifInhPatchY; lifInhPatchY = 21;
global deltaT; deltaT = 1;
global tLCA; tLCA = 20;
rootDir                                    = '/Users/slundquist';
workspaceDir                               = [rootDir,'/Documents/workspace/iHouse'];
pvpDir                                     = [workspaceDir,'/denseOutput/'];
outputDir                                  = [workspaceDir,'/denseOutput/analysis/'];

%Reconstruct Flags
global SPIKING_POST_FLAG;      SPIKING_POST_FLAG      = 1;  %Create spiking post output flag
global SPIKING_PRE_FLAG;       SPIKING_PRE_FLAG       = 0;

%Spiking Output
global FNUM_ALL; FNUM_ALL = 0;         %1 for all frames, 0 for FNUM_SPEC
global FNUM_SPEC; FNUM_SPEC    = {...    %start:int:end frames
   [10000:20000]...
   [50000:60000]...
   [90000:100000]...
   [130000:140000]...
   [180000:190000]...
};

global RECONSTRUCTION_FLAG;    RECONSTRUCTION_FLAG    = 0;  %Create reconstructions
global WEIGHTS_MAP_FLAG;  WEIGHTS_MAP_FLAG  = 0;     %Create weight maps
global WEIGHTS_CELL_FLAG; WEIGHTS_CELL_FLAG = 0;
global CELL; CELL = {...
   [35, 20]...
   [10, 20]...
};        %X by Y of weigh cell

global WEIGHT_HIST_FLAG;  WEIGHT_HIST_FLAG  = 1;
global NUM_BINS;               NUM_BINS               = 20;

%global COOR_FLAG; COOR_FLAG = 1;

global VIEW_FIGS;  VIEW_FIGS  = 0;
global WRITE_FIGS; WRITE_FIGS = 1;
global GRAY_SC;    GRAY_SC    = 0;              %Image in grayscale

%Difference between On/Off if 0, seperate otherwise
global RECON_IMAGE_SC;   RECON_IMAGE_SC   = -1;        %-1 for autoscale
global WEIGHTS_IMAGE_SC; WEIGHTS_IMAGE_SC = -1; %-1 for autoscale
global GRID_FLAG;        GRID_FLAG        = 0;
global NUM_PROCS;        NUM_PROCS        = nproc();


%File names
%rootDir                                    = '/Users/dpaiton';
%workspaceDir                               = [rootDir,'/Documents/Work/LANL/workspace/iHouse'];
postActivityFile                           = [pvpDir,'lif.pvp'];
preOnActivityFile                          = [pvpDir,'RetinaON.pvp'];
preOffActivityFile                         = [pvpDir,'RetinaOFF.pvp'];
ONweightfile                               = [pvpDir,'w5_post.pvp'];
OFFweightfile                              = [pvpDir,'w6_post.pvp'];
readPostSpikingDir                         = [outputDir, 'postpvp/'];
readPreSpikingDir                          = [outputDir, 'prepvp/'];
readPreOnSpikingDir                        = [readPreSpikingDir, 'On/'];
readPreOffSpikingDir                       = [readPreSpikingDir, 'Off/'];
reconstructOutDir                          = [outputDir, 'reconstruct/'];
weightMapOutDir                            = [outputDir, 'weight_map/'];
cellMapOutDir                              = [outputDir, 'cell_map/'];
onWeightHistOutDir                         = [outputDir, 'on_hist/'];
offWeightHistOutDir                        = [outputDir, 'off_hist/'];
sourcefile                                 = [workspaceDir,'/output/DropInput.txt'];

%Make nessessary directories
if (exist(outputDir, 'dir') ~= 7)
   mkdir(outputDir);
end
if (exist(readPostSpikingDir, 'dir') ~= 7)
   mkdir(readPostSpikingDir);
end
if (exist(readPreSpikingDir, 'dir') ~= 7)
   mkdir(readPreSpikingDir);
end
if (exist(readPreOnSpikingDir, 'dir') ~= 7)
   mkdir(readPreOnSpikingDir);
end
if (exist(readPreOffSpikingDir, 'dir') ~= 7)
   mkdir(readPreOffSpikingDir);
end
if (exist(reconstructOutDir, 'dir') ~= 7)
   mkdir(reconstructOutDir);
end
if (exist(weightMapOutDir, 'dir') ~= 7)
   mkdir(weightMapOutDir);
end
if (exist(cellMapOutDir, 'dir') ~= 7)
   mkdir(cellMapOutDir);
end
if (exist(onWeightHistOutDir, 'dir') ~= 7)
   mkdir(onWeightHistOutDir);
end
if (exist(offWeightHistOutDir, 'dir') ~= 7)
   mkdir(offWeightHistOutDir);
end

numFun = 0;


%Read activity file in parallel
%Post Activity File
if(RECONSTRUCTION_FLAG || SPIKING_POST_FLAG)
   numFun += 1;
   fileName{numFun} = postActivityFile;
   output_path{numFun} = readPostSpikingDir;
   print{numFun} = SPIKING_POST_FLAG;
end
%Pre Activity File
if(SPIKING_PRE_FLAG)
   %On
   numFun += 1;
   fileName{numFun} = preOnActivityFile;
   output_path{numFun} = readPreOnSpikingDir;
   print{numFun} = SPIKING_PRE_FLAG;
   %Off
   numFun += 1;
   fileName{numFun} = preOffActivityFile;
   output_path{numFun} = readPreOffSpikingDir;
   print{numFun} = SPIKING_PRE_FLAG;
end
%Post weights
if(RECONSTRUCTION_FLAG || WEIGHTS_MAP_FLAG || WEIGHTS_CELL_FLAG || WEIGHT_HIST_FLAG)
   %On
   numFun += 1;
   fileName{numFun} = ONweightfile;
   output_path{numFun} = '';  %No output path needed for weights
   print{numFun} = 0;
   %Off
   numFun += 1;
   fileName{numFun} = OFFweightfile;
   output_path{numFun} = '';  %No output path needed for weights
   print{numFun} = 0;
end

disp('stdpAnalysis: Reading pvp files')
fflush(1);
if NUM_PROCS == 1
   [data hdr] = cellfun(@readpvpfile, fileName, output_path, print, 'UniformOutput', 0);
else
   [data hdr] = parcellfun(NUM_PROCS, @readpvpfile, fileName, output_path, print, 'UniformOutput', 0);
end

%Grab data from (par)cellfun
numFun = 0;
%Post Activity File
if(RECONSTRUCTION_FLAG || SPIKING_POST_FLAG) 
   numFun += 1;
   activityData = data{numFun};
   activityHdr = hdr{numFun};
end
%Pre Activity File, no data needed
if(SPIKING_PRE_FLAG)
   numFun += 2;
end
%Post weights
if(RECONSTRUCTION_FLAG || WEIGHTS_MAP_FLAG || WEIGHTS_CELL_FLAG || WEIGHT_HIST_FLAG)
   %On
   numFun += 1;
   weightDataOn = data{numFun};
   weightHdrOn = hdr{numFun};
   %Off
   numFun += 1;
   weightDataOff = data{numFun};
   weightHdrOff = hdr{numFun};
end

%If no analysis needed
if(~(RECONSTRUCTION_FLAG || WEIGHTS_MAP_FLAG || WEIGHTS_CELL_FLAG || WEIGHT_HIST_FLAG))
   return
end

%PetaVision params
if(RECONSTRUCTION_FLAG || WEIGHTS_MAP_FLAG || WEIGHTS_CELL_FLAG || WEIGHT_HIST_FLAG)
   assert(weightHdrOn.nbands == weightHdrOff.nbands);
   assert(weightHdrOn.nfp == weightHdrOff.nfp);
   assert(weightHdrOn.nxprocs == weightHdrOff.nxprocs);
   assert(weightHdrOn.nyprocs == weightHdrOff.nyprocs);
   assert(weightHdrOn.nxp == weightHdrOff.nxp);
   assert(weightHdrOn.nyp == weightHdrOff.nyp);
   assert(weightHdrOn.nx == weightHdrOff.nx);
   assert(weightHdrOn.ny == weightHdrOff.ny);
   assert(weightHdrOn.nxGlobal == weightHdrOff.nxGlobal);
   assert(weightHdrOn.nyGlobal == weightHdrOff.nyGlobal);
   assert(length(weightDataOn) == length(weightDataOff));
   assert(length(weightDataOn) >= 2);
end
assert(length(activityData) >= 2);

activityWriteStep = activityData{2}.time - activityData{1}.time;
if(SPIKING_POST_FLAG)
   assert(activityWriteStep == 1, 'Post layer write step must be 1 with spiking post flag on');
end
weightWriteStep = weightDataOn{2}.time - weightDataOn{1}.time;

%Each weight must have an activity associated with it
assert(weightWriteStep >= activityWriteStep);
assert(mod(weightWriteStep, activityWriteStep) == 0);

weightToActivity = weightWriteStep/activityWriteStep %The factor of activity step to write step
numWeightSteps = length(weightDataOn)

%Number of arbors
numArbors = weightHdrOn.nbands
%Number of features
numFeatures = weightHdrOn.nfp

%Size of postweight patch size
patchSizeX = weightHdrOn.nxp
patchSizeY = weightHdrOn.nyp

%Number of processes
procsX = weightHdrOn.nxprocs 
procsY = weightHdrOn.nyprocs

%Size of post layer
global columnSizeX; columnSizeX = weightHdrOn.nxGlobal 
global columnSizeY; columnSizeY = weightHdrOn.nyGlobal 

%Size of post layer process sector
global sizeX; sizeX = weightHdrOn.nx
global sizeY; sizeY = weightHdrOn.ny

%Margin for post layer to avoid margins
marginX = floor(patchSizeX/2) * postNxScale
marginY = floor(patchSizeY/2) * postNyScale

%Create list of indicies in mask that are valid points
mask = zeros(columnSizeY, columnSizeX);
mask(1 + marginY:columnSizeY - marginY, 1+marginX:columnSizeX-marginX) = 1;
%Based on vectorized mask
%Row first index
global marginIndex; marginIndex = find(mask'(:));

if (WEIGHTS_CELL_FLAG)
   for cellIndex = 1:length(CELL)
      assert(CELL{cellIndex}(1) <= columnSizeX - marginX);
      assert(CELL{cellIndex}(1) > marginX);
      assert(CELL{cellIndex}(2) <= columnSizeY - marginY);
      assert(CELL{cellIndex}(2) > marginY);
   end
end

disp('stdpAnalysis: Creating Images');
fflush(1);

%Pull all activity time by index
aTimeI = [[activityData{:}].time];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Coorelation Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%if (COOR_FLAG > 0)
%   disp("Running coord")
%   coorFunc(activityData);
%end

for weightTimeIndex = 1:numWeightSteps; %For every weight timestep
   time = weightDataOn{weightTimeIndex}.time; 
   disp(['Time: ', num2str(time)]);
   %Calculate weight time index to activity
   activityTimeIndex = find(aTimeI == time);
   assert(length(activityTimeIndex) <= 1);
   %Time is not found
   if (length(activityTimeIndex) == 0)
      continue;
   end
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%Weight Histogram
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if (WEIGHT_HIST_FLAG > 0)
      weightHist(weightDataOn{weightTimeIndex}.values, time, NUM_BINS, onWeightHistOutDir, 'On_Weight_Hist');
      weightHist(weightDataOff{weightTimeIndex}.values, time, NUM_BINS, offWeightHistOutDir, 'Off_Weight_Hist');
   end

   for i = 1:numArbors      %Iterate through number of arbors 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Image reconstruction
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (RECONSTRUCTION_FLAG > 0)
         outMat = reconstruct(activityData{activityTimeIndex}.values,  weightDataOn{weightTimeIndex}.values, weightDataOff{weightTimeIndex}.values, i);
         printImage(outMat, time, i, reconstructOutDir, RECON_IMAGE_SC, 'Reconstruction');
      end %End reconstruction

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Weight Map
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (WEIGHTS_MAP_FLAG > 0)
         outMat = weightMap(weightDataOn{weightTimeIndex}.values, weightDataOff{weightTimeIndex}.values, i);
         printImage(outMat, time, i, weightMapOutDir, WEIGHTS_IMAGE_SC, 'Weight_Map');
      end
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% Weight Cell
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (WEIGHTS_CELL_FLAG > 0)
         for cellIndex = 1:length(CELL)
            outMat = cellMap(weightDataOn{weightTimeIndex}.values, weightDataOff{weightTimeIndex}.values, i, CELL{cellIndex});
            printImage(outMat, time, i, cellMapOutDir, WEIGHTS_IMAGE_SC, ['Cell_Map_',num2str(CELL{cellIndex}(1)),'_',num2str(CELL{cellIndex}(2))]) 
         end
      end
   end
end

