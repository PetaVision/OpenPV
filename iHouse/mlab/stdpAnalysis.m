clear all; close all; more off; clc;
system("clear");

%Nessessary Petavision Variables 
global postNxScale; postNxScale = 4; 
global postNyScale; postNyScale = 4;

%Reconstruct Flags
global SPIKING_OUT_FLAG;       SPIKING_OUT_FLAG       = 0;  %Create spiking output flag
global RECONSTRUCTION_FLAG;    RECONSTRUCTION_FLAG    = 1;  %Create reconstructions
global WEIGHTS_MAP_FLAG;  WEIGHTS_MAP_FLAG  = 1;     %Create weight maps
global WEIGHTS_CELL_FLAG; WEIGHTS_CELL_FLAG = 1;
global CELL; CELL = {...
   [35, 20]...
   [10, 20]...
};        %X by Y of weigh cell
global WEIGHT_HIST_FLAG;  WEIGHT_HIST_FLAG  = 1;
global NUM_BINS;               NUM_BINS               = 20;

global VIEW_FIGS;  VIEW_FIGS  = 0;
global WRITE_FIGS; WRITE_FIGS = 1;
global GRAY_SC;    GRAY_SC    = 0;              %Image in grayscale

%Difference between On/Off if 0, seperate otherwise
global RECON_IMAGE_SC;   RECON_IMAGE_SC   = -1;        %-1 for autoscale
global WEIGHTS_IMAGE_SC; WEIGHTS_IMAGE_SC = -1; %-1 for autoscale
global GRID_FLAG;        GRID_FLAG        = 0;
global NUM_PROCS;        NUM_PROCS        = nproc();


%File names
rootDir                                    = '/Users/slundquist';
workspaceDir                               = [rootDir,'/Documents/workspace/iHouse'];
%rootDir                                    = '/Users/dpaiton';
%workspaceDir                               = [rootDir,'/Documents/Work/LANL/workspace/iHouse'];
pvpDir                                     = [workspaceDir,'/output/'];
global activityfile; activityfile          = [pvpDir,'lif.pvp'];
ONweightfile                               = [pvpDir,'w5_post.pvp'];
OFFweightfile                              = [pvpDir,'w6_post.pvp'];
outputDir                                  = [workspaceDir,'/output/'];
global readPvpOutDir; readPvpOutDir        = [outputDir, 'pvp/'];
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
if (exist(readPvpOutDir, 'dir') ~= 7)
   mkdir(readPvpOutDir);
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

fflush(1);
%Read activity file in parallel
args{1} = activityfile;
args{2}= ONweightfile;
args{3} = OFFweightfile;

disp('stdpAnalysis: Reading pvp files')
if (SPIKING_OUT_FLAG)
   readspikingpvp;
end

if NUM_PROCS == 1
   [data hdr] = cellfun(@readpvpfile, args, 'UniformOutput', 0);
else
   [data hdr] = parcellfun(NUM_PROCS, @readpvpfile, args, 'UniformOutput', 0);
end

activityData = data{1};
weightDataOn = data{2};
weightDataOff = data{3};

activityHdr = hdr{1};
weightHdrOn = hdr{2};
weightHdrOff = hdr{3};

%PetaVision params
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
assert(length(activityData) >= 2);

activityWriteStep = activityData{2}.time - activityData{1}.time;
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

for weightTimeIndex = 1:numWeightSteps %For every weight timestep
   time = weightDataOn{weightTimeIndex}.time; 
   disp(['Time: ', num2str(time)]);
   %Calculate weight time index to activity
   activityTimeIndex = weightTimeIndex * weightToActivity;
   assert(activityData{activityTimeIndex}.time == time);
   
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
