clear all; close all; more off; clc;
system("clear");
%Reconstruct Flags
global SPIKING_OUT_FLAG;       SPIKING_OUT_FLAG       = 0;  %Create spiking output flag
global RECONSTRUCTION_FLAG;    RECONSTRUCTION_FLAG    = 0;  %Create reconstructions
global POST_WEIGHTS_MAP_FLAG;  POST_WEIGHTS_MAP_FLAG  = 1;     %Create weight maps
global POST_WEIGHTS_CELL_FLAG; POST_WEIGHTS_CELL_FLAG = 0;
global PRE_WEIGHTS_MAP_FLAG;   PRE_WEIGHTS_MAP_FLAG   = 0;     %Create weight maps
global PRE_WEIGHTS_CELL_FLAG;  PRE_WEIGHTS_CELL_FLAG  = 0;
global CELL; CELL = {...
   [35, 20]...
   [10, 20]...
};        %X by Y of weigh cell
global POST_WEIGHT_HIST_FLAG;  POST_WEIGHT_HIST_FLAG  = 1;
global PRE_WEIGHT_HIST_FLAG;   PRE_WEIGHT_HIST_FLAG   = 0;
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
global activityfile; activityfile          = [workspaceDir,'/output/lif.pvp'];
ONpreweightfile                            = [workspaceDir,'/output/w5.pvp'];
OFFpreweightfile                           = [workspaceDir,'/output/w6.pvp'];
ONpostweightfile                           = [workspaceDir,'/output/w5_post.pvp'];
OFFpostweightfile                          = [workspaceDir,'/output/w6_post.pvp'];
outputDir                                  = [workspaceDir,'/output/'];
global readPvpOutDir; readPvpOutDir        = [outputDir, 'pvp/'];
reconstructOutDir                          = [outputDir, 'reconstruct/'];
preWeightMapOutDir                         = [outputDir, 'pre_weight_map/'];
preCellMapOutDir                           = [outputDir, 'pre_cell_map/'];
postWeightMapOutDir                        = [outputDir, 'post_weight_map/'];
postCellMapOutDir                          = [outputDir, 'post_cell_map/'];
preOnWeightHistOutDir                      = [outputDir, 'pre_on_hist/'];
postOnWeightHistOutDir                     = [outputDir, 'post_on_hist/'];
preOffWeightHistOutDir                     = [outputDir, 'pre_off_hist/'];
postOffWeightHistOutDir                    = [outputDir, 'post_off_hist/'];
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
if (exist(preWeightMapOutDir, 'dir') ~= 7)
   mkdir(preWeightMapOutDir);
end
if (exist(preCellMapOutDir, 'dir') ~= 7)
   mkdir(preCellMapOutDir);
end
if (exist(postWeightMapOutDir, 'dir') ~= 7)
   mkdir(postWeightMapOutDir);
end
if (exist(postCellMapOutDir, 'dir') ~= 7)
   mkdir(postCellMapOutDir);
end
if (exist(preOnWeightHistOutDir, 'dir') ~= 7)
   mkdir(preOnWeightHistOutDir);
end
if (exist(postOnWeightHistOutDir, 'dir') ~= 7)
   mkdir(postOnWeightHistOutDir);
end
if (exist(preOffWeightHistOutDir, 'dir') ~= 7)
   mkdir(preOffWeightHistOutDir);
end
if (exist(postOffWeightHistOutDir, 'dir') ~= 7)
   mkdir(postOffWeightHistOutDir);
end

disp('stdpAnalysis: Reading activity pvp');
fflush(1);
%Read activity file in parallel
args{1} = activityfile;
args{2}= ONpostweightfile;
args{3} = OFFpostweightfile;
if (PRE_WEIGHTS_CELL_FLAG || PRE_WEIGHTS_MAP_FLAG)
   args{4} = ONpreweightfile;
   args{5} = OFFpreweightfile;
end

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
postWeightDataOn = data{2};
postWeightDataOff = data{3};
if (PRE_WEIGHTS_CELL_FLAG || PRE_WEIGHTS_MAP_FLAG || PRE_WEIGHT_HIST_FLAG)
   preWeightDataOn = data{4};
   preWeightDataOff = data{5};
else
   preWeightDataOn = postWeightDataOn;
   preWeightDataOff = postWeightDataOff;
end

activityHdr = hdr{1};
postWeightHdrOn = hdr{2};
postWeightHdrOff = hdr{3};
if (PRE_WEIGHTS_CELL_FLAG || PRE_WEIGHTS_MAP_FLAG || PRE_WEIGHT_HIST_FLAG)
   preWeightHdrOn = hdr{4};
   preWeightHdrOff = hdr{5};
else
   preWeightHdrOn = postWeightHdrOn;
   preWeightHdrOff = postWeightHdrOff;
end

%Output spiking
%readspikingpvp;

%PetaVision params
%TODO Replace with header values

assert((preWeightHdrOn.nbands == preWeightHdrOff.nbands) && (preWeightHdrOff.nbands == postWeightHdrOn.nbands) && (postWeightHdrOn.nbands == postWeightHdrOff.nbands));
assert((preWeightHdrOn.nfp == preWeightHdrOff.nfp) && (preWeightHdrOff.nfp == postWeightHdrOn.nfp) && (postWeightHdrOn.nfp == postWeightHdrOff.nfp));
assert((preWeightHdrOn.nxp == preWeightHdrOff.nxp) && (preWeightHdrOff.nxp == postWeightHdrOn.nxp) && (postWeightHdrOn.nxp == postWeightHdrOff.nxp));
assert((preWeightHdrOn.nyp == preWeightHdrOff.nyp) && (preWeightHdrOff.nyp == postWeightHdrOn.nyp) && (postWeightHdrOn.nyp == postWeightHdrOff.nyp));
assert((preWeightHdrOn.nxprocs == preWeightHdrOff.nxprocs) && (preWeightHdrOff.nxprocs == postWeightHdrOn.nxprocs) && (postWeightHdrOn.nxprocs == postWeightHdrOff.nxprocs));
assert((preWeightHdrOn.nyprocs == preWeightHdrOff.nyprocs) && (preWeightHdrOff.nyprocs == postWeightHdrOn.nyprocs) && (postWeightHdrOn.nyprocs == postWeightHdrOff.nyprocs));
assert((preWeightHdrOn.nx == preWeightHdrOff.nx) && (preWeightHdrOff.nx == postWeightHdrOn.nx) && (postWeightHdrOn.nx == postWeightHdrOff.nx));
assert((preWeightHdrOn.ny == preWeightHdrOff.ny) && (preWeightHdrOff.ny == postWeightHdrOn.ny) && (postWeightHdrOn.ny == postWeightHdrOff.ny));
assert((preWeightHdrOn.nxGlobal == preWeightHdrOff.nxGlobal) && (preWeightHdrOff.nxGlobal == postWeightHdrOn.nxGlobal) && (postWeightHdrOn.nxGlobal == postWeightHdrOff.nxGlobal));
assert((preWeightHdrOn.nyGlobal == preWeightHdrOff.nyGlobal) && (preWeightHdrOff.nyGlobal == postWeightHdrOn.nyGlobal) && (postWeightHdrOn.nyGlobal == postWeightHdrOff.nyGlobal));
assert((length(preWeightDataOn) == length(preWeightDataOff)) && (length(preWeightDataOff) == length(postWeightDataOn)) && (length(postWeightDataOn) == length(postWeightDataOff)));

%global numsteps; numsteps = activityHdr.nbands
global numWeightSteps; numWeightSteps = length(postWeightDataOn)
global columnSizeX; columnSizeX = postWeightHdrOn.nxGlobal 
global columnSizeY; columnSizeY = postWeightHdrOn.nyGlobal 
global sizeX; sizeX = postWeightHdrOn.nx
global sizeY; sizeY = postWeightHdrOn.ny
global postNxScale; postNxScale = 1 
global postNyScale; postNyScale = 1
global numArbors; numArbors = postWeightHdrOn.nbands
global numFeatures; numFeatures = postWeightHdrOn.nfp
global patchSizeX; patchSizeX = postWeightHdrOn.nxp
global patchSizeY; patchSizeY = postWeightHdrOn.nyp
global procsX; procsX = postWeightHdrOn.nxprocs 
global procsY; procsY = postWeightHdrOn.nyprocs

%Margin for post layer to avoid margins
global marginX; marginX = floor(patchSizeX/2) * postNxScale
global marginY; marginY = floor(patchSizeY/2) * postNyScale

%Create list of indicies in mask that are valid points
mask = zeros(columnSizeY, columnSizeX);
mask(1 + marginY:columnSizeY - marginY, 1+marginX:columnSizeX-marginX) = 1;
%Based on X, Y coords
%global marginIndexY marginIndexX;
%[marginIndexY marginIndexX] = find(mask);
global marginIndex; marginIndex = find(mask'(:));

if (POST_WEIGHTS_CELL_FLAG || PRE_WEIGHTS_CELL_FLAG)
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
   %Index based on X, Y coords
%   [activityIndexY activityIndexX] = find(activityData{activityTimeIndex});
   %Index based on one dimension, same index as activityIndexY and activityIndexX
   %TODO Use sub2ind instead of this
   %activityIndex = find(activityData{activityTimeIndex});
   %Convert to weight time index
%   weightTimeIndex = floor(activityTimeIndex / writeStep); 
   activityTimeIndex = postWeightDataOn{weightTimeIndex}.time + 1;
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%Weight Histogram
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if (POST_WEIGHT_HIST_FLAG > 0)
      weightHist(postWeightDataOn{weightTimeIndex}.values, activityTimeIndex, NUM_BINS, postOnWeightHistOutDir, 'Post_On_Weight_Hist');
      weightHist(postWeightDataOff{weightTimeIndex}.values, activityTimeIndex, NUM_BINS, postOffWeightHistOutDir, 'Post_Off_Weight_Hist');
   end
   if (PRE_WEIGHT_HIST_FLAG > 0)
      weightHist(preWeightDataOn{weightTimeIndex}.values, activityTimeIndex, NUM_BINS, preOnWeightHistOutDir, 'Pre_On_Weight_Hist');
      weightHist(preWeightDataOff{weightTimeIndex}.values, activityTimeIndex, NUM_BINS, preOffWeightHistOutDir, 'Pre_Off_Weight_Hist');
   end

   for i = 1:numArbors      %Iterate through number of arbors 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Image reconstruction
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (RECONSTRUCTION_FLAG > 0)
         if(activityTimeIndex > i)
            %To do
            outMat = reconstruct(activityData{activityTimeIndex - i}.values,  postWeightDataOn{weightTimeIndex}.values, postWeightDataOff{weightTimeIndex}.values, i);
            printImage(outMat, activityTimeIndex, i, reconstructOutDir, RECON_IMAGE_SC, 'Reconstruction');
         end
      end %End reconstruction

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Weight Map
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (PRE_WEIGHTS_MAP_FLAG > 0)
         outMat = weightMap(preWeightDataOn{weightTimeIndex}.values, preWeightDataOff{weightTimeIndex}.values, i);
         printImage(outMat, activityTimeIndex, i, preWeightMapOutDir, WEIGHTS_IMAGE_SC, 'Pre_Weight_Map');
      end
      if (POST_WEIGHTS_MAP_FLAG > 0)
         outMat = weightMap(postWeightDataOn{weightTimeIndex}.values, postWeightDataOff{weightTimeIndex}.values, i);
         printImage(outMat, activityTimeIndex, i, postWeightMapOutDir, WEIGHTS_IMAGE_SC, 'Post_Weight_Map');
      end
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% Weight Cell
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (PRE_WEIGHTS_CELL_FLAG > 0)
         for cellIndex = 1:length(CELL)
            outMat = cellMap(preWeightDataOn{weightTimeIndex}.values, preWeightDataOff{weightTimeIndex}.values, i, CELL{cellIndex});
            printImage(outMat, activityTimeIndex, i, preCellMapOutDir, WEIGHTS_IMAGE_SC, ['Pre_Cell_Map_',num2str(CELL{cellIndex}(1)),'_',num2str(CELL{cellIndex}(2))]) 
         end
      end
      if (POST_WEIGHTS_CELL_FLAG > 0)
         for cellIndex = 1:length(CELL)
            outMat = cellMap(postWeightDataOn{weightTimeIndex}.values, postWeightDataOff{weightTimeIndex}.values, i, CELL{cellIndex});
            printImage(outMat, activityTimeIndex, i, postCellMapOutDir, WEIGHTS_IMAGE_SC, ['Post_Cell_Map_',num2str(CELL{cellIndex}(1)),'_',num2str(CELL{cellIndex}(2))]) 
         end
      end
   end
end
