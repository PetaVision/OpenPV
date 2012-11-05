clear all; close all; more off; clc;
system("clear");

%Nessessary Petavision Variables 
global postNxScale; postNxScale = 4; 
global postNyScale; postNyScale = 4;
global lifInhPatchX; lifInhPatchX = 21;
global lifInhPatchY; lifInhPatchY = 21;
global deltaT; deltaT = 1;
global tLCA; tLCA = 300;

rootDir                                    = '/Users/slundquist/';
workspaceDir                               = [rootDir,'Documents/workspace/iHouse'];
pvpDir                                     = [workspaceDir,'/testOutput/'];
outputDir                                  = [workspaceDir,'/testOutput/analysis/'];

%Spiking Output
global FNUM_ALL;  FNUM_ALL = 1;   %1 for all frames, 0 for FNUM_SPEC
global FNUM_SPEC; FNUM_SPEC= {... %start:int:end frames
   [0:9999]...
};

global WEIGHTS_MAP_FLAG;       WEIGHTS_MAP_FLAG    = 1; %Create weight maps

global WEIGHT_HIST_FLAG;  WEIGHT_HIST_FLAG  = 1;
global NUM_BINS;          NUM_BINS          = 20;

global VIEW_FIGS;  VIEW_FIGS  = 0;
global WRITE_FIGS; WRITE_FIGS = 1;
global GRAY_SC;    GRAY_SC    = 1;              %Image in grayscale

%Difference between On/Off if 0, seperate otherwise
global RECON_IMAGE_SC;   RECON_IMAGE_SC   = -1; %-1 for autoscale
global WEIGHTS_IMAGE_SC; WEIGHTS_IMAGE_SC = .5; %-1 for autoscale
global GRID_FLAG;        GRID_FLAG        = 0;
global NUM_PROCS;        NUM_PROCS        = nproc();


%File names
latInhFile                                 = [pvpDir,'w6_post.pvp'];
latInhWeightMapDir                         = [outputDir, 'latInh_weight_map/'];
latInhHistDir                              = [outputDir, 'latInh_hist/'];

%Make nessessary directories
if (exist(outputDir, 'dir') ~= 7)
   mkdir(outputDir);
end
if (exist(latInhWeightMapDir, 'dir') ~= 7)
   mkdir(latInhWeightMapDir);
end
if (exist(latInhHistDir, 'dir') ~= 7)
   mkdir(latInhHistDir);
end

disp('latAnalysis: Reading pvp files')
fflush(1);
[inhData, inhHdr] = readpvpfile(latInhFile, latInhWeightMapDir, 0);

numSteps = length(inhData)

%Number of arbors
numArbors = inhHdr.nbands
%Number of features
numFeatures = inhHdr.nfp

%Size of postweight patch size
patchSizeX = inhHdr.nxp
patchSizeY = inhHdr.nyp

%Number of processes
procsX = inhHdr.nxprocs 
procsY = inhHdr.nyprocs

%Size of post layer
global columnSizeX; columnSizeX = inhHdr.nxGlobal 
global columnSizeY; columnSizeY = inhHdr.nyGlobal 

%Size of post layer process sector
global sizeX; sizeX = inhHdr.nx
global sizeY; sizeY = inhHdr.ny

%Margin for post layer to avoid margins
marginX = floor(patchSizeX/2)
marginY = floor(patchSizeY/2)

%Create list of indicies in mask that are valid points
%mask = zeros(columnSizeY, columnSizeX);
mask = ones(columnSizeY, columnSizeX);
mask(1 + marginY:columnSizeY - marginY, 1+marginX:columnSizeX-marginX) = 1;
%Based on vectorized mask
%Row first index
global marginIndex; marginIndex = find(mask'(:));

disp('stdpAnalysis: Creating Images');
fflush(1);

for timeIndex = 1:numSteps; %For every weight timestep
   time = inhData{timeIndex}.time; 
   disp(['Time: ', num2str(time)]);
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%Weight Histogram
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if (WEIGHT_HIST_FLAG > 0)
      weightHist(inhData{timeIndex}.values, time, NUM_BINS, latInhHistDir, 'Inh_Weight_Hist');
   end

   for arborId = 1:numArbors      %Iterate through number of arbors 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Weight Map
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (WEIGHTS_MAP_FLAG > 0)
         wm_outMat = zeros(columnSizeY * patchSizeY, columnSizeX * patchSizeX);
         for margini = 1:length(marginIndex)
            [mIx mIy] = ind2sub([columnSizeX columnSizeY], marginIndex(margini)); 
            procXi = floor((mIx - 1)/sizeX) + 1;
            procYi = floor((mIy - 1)/sizeY) + 1;
            startIndexX = (mIx - 1) * patchSizeX + 1; 
            startIndexY = (mIy - 1) * patchSizeY + 1; 
            endIndexX = startIndexX + patchSizeX - 1;
            endIndexY = startIndexY + patchSizeY - 1;
            %get index based on what quaderant
            newIndX = mod((mIx - 1), sizeX) + 1;
            newIndY = mod((mIy - 1), sizeY) + 1;
            newInd = sub2ind([sizeX sizeY], newIndX, newIndY);
            for nfi = 1:numFeatures  %For multiple features
               %Set out
               wm_outMat(startIndexY:endIndexY, startIndexX:endIndexX) =...
               inhData{timeIndex}.values{procXi, procYi, arborId}(:, :, nfi, newInd)';
            end
         end
         latPrintImage(wm_outMat, time, arborId, latInhWeightMapDir, WEIGHTS_IMAGE_SC, 'Inh_Weight_Map');
      end
   end
end

