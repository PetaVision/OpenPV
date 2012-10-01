clear all; close all; more off; clc;
system("clear");

%Reconstruct Flags
global RECONSTRUCTION_FLAG = 1;  %Create reconstructions
global WEIGHTS_MAP_FLAG = 1;     %Create weight maps
global WEIGHT_CELL_FLAG = 1;
global CELL = {[35, 20]...
        [10, 20]...
       };        %X by Y of weigh cell
global VIEW_FIGS = 0;
global WRITE_FIGS = 1;
global GRAY_SC = 0;              %Image in grayscale
%Difference between On/Off if 0, seperate otherwise
global ON_OFF_SEP_REC = 0;
global ON_OFF_SEP_WM = 0;
global ON_OFF_SEP_CW = 0;          
global RECON_IMAGE_SC = -1;        %-1 for autoscale
global WEIGHT_IMAGE_SC = -1; %-1 for autoscale
global GRID_FLAG = 0;

%File names
rootDir = '/Users/slundquist';
workspaceDir = [rootDir,'/Documents/workspace/iHouse'];
%rootDir = '/Users/dpaiton';
%workspaceDir = [rootDir,'/Documents/Work/LANL/workspace/iHouse'];
activityfile = [workspaceDir,'/output/lif.pvp'];
ONpostweightfile = [workspaceDir,'/output/w5_post.pvp'];
OFFpostweightfile = [workspaceDir,'/output/w6_post.pvp'];
outputDir = [workspaceDir,'/output/'];
readPvpOutDir = [outputDir, 'pvp/'];
reconstructOutDir = [outputDir, 'reconstruct/'];
weightMapOutDir = [outputDir, 'weight_map/'];
cellMapOutDir = [outputDir, 'cell_map/'];
sourcefile = [workspaceDir,'/output/DropInput.txt'];

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

%Params for readpvpfile
global MOVIE_FLAG = 0;
global SWEEP_POS = 0;
global PRINT_FLAG = 0;
post = 1;

display('Reconstruct: Reading activity pvp');
fflush(1);
%Read activity file
[activityData activityHdr] = readpvpfile(activityfile);

%Read weight matricies for on/off ret
display('Reconstruct: Reading ON weights pvp')
fflush(1);
[weightDataOn weightHdrOn] = readpvpfile(ONpostweightfile);
display('Reconstruct: Reading OFF weights pvp')
fflush(1);
[weightDataOff weightHdrOff] = readpvpfile(OFFpostweightfile);

%Output spiking
%readspikingpvp;

%PetaVision params
%TODO Replace with header values
global writeStep = 200    %Write Step of connection

assert(weightHdrOn.nbands == weightHdrOff.nbands);
assert(weightHdrOn.nfp == weightHdrOff.nfp);
assert(weightHdrOn.nxp == weightHdrOff.nxp);
assert(weightHdrOn.nyp == weightHdrOff.nyp);
assert(weightHdrOn.nxprocs == weightHdrOff.nxprocs);
assert(weightHdrOn.nyprocs == weightHdrOff.nyprocs);
assert(weightHdrOn.nx == weightHdrOff.nx);
assert(weightHdrOn.ny == weightHdrOff.ny);
assert(weightHdrOn.nxGlobal == weightHdrOff.nxGlobal);
assert(weightHdrOn.nyGlobal == weightHdrOff.nyGlobal);

global numsteps = activityHdr.nbands
global columnSizeX = weightHdrOn.nxGlobal 
global columnSizeY = weightHdrOn.nyGlobal 
global sizeX = weightHdrOn.nx
global sizeY = weightHdrOn.ny
global postNxScale = 1 
global postNyScale = 1
global numArbors = weightHdrOn.nbands
global numFeatures = weightHdrOn.nfp
global patchSizeX = weightHdrOn.nxp
global patchSizeY = weightHdrOn.nyp
global procsX = weightHdrOn.nxprocs 
global procsY = weightHdrOn.nyprocs

%Margin for post layer to avoid margins
global marginX = floor(patchSizeX/2) * postNxScale;
global marginY = floor(patchSizeY/2) * postNyScale;

%Create list of indicies in mask that are valid points
mask = zeros(columnSizeY, columnSizeX);
mask(1 + marginY:columnSizeY - marginY, 1+marginX:columnSizeX-marginX) = 1;
%Based on X, Y coords
%global marginIndexY marginIndexX;
%[marginIndexY marginIndexX] = find(mask);
global marginIndex = find(mask'(:));

assert(length(weightDataOn) == length(weightDataOff));
for cellIndex = 1:length(CELL)
   assert(CELL{cellIndex}(1) <= columnSizeY - marginY);
   assert(CELL{cellIndex}(1) > 0 + marginY);
   assert(CELL{cellIndex}(2) <= columnSizeX - marginX);
   assert(CELL{cellIndex}(2) > +marginX);
end

display('Reconstruct: Creating Images');
fflush(1);
for activityTimeIndex = writeStep:writeStep:numsteps    %For every timestep
   %Index based on X, Y coords
%   [activityIndexY activityIndexX] = find(activityData{activityTimeIndex});
   %Index based on one dimension, same index as activityIndexY and activityIndexX
   %TODO Use sub2ind instead of this
   %activityIndex = find(activityData{activityTimeIndex});
   %Convert to weight time index
   weightTimeIndex = floor(activityTimeIndex / writeStep); 
   for i = 1:numArbors      %Iterate through number of arbors 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Image reconstruction
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (RECONSTRUCTION_FLAG > 0)
         if(activityTimeIndex > numArbors)
            outMat = reconstruct(activityData{activityTimeIndex - i}.values,  weightDataOn{weightTimeIndex}.values, weightDataOff{weightTimeIndex}.values, i);
            printImage(outMat, activityTimeIndex, i, reconstructOutDir, RECON_IMAGE_SC, 'Reconstruction');
         end
      end %End reconstruction

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Weight Map
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (WEIGHTS_MAP_FLAG > 0)
         outMat = weightMap(weightDataOn{weightTimeIndex}.values, weightDataOff{weightTimeIndex}.values, i);
         printImage(outMat, activityTimeIndex, i, weightMapOutDir, WEIGHT_IMAGE_SC, 'Weight_Map');
      end
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% Weight Cell
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (WEIGHT_CELL_FLAG > 0)
         for cellIndex = 1:length(CELL)
            outMat = cellMap(weightDataOn{weightTimeIndex}.values, weightDataOff{weightTimeIndex}.values, i, CELL{cellIndex});
            printImage(outMat, activityTimeIndex, i, cellMapOutDir, WEIGHT_IMAGE_SC, ['Cell_Map_',num2str(CELL{cellIndex}(1)),'_',num2str(CELL{cellIndex}(2))]) 
         end
      end
   end
end
