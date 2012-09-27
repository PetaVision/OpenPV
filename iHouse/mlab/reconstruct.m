clear all; close all; more off; clc;
system("clear");

%Reconstruct Flags
RECONSTRUCTION_FLAG = 0;  %Create reconstructions
WEIGHTS_MAP_FLAG = 1;     %Create weight maps
WEIGHT_CELL_FLAG = 1;
CELL = {[35, 20]...
        [10, 20]...
       };        %Y by X of weigh cell
VIEW_FIGS = 0;
WRITE_FIGS = 1;
GRAY_SC = 0;              %Image in grayscale
%Difference between On/Off if 0, seperate otherwise
ON_OFF_SEP_REC = 0;
ON_OFF_SEP_WM = 0;
ON_OFF_SEP_CW = 0;          
WEIGHT_IMAGE_SC = [-.3 .3]; %Scale for imagesc
GRID_FLAG = 0;

%File names
activityfile = '/Users/slundquist/Documents/workspace/iHouse/output/lif.pvp';
ONpostweightfile = '/Users/slundquist/Documents/workspace/iHouse/output/w5_post.pvp';
OFFpostweightfile = '/Users/slundquist/Documents/workspace/iHouse/output/w6_post.pvp';
ONfilename = 'ON_post.pvp';
OFFfilename = 'OFF_post.pvp';
outputDir = '/Users/slundquist/Documents/workspace/iHouse/output/';
readPvpOutDir = [outputDir, 'pvp/'];
reconstructOutDir = [outputDir, 'reconstruct/'];
weightMapOutDir = [outputDir, 'weight_map/'];
cellMapOutDir = [outputDir, 'cell_map/'];
sourcefile = '/Users/slundquist/Documents/workspace/iHouse/output/DropInput.txt';

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
[activityData activityHdr] = readpvpfile(activityfile, readPvpOutDir, 'lif');

%Read weight matricies for on/off ret
display('Reconstruct: Reading ON weights pvp')
fflush(1);
[weightDataOn weightHdrOn] = readpvpfile(ONpostweightfile, readPvpOutDir, ONfilename, post);
display('Reconstruct: Reading OFF weights pvp')
fflush(1);
[weightDataOff weightHdrOff] = readpvpfile(OFFpostweightfile, readPvpOutDir, OFFfilename, post);

%Output spiking
%readspikingpvp;

%PetaVision params
%TODO Replace with header values
writeStep = 200    %Write Step of connection

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

numsteps = activityHdr.nbands
columnSizeX = weightHdrOn.nx 
columnSizeY = weightHdrOn.ny 
postNxScale = columnSizeX/weightHdrOn.nxGlobal 
postNyScale = columnSizeY/weightHdrOn.nyGlobal 
numArbors = weightHdrOn.nbands
numFeatures = weightHdrOn.nfp
patchSizeX = weightHdrOn.nxp
patchSizeY = weightHdrOn.nyp
procsX = weightHdrOn.nxprocs 
procsY = weightHdrOn.nyprocs

%Margin for post layer to avoid margins
marginX = floor(patchSizeX/2) * postNxScale;
marginY = floor(patchSizeY/2) * postNyScale;

%Create list of indicies in mask that are valid points
mask = zeros(columnSizeY, columnSizeX);
mask(1 + marginY:columnSizeY - marginY, 1+marginX:columnSizeX-marginX) = 1;
%Based on X, Y coords
[marginIndexY marginIndexX] = find(mask);
%TODO Use sub2ind instead of this
%Based on vectorized matrix
marginIndex = find(mask'(:)');

display('Reconstruct: Creating Images');
fflush(1);
for activityTimeIndex = writeStep:writeStep:numsteps    %For every timestep
   %Index based on X, Y coords
   [activityIndexY activityIndexX] = find(activityData{activityTimeIndex});
   %Index based on one dimension, same index as activityIndexY and activityIndexX
   %TODO Use sub2ind instead of this
   activityIndex = find(activityData{activityTimeIndex});
   %Convert to weight time index
   weightTimeIndex = floor(activityTimeIndex / writeStep); 
   for i = 1:numArbors      %Iterate through number of arbors 
      arborIndex = sub2ind([procsY procsX numArbors], procsY, procsX, i);
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Image reconstruction
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (RECONSTRUCTION_FLAG > 0)
         %Output baseline is zero 
         if(ON_OFF_SEP_REC)
            onOutMat = zeros(columnSizeY, columnSizeX);
            offOutMat = zeros(columnSizeY, columnSizeX);
         else
            outMat = zeros(columnSizeY, columnSizeX);
         end

         for j = 1:length(activityIndex)   %Iterate through spiking activity
            %If the spiking activity is not in the allowed area
            if isempty(find(marginIndex == activityIndex(j)))
               continue;   %Skip
            else
               %Set weight matrix to outmat to the square of the weight matrix
               for k = 1:numFeatures  %For multiple features
                  if(ON_OFF_SEP_REC)
                     %Add on weights based on size of weight matrix
                     onOutMat(activityIndexY(j) - marginY: activityIndexY(j) + marginY,...
                     activityIndexX(j) - marginX: activityIndexX(j) + marginX) += ...
                     weightDataOn{weightTimeIndex}.values{arborIndex}(:, :, k, activityIndex(j));
                     %Subtract off weights
                     offOutMat(activityIndexY(j) - marginY: activityIndexY(j) + marginY,...
                     activityIndexX(j) - marginX: activityIndexX(j) + marginX) -= ...
                     weightDataOff{weightTimeIndex}.values{arborIndex}(:, :, k, activityIndex(j));
                  else
                     %Add on weights based on size of weight matrix
                     outMat(activityIndexY(j) - marginY: activityIndexY(j) + marginY,...
                     activityIndexX(j) - marginX: activityIndexX(j) + marginX) += ...
                     weightDataOn{weightTimeIndex}.values{arborIndex}(:, :, k, activityIndex(j));
                     %Subtract off weights
                     outMat(activityIndexY(j) - marginY: activityIndexY(j) + marginY,...
                     activityIndexX(j) - marginX: activityIndexX(j) + marginX) -= ...
                     weightDataOff{weightTimeIndex}.values{arborIndex}(:, :, k, activityIndex(j));
                  end
               end
            end
         end
         %Scale image and print
         %On and off
         if(ON_OFF_SEP_REC)
            %On reconstruction
            if(VIEW_FIGS)
               figure;
            else
               figure('Visible', 'off');
            end
            imagesc(onOutMat);
            if(GRAY_SC)
               colormap(gray);
            else
               colormap(cm());
            end
            colorbar;
            title(['On Reconstruction - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(arborIndex)]);
            if(WRITE_FIGS)
               print_movie_filename = [reconstructOutDir, 'on_reconstruct_', num2str(activityTimeIndex), '_', num2str(arborIndex), '.jpg'];
               print(print_movie_filename);
            end

            %Off reconstruction
            if(VIEW_FIGS)
               figure;
            else
               figure('Visible', 'off');
            end
            imagesc(offOutMat);
            if(GRAY_SC)
               colormap(gray);
            else
               colormap(cm());
            end
            colorbar;
            title(['Off Reconstruction - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(arborIndex)]);
            if(WRITE_FIGS)
               print_movie_filename = [reconstructOutDir, 'off_reconstruct_', num2str(activityTimeIndex), '_', num2str(arborIndex), '.jpg'];
               print(print_movie_filename);
            end
            %Combined
         else
            if(VIEW_FIGS)
               figure;
            else
               figure('Visible', 'off');
            end
            imagesc(outMat);
            if(GRAY_SC)
               colormap(gray);
            else
               colormap(cm());
            end
            colorbar;
            title(['Reconstruction - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(arborIndex)]);
            if(WRITE_FIGS)
               print_movie_filename = [reconstructOutDir, 'reconstruct_', num2str(activityTimeIndex), '_', num2str(arborIndex), '.jpg'];
               print(print_movie_filename);
            end
         end
      end %End reconstruction

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Weight Map
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (WEIGHTS_MAP_FLAG > 0)
         %Output baseline is zero 
         if(ON_OFF_SEP_WM)
            if(GRID_FLAG)
               onWeightsOutMat = zeros(columnSizeY * patchSizeY + columnSizeY, columnSizeX * patchSizeX + columnSizeX);
               offWeightsOutMat = zeros(columnSizeY * patchSizeY + columnSizeY, columnSizeX * patchSizeX + columnSizeX);
            else
               onWeightsOutMat = zeros(columnSizeY * patchSizeY, columnSizeX * patchSizeX);
               offWeightsOutMat = zeros(columnSizeY * patchSizeY, columnSizeX * patchSizeX);
            end
         else
            if(GRID_FLAG)
               weightsOutMat = zeros(columnSizeY * patchSizeY + columnSizeY, columnSizeX * patchSizeX + columnSizeX);
            else
               weightsOutMat = zeros(columnSizeY * patchSizeY, columnSizeX * patchSizeX);
            end
         end
         for j = 1:length(marginIndex)   %Iterate through allowed margin index
            %Convert index to weight out mat
            if(GRID_FLAG)
               %Added 1 pixel grid between patches
               startIndexX = (marginIndexX(j) - 1) * (patchSizeX + 1) + 1; 
               startIndexY = (marginIndexY(j) - 1) * (patchSizeY + 1) + 1; 
            else
               %Added 1 pixel grid between patches
               startIndexX = (marginIndexX(j) - 1) * patchSizeX + 1; 
               startIndexY = (marginIndexY(j) - 1) * patchSizeY + 1; 
            end

            endIndexX = startIndexX + patchSizeX - 1;
            endIndexY = startIndexY + patchSizeY - 1;

            for k = 1:numFeatures  %For multiple features
               if(ON_OFF_SEP_WM)
                  %On weights
                  onWeightsOutMat(startIndexY:endIndexY, startIndexX:endIndexX) +=...
                  weightDataOn{weightTimeIndex}.values{arborIndex}(:, :, k, marginIndex(j));
                  %Off weights
                  offWeightsOutMat(startIndexY:endIndexY, startIndexX:endIndexX) -=...
                  weightDataOff{weightTimeIndex}.values{arborIndex}(:, :, k, marginIndex(j));
               else
                  %On weights
                  weightsOutMat(startIndexY:endIndexY, startIndexX:endIndexX) +=...
                  weightDataOn{weightTimeIndex}.values{arborIndex}(:, :, k, marginIndex(j));
                  %Off weights
                  weightsOutMat(startIndexY:endIndexY, startIndexX:endIndexX) -=...
                  weightDataOff{weightTimeIndex}.values{arborIndex}(:, :, k, marginIndex(j));
               end
            end
         end
         %Scale image and print
         if(ON_OFF_SEP_WM)
            if(GRID_FLAG)
               %Make grid black
               onMarginVal = min(onWeightsOutMat(:));
               onWeightsOutMat(:, patchSizeX:patchSizeX:end) = onMarginVal;
               onWeightsOutMat(patchSizeY:patchSizeY:end, :) = onMarginVal;

               offMarginVal = min(offWeightsOutMat(:));
               offWeightsOutMat(:, patchSizeX:patchSizeX:end) = offMarginVal;
               offWeightsOutMat(patchSizeY:patchSizeY:end, :) = offMarginVal;
            end

            %On
            if(VIEW_FIGS)
               figure;
            else
               figure('Visible', 'off');
            end
            imagesc(onWeightsOutMat);
            if(GRAY_SC)
               colormap(gray);
            else
               colormap(cm());
            end
            colorbar;
            title(['On Weight Map - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(arborIndex)])
            if(WRITE_FIGS)
               print_movie_filename = [weightMapOutDir, 'on_weight_map_', num2str(activityTimeIndex), '_', num2str(arborIndex), '.jpg'];
               print(print_movie_filename);
            end

            %Off
            if(VIEW_FIGS)
               figure;
            else
               figure('Visible', 'off');
            end
            imagesc(offWeightsOutMat);
            if(GRAY_SC)
               colormap(gray);
            else
               colormap(cm());
            end
            colorbar;
            title(['Off Weight Map - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(arborIndex)])
            if(WRITE_FIGS)
               print_movie_filename = [weightMapOutDir, 'off_weight_map_', num2str(activityTimeIndex), '_', num2str(arborIndex), '.jpg'];
               print(print_movie_filename);
            end

         else
            if(GRID_FLAG)
               marginVal = min(weightsOutMat(:));
               weightsOutMat(:, patchSizeX:patchSizeX:end) = marginVal;
               weightsOutMat(patchSizeY:patchSizeY:end, :) = marginVal;
            end

            if(VIEW_FIGS)
               figure;
            else
               figure('Visible', 'off');
            end
            imagesc(weightsOutMat, WEIGHT_IMAGE_SC);
            if(GRAY_SC)
               colormap(gray);
            else
               colormap(cm());
            end
            colorbar;
            title(['Weight Map - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(arborIndex)])
            if(WRITE_FIGS)
               print_movie_filename = [weightMapOutDir, 'weight_map_', num2str(activityTimeIndex), '_', num2str(arborIndex), '.jpg'];
               print(print_movie_filename);
            end
         end
      end
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% Weight Cell
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (WEIGHT_CELL_FLAG > 0)
         for cellIndex = 1:length(CELL)
            assert(CELL{cellIndex}(1) <= columnSizeY - marginY);
            assert(CELL{cellIndex}(1) > 0 + marginY);
            assert(CELL{cellIndex}(2) <= columnSizeX - marginX);
            assert(CELL{cellIndex}(2) > +marginX);
         end
         for cellIndex = 1:length(CELL)
            if(ON_OFF_SEP_CW)
               onCellOutMat = zeros(patchSizeY, patchSizeX);
               offCellOutMat = zeros(patchSizeY, patchSizeX);
            else
               cellOutMat = zeros(patchSizeY, patchSizeX);
            end
            index = sub2ind([columnSizeY columnSizeX], CELL{cellIndex}(1), CELL{cellIndex}(2));
            for k = 1:numFeatures  %For multiple features
               if(ON_OFF_SEP_CW)
                  %On weights
                  onCellOutMat +=...
                  weightDataOn{weightTimeIndex}.values{arborIndex}(:, :, k, index);
                  %Off weights
                  offCellOutMat -=...
                  weightDataOff{weightTimeIndex}.values{arborIndex}(:, :, k, index);
               else
                  %On weights
                  cellOutMat +=...
                  weightDataOn{weightTimeIndex}.values{arborIndex}(:, :, k, index);
                  %Off weights
                  cellOutMat -=...
                  weightDataOff{weightTimeIndex}.values{arborIndex}(:, :, k, index);
               end
            end
            %Scale image and print
            if(ON_OFF_SEP_CW)
               %On
               if(VIEW_FIGS)
                  figure;
               else
                  figure('Visible', 'off');
               end
               imagesc(onCellOutMat);
               if(GRAY_SC)
                  colormap(gray);
               else
                  colormap(cm());
               end
               colorbar;
               title(['On Cell (', num2str(CELL{cellIndex}(1)), ', ', num2str(CELL{cellIndex}(2)), ') Map - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(arborIndex)])
               if(WRITE_FIGS)
                  print_movie_filename = [cellMapOutDir, 'on_cell_', num2str(CELL{cellIndex}(1)), '_', num2str(CELL{cellIndex}(2)),'_map_', num2str(activityTimeIndex), '_', num2str(arborIndex), '.jpg'];
                  print(print_movie_filename);
               end

               %Off
               if(VIEW_FIGS)
                  figure;
               else
                  figure('Visible', 'off');
               end
               imagesc(offCellOutMat);
               if(GRAY_SC)
                  colormap(gray);
               else
                  colormap(cm());
               end
               colorbar;
               title(['Off Cell (', num2str(CELL{cellIndex}(1)), ', ', num2str(CELL{cellIndex}(2)), ') Map - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(arborIndex)])
               if(WRITE_FIGS)
                  print_movie_filename = [cellMapOutDir, 'off_cell_', num2str(CELL{cellIndex}(1)), '_', num2str(CELL{cellIndex}(2)),'_map_', num2str(activityTimeIndex), '_', num2str(arborIndex), '.jpg'];
                  print(print_movie_filename);
               end

            else
               if(VIEW_FIGS)
                  figure;
               else
                  figure('Visible', 'off');
               end
               imagesc(cellOutMat, WEIGHT_IMAGE_SC);
               if(GRAY_SC)
                  colormap(gray);
               else
                  colormap(cm());
               end
               colorbar;
               title(['Cell (', num2str(CELL{cellIndex}(1)), ', ', num2str(CELL{cellIndex}(2)), ') Map - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(arborIndex)])
               if(WRITE_FIGS)
                  print_movie_filename = [cellMapOutDir, 'cell_', num2str(CELL{cellIndex}(1)), '_', num2str(CELL{cellIndex}(2)),'_map_', num2str(activityTimeIndex), '_', num2str(arborIndex), '.jpg'];
                  print(print_movie_filename);
               end
            end
         end
      end
   end
end
