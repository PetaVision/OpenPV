clear all; close all; more off; clc;
system("clear");

RECONSTRUCTION_FLAG = 1;
WEIGHTS_MAP_FLAG = 1;

activityfile = '/Users/slundquist/Documents/workspace/iHouse/output/lif.pvp';
ONpostweightfile = '/Users/slundquist/Documents/workspace/iHouse/output/w5_post.pvp';
OFFpostweightfile = '/Users/slundquist/Documents/workspace/iHouse/output/w6_post.pvp';
ONfilename = 'ON_post.pvp';
OFFfilename = 'OFF_post.pvp';
outputdir = '/Users/slundquist/Documents/workspace/iHouse/output/reconstruct';
sourcefile = '/Users/slundquist/Documents/workspace/iHouse/output/DropInput.txt';

numsteps = 500;
writeStep = 100;
numArbors = 1;
numFeatures = 1;
patchSizeX = 5;
patchSizeY = 5;
postNxScale = 1;
postNyScale = 1;
columnSizeX = 128;
columnSizeY = 128;

%Params for readpvpfile
global MOVIE_FLAG = 0;
global SWEEP_POS = 0;
global PRINT_FLAG = 0;
post = 1;

%Margin for post layer to avoid margins
marginX = floor(patchSizeX/2) * postNxScale;
marginY = floor(patchSizeY/2) * postNyScale;

display('Reconstruct: Reading activity pvp');
fflush(1);
%Read activity file
[activityData activityHdr] = readpvpfile(activityfile, outputdir, 'lif');

%Read weight matricies for on/off ret
display('Reconstruct: Reading ON weights pvp')
fflush(1);
[weightDataOn weightHdrOn] = readpvpfile(ONpostweightfile, outputdir, ONfilename, post);
display('Reconstruct: Reading OFF weights pvp')
fflush(1);
[weightDataOff weightHdrOff] = readpvpfile(OFFpostweightfile, outputdir, OFFfilename, post);

%Create list of indicies in mask that are valid points
mask = zeros(columnSizeY, columnSizeX);
mask(1 + marginY:columnSizeY - marginY, 1+marginX:columnSizeX-marginX) = 1;
%Based on X, Y coords
[marginIndexY marginIndexX] = find(mask);
%Based on vectorized matrix
marginIndex = find(mask'(:)');

display('Reconstruct: Creating Images');
fflush(1);
for activityTimeIndex = writeStep:writeStep:numsteps    %For every timestep
   
   %Index based on X, Y coords
   [activityIndexY activityIndexX] = find(activityData{activityTimeIndex});
   %Index based on one dimension, same index as activityIndexY and activityIndexX
   activityIndex = find(activityData{activityTimeIndex});
   %parse weight matrix excluding margins
   weightTimeIndex = floor(activityTimeIndex / writeStep); %Convert to weight time index 

   for i = 1:numArbors      %Iterate through number of arbors 
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Image reconstruction
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (RECONSTRUCTION_FLAG > 0)
         %Output baseline is zero 
         outMat = zeros(columnSizeY, columnSizeX);
         for j = 1:length(activityIndex)   %Iterate through spiking activity
            %If the spiking activity is not in the allowed area
            if isempty(find(marginIndex == activityIndex(j)))
               continue;   %Skip
            else
               %Set weight matrix to outmat to the square of the weight matrix
               for k = 1:numFeatures  %For multiple features
                  %Add on weights based on size of weight matrix
                  outMat(activityIndexY(j) - marginY: activityIndexY(j) + marginY,...
                  activityIndexX(j) - marginX: activityIndexX(j) + marginX) += ...
                  weightDataOn{weightTimeIndex}.values{i}(:, :, k, activityIndex(j));

                  %Subtract off weights
                  outMat(activityIndexY(j) - marginY: activityIndexY(j) + marginY,...
                  activityIndexX(j) - marginX: activityIndexX(j) + marginX) -= ...
                  weightDataOff{weightTimeIndex}.values{i}(:, :, k, activityIndex(j));
               end
            end
         end

         %Scale image and print
         figure;
         imagesc(outMat);
         colorbar;
         colormap(gray);
         title(['Reconstruction - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(i)])
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%Weight Map
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if (WEIGHTS_MAP_FLAG > 0)
         %Output baseline is zero 
         onWeightsOutMat = zeros(columnSizeY * patchSizeY, columnSizeX * patchSizeX);
         offWeightsOutMat = zeros(columnSizeY * patchSizeY, columnSizeX * patchSizeX);
         for j = 1:length(marginIndex)   %Iterate through allowed margin index
            %Convert index to weight out mat
            startIndexX = (marginIndexX(j) - 1) * patchSizeX + 1; 
            startIndexY = (marginIndexY(j) - 1) * patchSizeY + 1; 
            endIndexX = marginIndexX(j) * patchSizeX;
            endIndexY = marginIndexY(j) * patchSizeY;

            for k = 1:numFeatures  %For multiple features
               %Add on weights
               onWeightsOutMat(startIndexY:endIndexY, startIndexX:endIndexX) =...
               weightDataOn{weightTimeIndex}.values{i}(:, :, k, marginIndex(j));
               
               %Subtract off weights
               offWeightsOutMat(startIndexY:endIndexY, startIndexX:endIndexX) =...
               weightDataOff{weightTimeIndex}.values{i}(:, :, k, marginIndex(j));
            end
         end
         %Scale image and print
         figure;
         imagesc(onWeightsOutMat);
         colorbar;
         colormap(gray);
         title(['On Weight Map - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(i)])

         figure;
         imagesc(offWeightsOutMat);
         colorbar;
         colormap(gray);
         title(['Off Weight Map - time: ', num2str(activityTimeIndex), ' arbor: ', num2str(i)])
      end
   end
end
