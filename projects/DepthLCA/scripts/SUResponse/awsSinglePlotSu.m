addpath("/home/ec2-user/workspace/PetaVision/mlab/util");
outputDir = "/home/ec2-user/mountData/SUResponse/single_ICA_cell_response/";
global disparityDisplayPeriod = 40;

probeFilename = [outputDir, 'ICA_V1.probe']
probeFile = fopen(probeFilename, 'r');
plotOutDir = [outputDir, 'ICA_Plot/']
mkdir(plotOutDir);

function [time, aVal, neuronIdx, isLeft] = parseLine(line)
   global disparityDisplayPeriod;
   split = strsplit(line, ' ');
   %1st split is time, 3rd is activity
   time = str2num(strsplit(split{1}, '='){end});
   dispIdx = floor((time-1)/(disparityDisplayPeriod));
   neuronIdx = floor(dispIdx/2);
   isRight = mod(dispIdx, 2);
   isLeft = ~isRight;
   aVal = str2num(strsplit(split{3}, '='){end});
end

xAct = [0:2:79]';

%First line is useless, initialization
pline = fgetl(probeFile);

while(pline != -1)
%   %Read lines until no longer matches
%   checkFilename = filename;
%   while(strcmp(filename, checkFilename))
%      tline = fgetl(timestampFile);
%      [next_fullfilename, checkFilename, next_startFrame, next_VLayer, next_neuronIdx] = parseLine(tline);
%   end
%   %Next frame is ready to go
%   %Current frame's end frame is one less than the next start frame
%
   left_activation = zeros(disparityDisplayPeriod, 1);
   right_activation = zeros(disparityDisplayPeriod, 1);
   %Loop through frames
   for(eye = 1:2)
      for(i = 1:disparityDisplayPeriod)
         pline = fgetl(probeFile)
         if(pline == -1)
            exit(0)
         end
         [time, aVal, neuronIdx, isLeft] = parseLine(pline);
         if(isLeft)
            assert(eye == 1)
            left_activation(i) = aVal;
         else
            assert(eye == 2)
            right_activation(i) = aVal;
         end
      end
   end

   y_min = min([min(left_activation(:)) min(right_activation(:))]);
   y_max = max([max(left_activation(:)) max(right_activation(:))]);

   left_fullfilename = ['/home/ec2-user/mountData/DepthLCA/input/SUResponse/single/weightImg/V1ToLeftError_W_patch_', num2str(neuronIdx), '.png'];
   h  = subplot(2, 2, 1);
   h  = imshow(left_fullfilename);
   h  = subplot(2, 2, 3);
   h = plot(xAct, left_activation);
   ylim([y_min y_max]);
   xlabel("Disparity");
   ylabel("Activation");

   right_fullfilename = ['/home/ec2-user/mountData/DepthLCA/input/SUResponse/single/weightImg/V1ToRightError_W_patch_', num2str(neuronIdx), '.png'];
   h  = subplot(2, 2, 2);
   h  = imshow(right_fullfilename);
   h  = subplot(2, 2, 4);
   h = plot(xAct, right_activation);
   ylim([y_min y_max]);
   xlabel("Disparity");
   saveas(h, [plotOutDir, "SU_Neuron_", num2str(neuronIdx), ".png"]);
end
