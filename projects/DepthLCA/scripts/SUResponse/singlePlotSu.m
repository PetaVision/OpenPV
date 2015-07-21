addpath("/home/slundquist/workspace/PetaVision/mlab/util");
outputDir = "/nh/compneuro/Data/Depth/LCA/SUResponse/single_ICA_cell_response/";
timestampFilename = [outputDir, "/timestamps/LeftImage.txt"];
timestampFile = fopen(timestampFilename, 'r');

function [fullfilename, filename, frame, VLayer, neuronIdx] = parseLine(line)
   fullfilename = strsplit(line, ','){end};
   split = strsplit(line, '/');
   %Get frame number
   frame = str2num(strsplit(split{1}, ','){2});
   %Get which V1 activity it is and neuron index
   filename = split{end};
   temp = strsplit(filename, '_');
   neuronIdx = num2str(temp{4}(1:end-4));
   VLayer = temp{1}(1:2);
end


plotOutDir = [outputDir, 'ICA_Plot/']

mkdir(plotOutDir);

tline = fgetl(timestampFile);
[fullfilename, filename, startFrame, VLayer, neuronIdx] = parseLine(tline);


isLeft = true;
while(tline != -1)
   tline
   pvpFilename = [outputDir, "a4_V1.pvp"];
   %Read lines until no longer matches
   checkFilename = filename;
   while(strcmp(filename, checkFilename))
      tline = fgetl(timestampFile);
      [next_fullfilename, checkFilename, next_startFrame, next_VLayer, next_neuronIdx] = parseLine(tline);
   end
   %Next frame is ready to go
   %Current frame's end frame is one less than the next start frame

   %Grab pvp file given by Vlayer
   [data, hdr] = readpvpfile(pvpFilename, 0, next_startFrame, startFrame+1);

   %Calculate the index (in pv indices) for the middle of the layer, plus the neuron we're looking at
   pv_idx = ((hdr.nyGlobal/2)-1)*hdr.nxGlobal*hdr.nf + ...
            ((hdr.nxGlobal/2)-1) * hdr.nf + ...;
            neuronIdx;

   activation = zeros(length(data), 1);
   %Loop through frames
   for(i = 1:length(data))
      data_idxs = data{i}.values(:, 1);
      activeIdx = find(data_idxs == pv_idx);
      if(isempty(activeIdx)) 
         activation(i) = 0;
      else
         activation(i) = data{i}.values(activeIdx, 2);
      end
   end
   
   if(isLeft)
      h  = subplot(2, 2, 1);
      h  = imshow(fullfilename);
      h  = subplot(2, 2, 3);
      h = plot(activation);
      xlabel("Disparity");
      ylabel("Activation");
      ylim([-1, 2]);
   else
      h  = subplot(2, 2, 2);
      h  = imshow(fullfilename);
      h  = subplot(2, 2, 4);
      h = plot(activation);
      xlabel("Disparity");
      ylim([-1, 2]);
      saveas(h, [plotOutDir, "SU_Neuron_", num2str(neuronIdx), ".png"]);
   end

   %Set next values to current values
   fullfilename = next_fullfilename;
   filename = checkFilename;
   startFrame = next_startFrame;
   VLayer = next_VLayer;
   neuronIdx = next_neuronIdx;
   if(isLeft)
      isLeft=false;
   else
      isLeft=true;
   end
end




