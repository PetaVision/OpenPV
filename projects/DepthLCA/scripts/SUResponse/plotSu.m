addpath("/home/slundquist/workspace/PetaVision/mlab/util");
outputDir = "/nh/compneuro/Data/Depth/LCA/SUResponse/ICA_cell_response/";
timestampFilename = [outputDir, "/timestamps/LeftImage.txt"];
timestampFile = fopen(timestampFilename, 'r');

function [filename, frame, VLayer, neuronIdx] = parseLine(line)
   split = strsplit(line, '/');
   %Get frame number
   frame = str2num(strsplit(split{1}, ','){2});
   %Get which V1 activity it is and neuron index
   filename = split{9};
   temp = strsplit(filename, '_');
   neuronIdx = num2str(temp{4}(1:end-4));
   VLayer = temp{1}(1:4);
end



tline = fgetl(timestampFile);
[filename, startFrame, VLayer, neuronIdx] = parseLine(tline);

while(tline != -1)
   tline
   if(strcmp(VLayer, "V1S2"))
      pvpFilename = [outputDir, "a4_", VLayer, ".pvp"];
   elseif(strcmp(VLayer, "V1S4"))
      pvpFilename = [outputDir, "a5_", VLayer, ".pvp"];
   elseif(strcmp(VLayer, "V1S8"))
      pvpFilename = [outputDir, "a6_", VLayer, ".pvp"];
   else
      disp(["Unrecognized VLayer ", VLayer]);
      exit
   end
   %Read lines until no longer matches
   checkFilename = filename;
   while(strcmp(filename, checkFilename))
      tline = fgetl(timestampFile);
      [checkFilename, next_startFrame, next_VLayer, next_neuronIdx] = parseLine(tline);
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
   figure;
   plot(activation)
   keyboard

   %Set next values to current values
   filename = checkFilename;
   startFrame = next_startFrame;
   VLayer = next_VLayer;
   neuronIdx = next_neuronIdx;
end




