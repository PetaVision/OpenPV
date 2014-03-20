function createRankFileFromPvp
   more off;
   format long;
   %%pvpFile = '/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_vine/a2_V1_S2.pvp';
   %%pvpFile = '/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_vine/a3_V1_S4.pvp';
   pvpFile = '/nh/compneuro/Data/vine/LCA/2013_01_24_2013_02_01/output_stack_vine/a4_V1_S8.pvp';
   [path,name,ext] = fileparts(pvpFile);
   outputFile = [path, filesep, name, "_ranks.txt"];
   fileID = fopen(outputFile, "a");

   addpath('/nh/home/wshainin/workspace/PetaVision/mlab/util');

   [data, hdr] = readpvpfile(pvpFile);
   global nf = hdr.nf
   numProcs = nproc;
   A = zeros(1,nf);

   AA = parcellfun(numProcs, @getRanks, data, 'UniformOutput',false);
   %%AA = cellfun(@getRanks, data, 'UniformOutput', false);

   for i=1:numel(data)
      A += AA{i};
   end
   for k=1:numel(A)
      fprintf(fileID, '%u\n',A(k));
   end

   fclose(fileID);
   keyboard;
end

function [A] = getRanks(in)
   global nf;
   A = zeros(1,nf);

   numVals = numel(in.values(:,1));
   for j=1:numVals
      fIDX = mod(in.values(j,1),nf) + 1;
      A(1,fIDX) += 1;
   end
end
