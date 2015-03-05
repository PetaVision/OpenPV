addpath('~/workspace/PetaVision/mlab/util');

outDir = '/home/ec2-user/mountData/benchmark/train/aws_rcorr_LCA/';
v1ActFile = [outDir, 'a0_LCA_V1.pvp'];
depthFile = [outDir, 'a3_DepthDownsample.pvp'];

%Given a depth and a neuron, these values define how big of an x/y patch to look for that neuron at
sampleDim = 3;

numDepthBins = 64;



%function calcDepthTuning()

%Create output directory in outDir
plotOutDir = [outDir, '/depthTuning/'];
mkdir(plotOutDir);

disp(['Reading pvp files']);
[v1_data, v1_hdr] = readpvpfile(v1ActFile);
[depth_data, depth_hdr] = readpvpfile(depthFile);

%sampleDim must be odd
assert(mod(sampleDim, 2) == 1);

%v1's first write is at time 0, remove
v1_data(1) = [];
assert(v1_data{1}.time == depth_data{1}.time);
assert(size(v1_data) == size(depth_data));
assert(v1_hdr.nxGlobal == depth_hdr.nxGlobal);
assert(v1_hdr.nyGlobal == depth_hdr.nyGlobal);
assert(depth_hdr.nf == 1);

numTimes = length(v1_data);
nx = v1_hdr.nxGlobal;
ny = v1_hdr.nyGlobal;
nf = v1_hdr.nf;

disp(['Reshaping pvp files']);
%Reshape cell structs to be one big array
v1_act = zeros(numTimes, nf, nx, ny);
depth_act = zeros(numTimes, nx, ny);
for(t = 1:length(v1_data))
   %TODO check if layer is sparse
   N = nx * ny * nf;
   active_ndx = v1_data{t}.values(:, 1);
   active_vals = v1_data{t}.values(:, 2);
   tmp_v1 = full(sparse(active_ndx+1, 1, active_vals, N, 1, N));
   %pv does [nf, nx, ny] ordering
   tmp_v1 = reshape(tmp_v1, [nf, nx, ny]);
   v1_act(t, :, :, :) = tmp_v1;
   depth_act(t, :, :) = depth_data{t}.values;
end

%Take out unnessessary depth_data and v1_data for memory
clear depth_data v1_data tmp_v1 active_ndx active_vals;

disp(['Histograming Depth']);
%bin depth_act into 64 bins, and use the index as the new matrix
%Note that 1 is now the DNC region
[drop, depth_act] = histc(depth_act, 0:(1/numDepthBins):1);

%Given a depth and neuron, we want to find the family of the maximum activity
%Final out matrix will be [neuronIdx, depth, sample patch]
edgeMag = floor(sampleDim / 2);
for(ni = 1:nf)
   outVals = zeros(numDepthBins - 1, sampleDim * sampleDim);
   disp(['Calculating neuron ', num2str(ni), ' out of ' , num2str(nf)]);
   %Take a slice and squeeze out singleton dimension
   v1_slice = reshape(v1_act(:, ni, :, :), numTimes, nx, ny);
   assert(size(v1_slice) == size(depth_act));
   for(di = 2:numDepthBins)
      %Find locations of a given depth
      targetDepthIdx = find(depth_act == di);
      %If depth doesn't exist, skip
      if(isempty(targetDepthIdx))
         continue
      end
      %t, x, and y are subscripts for locations where depth == di
      [t, x, y] = ind2sub([numTimes nx ny], targetDepthIdx);
      %Variable to index into outVals
      sampleDimIdx = 1;
      for(diffx = -edgeMag:edgeMag)
         %Calculate the difference and set to new x
         newx = x + diffx;
         %Check for border conditions
         newx(find(newx <= 0)) = 1;
         newx(find(newx > nx)) = nx;
         for(diffy = -edgeMag:edgeMag)
            newy = y + diffy;
            newy(find(newy <= 0)) = 1;
            newy(find(newy > ny)) = ny;
            %Convert back to linear index
            linIdxs = sub2ind([numTimes, nx, ny], t, newx, newy);
            %Find maximum activity of v1 activity slice
            maxVal = max(v1_slice(linIdxs)(:));
            if(length(maxVal) ~= 1)
               disp('maxVal not size of 1');
               keyboard
            end
            %Set to outVals
            outVals(di-1, sampleDimIdx) = maxVal;
            %Increment sampleDimIdx
            sampleDimIdx += 1;
         end
      end
      %Make sure the idx matches up and reset
      assert(sampleDimIdx == (sampleDim * sampleDim)+1);
      sampleDimIdx = 1;
   end
   handle = figure;
   hold on;
   for(plotIdx = 1:sampleDim * sampleDim)
      plot(outVals(:, plotIdx));
   end
   hold off;
   print(handle, [plotOutDir, num2str(ni), '.png']);
end




