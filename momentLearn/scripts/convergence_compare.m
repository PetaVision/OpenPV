output_dir = '/nh/compneuro/Data/momentLearn/output/';
runs_dir = {[output_dir, 'no_momentum_out/'];...
            [output_dir, 'simple_momentum_out/'];...
            [output_dir, 'viscosity_momentum_out/']};

lineSpecs = {'r', 'g', 'b'};
legendStr = {'No momentum'; 'Simple momentum'; 'Viscosity momentum'};

mkdir([output_dir, '/convergence/']);

handle = figure;
hold on
for(outIdx = 1:length(runs_dir))
   checkpointdirs = [runs_dir{outIdx}, 'Checkpoints/'];
   weightspvpfile = [runs_dir{outIdx}, 'w1_S1ToError.pvp'];

   [data, hdr] = readpvpfile(weightspvpfile);

   function out = getWeightFrame(iData, iHdr, iFrame)
      out = reshape(iData{iFrame}.values{1}, [iHdr.nxp * iHdr.nyp * iHdr.nfp, iHdr.numPatches]);
   end

   outVals = zeros(size(data)-1, 1);

   %Grab flattened final frame
   finalWeights = getWeightFrame(data, hdr, size(data, 1));
   finalSq = finalWeights .^ 2;
   finalL2 = sum(finalSq, 1);
   finalTime = data{end}.time;

   for(time = 1:size(data)-1)
      currWeights = getWeightFrame(data, hdr, time);
      diff = finalWeights - currWeights;
      diffSq = diff .^ 2;
      diffL2 = sum(diffSq, 1);
      convVal = sum(diffL2) ./ sum(finalL2);
      outVals(time) = convVal;
   end

   plot(outVals, lineSpecs{outIdx});
end

hold off
legend(legendStr{1}, legendStr{2}, legendStr{3});
print(handle, [output_dir, 'convergence/', num2str(finalTime), '.png']);
