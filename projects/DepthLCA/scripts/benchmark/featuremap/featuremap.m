addpath('~/workspace/pv-core/mlab/util');

baseDir = "/home/ec2-user/mountData/benchmark/featuremap/fine/icaweights_binoc_LCA_fine/";
tuningFile = "/home/ec2-user/mountData/benchmark/featuremap/fine/LCA_peakmean.txt";
outDir = [baseDir, '/featuremaps/'];
mkdir(outDir);

layers = { ...
   'LeftRecon_slice'; ...
   'RightRecon_slice';...
   };

baseLayers = { ...
   [baseDir, '/all/LeftRecon_slice.pvp']; ...
   [baseDir, '/all/RightRecon_slice.pvp']; ...
};

numNeurons = 512;

sliceAlpha = .7;

assert(length(layers) == length(baseLayers));
tf = fopen(tuningFile, 'r');

tline = fgetl(tf); %Throw away first line
tline = fgetl(tf);

rank = zeros(numNeurons, 1);

%neuron values here are 1 indexed
for i = 1:numNeurons
   rank(i) = str2num(strsplit(tline, ':'){1});
   tline = fgetl(tf);
end

%For each image
for li = 1:length(layers)
   %Get baseline image
   [baseData, baseHdr] = readpvpfile(baseLayers{li});
   baseImg = baseData{1}.values';
   %Scale image to be between 0 and 1
   baseImg = (baseImg - min(baseImg(:)))/(max(baseImg(:)) - min(baseImg(:)));

   %Make output image per neuron
   for ranki = 1:numNeurons
      ni = rank(ranki);
      imgOutName = sprintf('%s/%s_rank%03d_neuron%03d.png', outDir, layers{li}, ranki-1, ni-1)
      pvpInName = sprintf('%s/paramsweep_%03d/%s.pvp', baseDir, ni-1, layers{li});
      [sliceData, sliceHdr] = readpvpfile(pvpInName);
      sliceImg = sliceData{1}.values';
      %Scale sliceImg with mean of 0 and std of 1
      sliceImg = (sliceImg - mean(sliceImg(:)))/std(sliceImg(:));
      %Scale slice to be between 0 and 1
      sliceImg = (sliceImg - min(sliceImg(:)))/(max(sliceImg(:))-min(sliceImg(:)));
      outImg = (1-sliceAlpha).*baseImg + sliceAlpha.*sliceImg;
      imwrite(outImg, imgOutName);
   end
end
