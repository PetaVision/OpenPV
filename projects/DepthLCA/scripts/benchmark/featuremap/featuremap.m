addpath('~/workspace/pv-core/mlab/util');
addpath('~/workspace/OpenPV/pv-core/mlab/util');

doRank = false;
baseDir = "/home/sheng/mountData/benchmark/featuremap/icaweights_binoc_RELU_fine_sparse/";
tuningFile = "/home/ec2-user/mountData/benchmark/featuremap/fine/LCA_peakmean.txt";
outDir = [baseDir, '/featuremaps/'];
weightsOutDir = [outDir, '/weights/'];
mkdir(outDir);
mkdir(weightsOutDir);

layers = { ...
   'LeftRecon_slice'; ...
   'RightRecon_slice';...
   };

baseLayers = { ...
   ['/home/sheng/mountData/benchmark/featuremap/icaweights_binoc_LCA_fine/LeftRecon_slice.pvp']; ...
   ['/home/sheng/mountData/benchmark/featuremap/icaweights_binoc_LCA_fine/RightRecon_slice.pvp']; ...
};

dictElements = { ...
   '~/mountData/benchmark/icaweights_binoc_LCA_fine/Checkpoints/Checkpoint194000/V1ToLeftError_W.pvp'; ...
   '~/mountData/benchmark/icaweights_binoc_LCA_fine/Checkpoints/Checkpoint194000/V1ToRightError_W.pvp'; ...
}

numNeurons = 512;

sliceWeight = .9;
baseWeight = .8;

assert(length(layers) == length(baseLayers));

function res = grs2rgb(img, map)
   %%Convert grayscale images to RGB using specified colormap.
   %  IMG is the grayscale image. Must be specified as a name of the image 
   %  including the directory, or the matrix.
   %  MAP is the M-by-3 matrix of colors.
   %
   %  RES = GRS2RGB(IMG) produces the RGB image RES from the grayscale image IMG 
   %  using the colormap HOT with 64 colors.
   %
   %  RES = GRS2RGB(IMG,MAP) produces the RGB image RES from the grayscale image 
   %  IMG using the colormap matrix MAP. MAP must contain 3 columns for Red, 
   %  Green, and Blue components.  
   %
   %  Example 1:
   %  open 'image.tif'; 
   %  res = grs2rgb(image);
   %
   %  Example 2:
   %  cmap = colormap(summer);
   %  res = grs2rgb('image.tif',cmap);
   %
   %  See also COLORMAP, HOT
   %
   %  Written by 
   %  Valeriy R. Korostyshevskiy, PhD
   %  Georgetown University Medical Center
   %  Washington, D.C.
   %  December 2006
   %
   %  vrk@georgetown.edu

   % Check the arguments
   if nargin<1
   error('grs2rgb:missingImage','Specify the name or the matrix of the image');
   end;
   if ~exist('map','var') || isempty(map)
      map = hot(64);
   end;
   [l,w] = size(map);
   if w~=3
      error('grs2rgb:wrongColormap','Colormap matrix must contain 3 columns');
   end;
   if ischar(img)
      a = imread(img);
   elseif isnumeric(img)
      a = img;
   else
   error('grs2rgb:wrongImageFormat','Image format: must be name or matrix');
   end;
   % Calculate the indices of the colormap matrix
   a = double(a);
   a(a==0) = 1; % Needed to produce nonzero index of the colormap matrix
   ci = ceil(l*a/max(a(:))); 
   % Colors in the new image
   [il,iw] = size(a);
   r = zeros(il,iw); 
   g = zeros(il,iw);
   b = zeros(il,iw);
   r(:) = map(ci,1);
   g(:) = map(ci,2);
   b(:) = map(ci,3);
   % New image
   res = zeros(il,iw,3);
   res(:,:,1) = r; 
   res(:,:,2) = g; 
   res(:,:,3) = b;
end




if(doRank)
   tf = fopen(tuningFile, 'r');

   tline = fgetl(tf); %Throw away first line
   tline = fgetl(tf);

   rank = zeros(numNeurons, 1);

   %neuron values here are 1 indexed
   for i = 1:numNeurons
      rank(i) = str2num(strsplit(tline, ':'){1});
      tline = fgetl(tf);
   end
else
   rank = 1:numNeurons;
end

%For each image
for li = 1:length(layers)
   %Get element
   [weightData, weightHdr] = readpvpfile(dictElements{li});
   weights = weightData{1}.values{1};

   %Get baseline image
   [baseData, baseHdr] = readpvpfile(baseLayers{li});
   baseImg = baseData{1}.values';
   %Scale image to have same std
   normVal = max(abs(max(baseImg(:))),abs(min(baseImg(:))));
   if(normVal != 0)
      baseImg = ((baseImg/normVal)+1)/2;
   end
   [ny, nx] = size(baseImg);
   colorBaseImg = zeros(ny, nx, 3);
   colorBaseImg(:, :, 1) = baseImg;
   colorBaseImg(:, :, 2) = baseImg;
   colorBaseImg(:, :, 3) = baseImg;

   %baseImg = (baseImg - min(baseImg(:)))/(max(baseImg(:)) - min(baseImg(:)));

   %Make output image per neuron
   for ranki = 1:numNeurons
      ni = rank(ranki);
      if(doRank)
         imgOutName = sprintf('%s/%s_rank%03d_neuron%03d.png', outDir, layers{li}, ranki-1, ni-1)
         weightOutName = sprintf('%s/%s_rank%03d_neuron%03d.png', weightsOutDir, layers{li}, ranki-1, ni-1)
      else
         imgOutName = sprintf('%s/%s_neuron%03d.png', outDir, layers{li}, ni-1)
         weightOutName = sprintf('%s/%s_neuron%03d.png', weightsOutDir, layers{li}, ni-1)
      end
      pvpInName = sprintf('%s/paramsweep_%03d/%s.pvp', baseDir, ni-1, layers{li});
      [sliceData, sliceHdr] = readpvpfile(pvpInName);
      sliceImg = sliceData{1}.values';
      normVal = max(abs(max(sliceImg(:))),abs(min(sliceImg(:))));
      if(normVal != 0)
         sliceImg = ((sliceImg/normVal)+1)/2;
      end
      cmap = zeros(128, 3);
      cmap(1:64, 1) = 1:-1/(64-1):0;
      cmap(65:end, 3) = 0:1/(64-1):1;
      try
         colorSliceImg = grs2rgb(sliceImg, cmap);
      catch e
         keyboard
      end

      outImg = baseWeight.*colorBaseImg+ sliceWeight.*colorSliceImg;
      outImg(find(outImg(:) > 1)) = 1;

      imwrite(outImg, imgOutName);

      weightOut = weights(:, :, :, ni);
      weightOut = squeeze(weightOut)';
      weightOut = (weightOut - min(weightOut(:)))/(max(weightOut(:)) - min(weightOut(:)));
      imwrite(weightOut, weightOutName);
   end
end
