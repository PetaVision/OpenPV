outDir = '/nh/compneuro/Data/Depth/white_RELU/';
v1ActFile = ['/nh/compneuro/Data/Depth/a12_V1_RELU.pvp'];
depthFile = ['/nh/compneuro/Data/Depth/white_LCA/a3_DepthDownsample.pvp'];
plotOutDir = [outDir, '/depthTuning/'];

dictPvpDir = '/nh/compneuro/Data/Depth/white_LCA_dictLearn/'
dictPvpFiles = {[dictPvpDir, 'V1ToLeftError_W.pvp'];...
                [dictPvpDir, 'V1ToRightError_W.pvp']};

%Given a depth and a neuron, these values define how big of an x/y patch to look for that neuron at
sampleDim = 5;
numDepthBins = 64;

calcDepthTuning(plotOutDir, v1ActFile, depthFile, dictPvpFiles, sampleDim, numDepthBins);

