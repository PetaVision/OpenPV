outDir = '/nh/compneuro/Data/Depth/LCA/benchmark/LCATrain/';
v1ActFile = [outDir, 'a12_V1.pvp'];
depthFile = ['/nh/compneuro/Data/Depth/LCA/benchmark/train/nmc_rcorr_LCA/a3_DepthDownsample.pvp'];
plotOutDir = [outDir, '/depthTuning/'];

dictPvpDir = '/nh/compneuro/Data/Depth/LCA/benchmark/LCATrain/Checkpoint39000/'
dictPvpFiles = {[dictPvpDir, 'V1ToLeftError_W.pvp'];...
                [dictPvpDir, 'V1ToRightError_W.pvp']};

%Given a depth and a neuron, these values define how big of an x/y patch to look for that neuron at
sampleDim = 5;
numDepthBins = 64;

calcDepthTuning(plotOutDir, v1ActFile, depthFile, dictPvpFiles, sampleDim, numDepthBins);

