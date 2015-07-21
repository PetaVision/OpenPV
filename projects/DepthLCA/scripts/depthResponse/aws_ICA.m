outDir = '/nh/compneuro/Data/Depth/LCA/benchmark/train/aws_rcorr_ICA/';
v1ActFile = ['~/workspace/DepthLCA/input/benchmark/Data/a6_ICA_V1.pvp'];
depthFile = ['/nh/compneuro/Data/Depth/LCA/benchmark/train/aws_rcorr_LCA/a3_DepthDownsample.pvp'];
plotOutDir = [outDir, '/depthTuning/'];

dictPvpDir = '/home/slundquist/workspace/DepthLCA/input/benchmark/Data/'
dictPvpFiles = {[dictPvpDir, 'LCA_V1ToLeftError_W.pvp'];...
                [dictPvpDir, 'LCA_V1ToRightError_W.pvp']};

%Given a depth and a neuron, these values define how big of an x/y patch to look for that neuron at
sampleDim = 7;
numDepthBins = 64;

calcDepthTuning(plotOutDir, v1ActFile, depthFile, dictPvpFiles, sampleDim, numDepthBins);

