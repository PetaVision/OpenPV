outDir = '/home/ec2-user/mountData/benchmark/train/aws_rcorr_white_LCA/';
v1ActFile = [outDir, 'a0_LCA_V1.pvp'];
depthFile = [outDir, 'a3_DepthDownsample.pvp'];
plotOutDir = [outDir, '/depthTuning/'];

dictPvpDir = '/home/ec2-user/mountData/benchmark/LCATrain/Checkpoints/Checkpoint430000/'
dictPvpFiles = {[dictPvpDir, 'V1ToLeftError_W.pvp'];...
                [dictPvpDir, 'V1ToRightError_W.pvp']};

%Given a depth and a neuron, these values define how big of an x/y patch to look for that neuron at
sampleDim = 5;
numDepthBins = 64;

calcDepthTuning(plotOutDir, v1ActFile, depthFile, dictPvpFiles, sampleDim, numDepthBins);

