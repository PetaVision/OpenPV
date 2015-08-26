--Load packages
package.path = package.path .. ";" 
            .. os.getenv("HOME") 
            .. "/workspaceGit/OpenPV/pv-core/parameterWrapper/?.lua";
local pv = require "PVModule";
local subnets = require "PVSubnets";

--Parameters
local nxSize = 32; --Cifar is 32 x 32
local nySize = 32;
local xPatchSize = 5; --Patch size of basis vectors
local yPatchSize = 5;
local displayPeriod = 1000; --Number of timesteps to find sparse approximation

local basisVectorFile = nil; --nil for initial weights, otherwise, specifies the weights file to load for dictionaries
local plasticityFlag = true; --Determines if we are learning weights or holding them constant
local momentumTau = 100; --The momentum parameter. A single weight update will last for momentumTau timesteps.
local dwMax = .5; --The learning rate

local numImages = 50000; --Total number of images in dataset
local numEpochs = 1; --Number of times to run through dataset
local nbatch = 1; --Batch size of learning

local cifarInputPath = ""; --TODO
local outputPath = ""; --TODO

local numBasisVectors = 128; --Total number of basis vectors being learned
local VThresh = .015; --The threshold, or lambda, of the network
local AShift = .015; --This being equal to VThresh is a soft threshold
local timeConstantTau = 100; --The integration tau for sparse approximation


local params = {};
--HyPerCol, or the container of the model
params['column'] = {
  groupType = "HyPerCol";
  nx = nxSize;
  ny = nySize;
  nbatch = nbatch;
  randomSeed = 123456890;
  startTime = 0;
  stopTime = (numImages * displayPeriod * numEpochs) / nbatch;
  outputPath = outputPath;
  checkpointWrite = true;
  checkpointWriteDir = outputPath .. "/Checkpoints";
  checkpointWriteStepInterval = (displayPeriod * 100) / nbatch;
  writeTimescales = true;
  dt = 1;
  dtAdaptFlag = true;
  dtChangeMin = -0.01;
  dtChangeMax =  0.01;
  dtScaleMin  =  0.001;
  dtMinToleratedTimeScale = 0.0001;
  filenamesContainLayerNames = 2;
  progressInterval = (10 * displayPeriod) / nbatch;
};

--Specifies the layer to read the images
params['Input'] = {
  groupType = "Movie";
  nxScale = 1;
  nyScale = 1;
  nf      = 3;

  phase = 0;

  displayPeriod = displayPeriod;

  inputPath = cifarInputPath;
  batchMethod = "byImage";

  start_frame_index = 0;

  writeStep = -1;

  normalizeLuminanceFlag = false;
  normalizeStdDev = false;
};

--Creates a new layer that scales the input. This makes the scale of the input
--match the scale of the basis vectors, as we are normalizing the basis vectors
--to have a unit norm.
subnets.addScaleValueConn{
   pvParams       = params;
   inputLayerName = "Input";
   scaleFactor    = 1/math.sqrt(xPatchSize * yPatchSize);
};

--Connection table input to addLCASubnet
connTable = {
   nxp              = xPatchSize;
   nyp              = yPatchSize;
   plasticityFlag   = plasticityFlag;
   momentumTau      = momentumTau;
   dWMax            = dwMax;
   normalizeMethod  = "normalizeL2";
   strength         = 1;
   momentumMethod   = "viscosity";
   triggerLayerName = "Input";
};

--Change parameters as needed based on if a basisVectorFile was specified
if(basisVectorFile == nil) then
   connTable["weightInitType"] = "UniformRandomWeight"
   connTable["wMinInit"] = -1
   connTable["wMaxInit"] = 1
   connTable["sparseFraction"] = .9;
else
   connTable["weightInitType"] = "FileWeight"
   connTable["initWeightsFile"] = basisVectorFile;
end

--Retrieve an LCA module
subnets.addLCASubnet{
   pvParams                      = params;
   lcaLayerName                  = "V1";
   inputLayerName                = "InputScaled";
   stride                        = 1;

   lcaParams = { 
      nf              = numBasisVectors;
      VThresh         = VThresh;
      AShift          = AShift;
      AMin            = 0;
      AMax            = INFINITY;
      timeConstantTau = timeConstantTau;
      InitVType       = "UniformRandomV";
      minV            = -1;
      maxV            = 0.05; 
   };

   connParams = connTable;
   triggerLayerName = "Input";
};

--Print the OpenPV friendly parameter file to stdout
pv.printConsole(params);
