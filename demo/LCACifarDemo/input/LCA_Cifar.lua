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
local nbatch = 32; --Batch size of learning

local cifarInputPath = ""; --TODO
local outputPath = ""; --TODO

local numBasisVectors = 128; --Total number of basis vectors being learned
local VThresh = .015; --The threshold, or lambda, of the network
local AShift = .015; --This being equal to VThresh is a soft threshold
local timeConstantTau = 100; --The integration tau for sparse approximation


local params = {};
--HyPerCol, or the container of the model
params['column'] = {
  groupType = "HyPerCol"; --Type of group. HyPerCol is the outermost wrapper of the model.
  nx = nxSize; --The size of the model. All other layers are relative to these sizes
  ny = nySize;
  nbatch = nbatch; --The batch size of the model
  randomSeed = 123456890; --The random seed for the random number generator
  startTime = 0; --The starting time of the simulation
  stopTime = (numImages * displayPeriod * numEpochs) / nbatch; --The ending time of the simulation
  outputPath = outputPath; --The output path of the simulation
  checkpointWrite = true; --Specifies if we are writing out checkpoints. We can boot from checkpoints later on
  checkpointWriteDir = outputPath .. "/Checkpoints"; --The checkpoint output directory
  checkpointWriteStepInterval = (displayPeriod * 100) / nbatch; --How often to checkpoint
  writeTimescales = true; --If we are writing out the adaptive timesteps
  dt = 1; --Each timestep advances the time by dt
  dtAdaptFlag = true; --If we are adapting the timestep for faster LCA convergence
  dtChangeMin = -0.05; --Adaptive timestep parameters
  dtChangeMax =  0.01;
  dtScaleMin  =  0.001;
  dtMinToleratedTimeScale = 0.0001;
  filenamesContainLayerNames = 2; --Filenames will only contain layer and connection name
  progressInterval = (10 * displayPeriod) / nbatch; --How often to print out a progress report
};

--Specifies the layer to read the images
params['Input'] = {
  groupType = "Movie"; --Movie layers read in a list of images to read
  nxScale = 1; --The relative scale to HyPerCol's nx and ny
  nyScale = 1;
  nf      = 3; --The number of features in this layer. 3 for color
  phase = 0; --The relative update time as compared to other layers
  displayPeriod = displayPeriod; --How often we show each image for
  inputPath = cifarInputPath; --The list of images to use
  batchMethod = "byImage"; --How each batch reads through the list of images.
  start_frame_index = 0; --Starting point into the list of iamges.
  writeStep = -1; --How often we are writing out this layer. -1 means do not write out.
  normalizeLuminanceFlag = false; --Image preprocessing
  normalizeStdDev = false;
};

--Creates a new layer that scales the input. This makes the scale of the input
--match the scale of the basis vectors, as we are normalizing the basis vectors
--to have a unit norm.
subnets.addScaleValueConn{
   pvParams       = params; --Parameter group object to add layers to
   inputLayerName = "Input"; --The input layer name
   scaleFactor    = 1/math.sqrt(xPatchSize * yPatchSize); --The scale factor to scale the image with
};

--Connection table input to addLCASubnet
connTable = {
   nxp              = xPatchSize; --The patch size of the basis vectors
   nyp              = yPatchSize;
   plasticityFlag   = plasticityFlag; --If we are updateing this layer via hebbian learning rule.
   momentumTau      = momentumTau; --The momentum parameter
   dWMax            = dwMax; --The learning rate
   normalizeMethod  = "normalizeL2"; --We are normalizing the weights to have a unit norm
   strength         = 1; --We are setting the l2 norm to be equal to 1
   momentumMethod   = "viscosity"; --Momentum method
};

--Change parameters as needed based on if a basisVectorFile was specified
if(basisVectorFile == nil) then
   connTable["weightInitType"] = "UniformRandomWeight" --How to initialize the weights
   connTable["wMinInit"] = -1 --Range of weights to initialize to
   connTable["wMaxInit"] = 1
   connTable["sparseFraction"] = .9; --90% of the weights will be initialized to 0
else
   connTable["weightInitType"] = "FileWeight" --Loading weights from a file
   connTable["initWeightsFile"] = basisVectorFile; --Name of file to load
end

--Retrieve an LCA module
subnets.addLCASubnet{
   pvParams                      = params; --Outermost parameter group to add groups to
   lcaLayerName                  = "V1"; --The name of the LCA layer
   inputLayerName                = "InputScaled"; --The name of the input to the LCA layer
   stride                        = 2; --The stride of the receptive fields

   lcaParams = { 
      nf              = numBasisVectors; --Number of basis vectors to use
      VThresh         = VThresh; --A threshold value for neurons to be active
      AShift          = AShift; --If this value is equal to VThresh, using a soft threshold
      AMin            = 0; --The minimum value of activations in this layer is 0
      AMax            = INFINITY; --The maximum value of activations in this layer is infinity
      timeConstantTau = timeConstantTau; --The integration time constant
      InitVType       = "UniformRandomV"; --How to initialize the values in this layer
      minV            = -1; --Range of values to initialize
      maxV            = 0.05; 
   };

   connParams = connTable;
   triggerLayerName = "Input"; --LCA will update the weights whenever the Input layer updates
};

--Print the OpenPV friendly parameter file to stdout
pv.printConsole(params);
