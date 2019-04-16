-- Path to OpenPV. Replace this with an absolute path.
package.path = package.path .. ";" .. "../../../parameterWrapper/?.lua";
local pv = require "PVModule";

local nbatch           = 20;    --Number of images to process in parallel
local nxSize           = 32;    --CIFAR images are 32 x 32
local nySize           = 32;
local patchSize        = 8;
local stride           = 2
local displayPeriod    = 400;   --Number of timesteps to find sparse approximation
local numEpochs        = 1;     --Number of times to run through dataset
local numImages        = 50000; --Total number of images in dataset
local stopTime         = math.ceil((numImages  * numEpochs) / nbatch) * displayPeriod;
local writeStep        = displayPeriod; 
local initialWriteTime = displayPeriod; 

local inputPath        = "../cifar-10-batches-mat/mixed_cifar.txt";
local outputPath       = "../output/";
local checkpointPeriod = (displayPeriod * 100); -- How often to write checkpoints

local dictionarySize   = 128;   --Number of patches/elements in dictionary 
local dictionaryFile   = nil;   --nil for initial weights, otherwise, specifies the weights file to load.
local plasticityFlag   = true;  --Determines if we are learning our dictionary or holding it constant
local momentumTau      = 500;   --Weight momentum parameter. A single weight update will last for momentumTau timesteps.
local dWMax            = 0.05;  --The learning rate
local VThresh          = 0.55;  --The threshold, or lambda, of the network
local AMin             = 0;
local AMax             = infinity;
local AShift           = VThresh;  --This being equal to VThresh is a soft threshold
local VWidth           = 0; 
local timeConstantTau  = 100;   --The integration tau for sparse approximation
local weightInit       = 1.0;

-- Base table variable to store
local pvParameters = {

   --Layers------------------------------------------------------------
   --------------------------------------------------------------------   
   column = {
      groupType = "HyPerCol";
      startTime                           = 0;
      dt                                  = 1;
      stopTime                            = stopTime;
      progressInterval                    = (displayPeriod * 10);
      writeProgressToErr                  = true;
      verifyWrites                        = false;
      outputPath                          = outputPath;
      printParamsFilename                 = "CIFAR_Tutorial.params";
      randomSeed                          = 1234567890;
      nx                                  = nxSize;
      ny                                  = nySize;
      nbatch                              = nbatch;
      checkpointWrite                     = true;
      checkpointWriteDir                  = outputPath .. "/Checkpoints"; --The checkpoint output directory
      checkpointWriteTriggerMode          = "step";
      checkpointWriteStepInterval         = checkpointPeriod; --How often to checkpoint
      checkpointIndexWidth                = -1; -- Automatically select width of index in checkpoint directory name
      deleteOlderCheckpoints              = false;
      suppressNonplasticCheckpoints       = false;
      initializeFromCheckpointDir         = "";
      errorOnNotANumber                   = false;
   };

   AdaptiveTimeScales = {
      groupType = "KneeTimeScaleProbe";
      targetName                          = "V1EnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "AdaptiveTimeScales.txt";
      triggerLayerName                    = "Input";
      triggerOffset                       = 0;
      baseMax                             = 0.06;  -- Initial upper bound for timescale growth
      baseMin                             = 0.05;  -- Initial value for timescale growth
      tauFactor                           = 0.03;  -- Percent of tau used as growth target
      growthFactor                        = 0.025; -- Exponential growth factor. The smaller value between this and the above is chosen. 
      writeTimeScalesFieldnames           = false;
      kneeThresh                          = 1.0;
      kneeSlope                           = 0.01;
   };

   Input = {
      groupType = "ImageLayer";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = 3;
      phase                               = 0;
      mirrorBCflag                        = true;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
      inputPath                           = inputPath;
      offsetAnchor                        = "tl";
      offsetX                             = 0;
      offsetY                             = 0;
      inverseFlag                         = false;
      normalizeLuminanceFlag              = true;
      normalizeStdDev                     = true;
      useInputBCflag                      = false;
      autoResizeFlag                      = false;
      displayPeriod                       = displayPeriod;
      batchMethod                         = "byImage";
      writeFrameToTimestamp               = true;
      resetToStartOnLoop                  = false;
   };

   InputError = {
      groupType = "HyPerLayer";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = 3;
      phase                               = 1;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };

   V1 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 1/stride;
      nyScale                             = 1/stride;
      nf                                  = dictionarySize;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      updateGpu                           = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
      adaptiveTimeScaleProbe              = "AdaptiveTimeScales";
   };

   InputRecon = {
      groupType = "HyPerLayer";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = 3;
      phase                               = 3;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ZeroV";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
   };

--Connections ------------------------------------------------------
--------------------------------------------------------------------

   InputToError = {
      groupType = "RescaleConn";
      preLayerName                        = "Input";
      postLayerName                       = "InputError";
      channelCode                         = 0;
      delay                               = {0.000000};
      scale                               = weightInit;
   };

   ErrorToV1 = {
      groupType = "TransposeConn";
      preLayerName                        = "InputError";
      postLayerName                       = "V1";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = true;
      updateGSynFromPostPerspective       = true;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "V1ToInputError";
   };

   V1ToInputError = {
      groupType = "MomentumConn";
      preLayerName                        = "V1";
      postLayerName                       = "InputError";
      channelCode                         = -1;
      delay                               = {0.000000};
      numAxonalArbors                     = 1;
      plasticityFlag                      = plasticityFlag;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false; -- non-sparse -> non-sparse
      sharedWeights                       = true;
      weightInitType                      = "UniformRandomWeight";
      wMinInit                            = -1;
      wMaxInit                            = 1;
      sparseFraction                      = 0.9;
      minNNZ                              = 0;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "Input";
      triggerOffset                       = 0;
      immediateWeightUpdate               = true;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      writeCompressedWeights              = false;
      writeCompressedCheckpoints          = false;
      combine_dW_with_W_flag              = false;
      nxp                                 = patchSize;
      nyp                                 = patchSize;
      normalizeMethod                     = "normalizeL2";
      strength                            = 1;
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = false;
      minL2NormTolerated                  = 0;
      dWMax                               = dWMax; 
      momentumTau                         = momentumTau;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
      momentumMethod                      = "viscosity";
      momentumDecay                       = 0;
   }; 

   V1ToRecon = {
      groupType = "CloneConn";
      preLayerName                        = "V1";
      postLayerName                       = "InputRecon";
      channelCode                         = 0;
      delay                               = {0.000000};
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeCompressedCheckpoints          = false;
      originalConnName                    = "V1ToInputError";
   };

   ReconToError = {
      groupType = "IdentConn";
      preLayerName                        = "InputRecon";
      postLayerName                       = "InputError";
      channelCode                         = 1;
      delay                               = {0.000000};
      initWeightsFile                     = nil;
   };

   --Probes------------------------------------------------------------
   --------------------------------------------------------------------

   V1EnergyProbe = {
      groupType = "ColumnEnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V1EnergyProbe.txt";
      triggerLayerName                    = nil;
      energyProbe                         = nil;
   };

   InputErrorL2NormEnergyProbe = {
      groupType = "L2NormProbe";
      targetLayer                         = "InputError";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "InputErrorL2NormEnergyProbe.txt";
      energyProbe                         = "V1EnergyProbe";
      coefficient                         = 0.5;
      maskLayerName                       = nil;
      exponent                            = 2;
   };

   V1L1NormEnergyProbe = {
      groupType = "L1NormProbe";
      targetLayer                         = "V1";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "V1L1NormEnergyProbe.txt";
      energyProbe                         = "V1EnergyProbe";
      coefficient                         = VThresh;
      maskLayerName                       = nil;
   };

} --End of pvParameters

if dictionaryFile ~= nil then
   pvParameters.V1ToInputError.weightInitType  = "FileWeight";
   pvParameters.V1ToInputError.initWeightsFile = dictionaryFile;
end
-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParameters)
