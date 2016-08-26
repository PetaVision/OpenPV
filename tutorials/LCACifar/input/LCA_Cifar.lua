package.path = package.path .. ";" .. "../../../parameterWrapper/?.lua";
local pv = require "PVModule";
--local subnets = require "PVSubnets";

local nbatch           = 32;    --Batch size of learning
local nxSize           = 32;    --Cifar is 32 x 32
local nySize           = 32;
local patchSize        = 12;
local stride           = 2
local displayPeriod    = 400;   --Number of timesteps to find sparse approximation
local numEpochs        = 4;     --Number of times to run through dataset
local numImages        = 50000; --Total number of images in dataset
local stopTime         = math.ceil((numImages  * numEpochs) / nbatch) * displayPeriod;
local writeStep        = displayPeriod; 
local initialWriteTime = displayPeriod; 

local inputPath        = "../cifar-10-batches-mat/mixed_cifar.txt";
local outputPath       = "../output/";
local checkpointPeriod = (displayPeriod * 100); -- How often to write checkpoints

local numBasisVectors  = 128;   --overcompleteness x (stride X) x (Stride Y) * (# color channels) * (2 if rectified) 
local basisVectorFile  = nil;   --nil for initial weights, otherwise, specifies the weights file to load. Change init parameter in MomentumConn
local plasticityFlag   = true;  --Determines if we are learning weights or holding them constant
local momentumTau      = 200;   --The momentum parameter. A single weight update will last for momentumTau timesteps.
local dWMax            = 10;    --The learning rate
local VThresh          = .015;  -- .005; --The threshold, or lambda, of the network
local AMin             = 0;
local AMax             = infinity;
local AShift           = .015;  --This being equal to VThresh is a soft threshold
local VWidth           = 0; 
local timeConstantTau  = 100;   --The integration tau for sparse approximation
local weightInit       = math.sqrt((1/patchSize)*(1/patchSize)*(1/3));

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
      filenamesContainLayerNames          = 2;
      filenamesContainConnectionNames     = 2;
      initializeFromCheckpointDir         = "";
      defaultInitializeFromCheckpointFlag = false;
      checkpointWrite                     = true;
      checkpointWriteDir                  = outputPath .. "/Checkpoints"; --The checkpoint output directory
      checkpointWriteTriggerMode          = "step";
      checkpointWriteStepInterval         = checkpointPeriod; --How often to checkpoint
      deleteOlderCheckpoints              = false;
      suppressNonplasticCheckpoints       = false;
      errorOnNotANumber                   = false;
   };

   AdaptiveTimeScaleProbe "AdaptiveTimeScales" = {
      targetName                          = "V1EnergyProbe";
      message                             = nil;
      textOutputFlag                      = true;
      probeOutputFile                     = "AdaptiveTimeScales.txt";
      triggerLayerName                    = "Input";
      triggerOffset                       = 0;
      baseMax                             = .1; --1.0; -- minimum value for the maximum time scale, regardless of tau_eff
      baseMin                             = 0.01; -- default time scale to use after image flips or when something is wacky
      tauFactor                           = 0.1; -- determines fraction of tau_effective to which to set the time step, can be a small percentage as tau_eff can be huge
      growthFactor                        = 0.01; -- percentage increase in the maximum allowed time scale whenever the time scale equals the current maximum
      writeTimeScales                     = true;
   };

   Input = {
      groupType = "Movie";
      nxScale                             = 1;
      nyScale                             = 1;
      nf                                  = 3;
      phase                               = 0;
      mirrorBCflag                        = true;
      initializeFromCheckpointFlag        = false;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = false;
      updateGpu                           = false;
      dataType                            = nil;
      inputPath                           = inputPath;
      offsetAnchor                        = "tl";
      offsetX                             = 0;
      offsetY                             = 0;
      writeImages                         = 0;
      inverseFlag                         = false;
      normalizeLuminanceFlag              = true;
      normalizeStdDev                     = true;
      jitterFlag                          = 0;
      useImageBCflag                      = false;
      padValue                            = 0;
      autoResizeFlag                      = false;
      displayPeriod                       = displayPeriod;
      echoFramePathnameFlag               = true;
      batchMethod                         = "byImage";
      start_frame_index                   = {0.000000};
      writeFrameToTimestamp               = true;
      flipOnTimescaleError                = true;
      resetToStartOnLoop                  = false;
   };

   InputError = {
      groupType = "ANNLayer";
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
      VThresh                             = -infinity;
      AMin                                = -infinity;
      AMax                                = infinity;
      AShift                              = 0;
      clearGSynInterval                   = 0;
      useMask                             = false;
   };

   V1 = {
      groupType = "HyPerLCALayer";
      nxScale                             = 1/stride;
      nyScale                             = 1/stride;
      nf                                  = numBasisVectors;
      phase                               = 2;
      mirrorBCflag                        = false;
      valueBC                             = 0;
      initializeFromCheckpointFlag        = false;
      InitVType                           = "ConstantV";
      valueV                              = VThresh;
      --InitVType                           = "InitVFromFile";
      --Vfilename                           = "V1_V.pvp";
      triggerLayerName                    = NULL;
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      sparseLayer                         = true;
      writeSparseValues                   = true;
      updateGpu                           = true;
      dataType                            = nil;
      VThresh                             = VThresh;
      AMin                                = AMin;
      AMax                                = AMax;
      AShift                              = AShift;
      VWidth                              = VWidth;
      clearGSynInterval                   = 0;
      timeConstantTau                     = timeConstantTau;
      selfInteract                        = true;
   };

   InputRecon = {
      groupType = "ANNLayer";
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
      VThresh                             = -infinity;
      AMin                                = -infinity;
      AMax                                =  infinity;
      AShift                              = 0;
      VWidth                              = 0;
      clearGSynInterval                   = 0;
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
      selfFlag                            = false;
      gpuGroupIdx                         = -1;
      originalConnName                    = "V1ToError";
   };

   V1ToError = {
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
      --weightInitType                      = "FileWeight";
      --initWeightsFile                     = basisVectorFile;
      useListOfArborFiles                 = false;
      combineWeightFiles                  = false;
      initializeFromCheckpointFlag        = false;
      triggerLayerName                    = "Input";
      triggerOffset                       = 0;
      updateGSynFromPostPerspective       = false; -- Should be false from V1 (sparse layer) to Error (not sparse). Otherwise every input from pre will be calculated (Instead of only active ones)
      pvpatchAccumulateType               = "convolve";
      writeStep                           = writeStep;
      initialWriteTime                    = initialWriteTime;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = patchSize;
      nyp                                 = patchSize;
      shrinkPatches                       = false;
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
      keepKernelsSynchronized             = true;
      useMask                             = false;
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
      selfFlag                            = false;
      originalConnName                    = "V1ToError";
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
      coefficient                         = 0.025;
      maskLayerName                       = nil;
   };

} --End of pvParameters

-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParameters)
