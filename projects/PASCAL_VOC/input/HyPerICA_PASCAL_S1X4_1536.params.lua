-- PetaVision params file for dictionary of experts: createded by garkenyon May  6 15:19:53 2015

-- Load util module in PV trunk: NOTE this may need to change
package.path = package.path .. ";" .. os.getenv("HOME") .. "/workspace/pv-core/parameterWrapper/PVModule.lua"
local pv = require "PVModule"

-- Global variable, for debug parsing
-- Needed by printConsole
debugParsing              = true

--Image parameters
local imageListPath = "/home/ec2-user/mountData/PASCAL_VOC/VOC2007/VOC2007_landscape_192X256_list.txt";
local GroundTruthPath = "/home/ec2-user/mountData/PASCAL_VOC/VOC2007/VOC2007_landscape_192X256.pvp";
local displayPeriod       = 1200
local startFrame          = 1
local numEpochs           = 4
local stopTime            = 7958 * displayPeriod;

-- User defined variables
local nxSize              = 256
local nySize              = 192
local experimentName      = "PASCAL_S1X4_1536_ICA"
local runName             = "VOC2007_landscape"
local runVersion          = 10
local machinePath         = "/home/ec2-user/mountData"
local databasePath        = "PASCAL_VOC"
local outputPath          = machinePath .. "/" .. databasePath .. "/" .. experimentName .. "/" .. runName .. runVersion
local inputPath           = machinePath .. "/" .. databasePath .. "/" .. experimentName .. "/" .. runName .. runVersion-2
--local inputPath           = machinePath .. "/" .. databasePath .. "/" .. "PASCAL_S1_1536_ICA" .. "/" .. runName .. "9"
local checkpointID         = stopTime*numEpochs
local inf                 = 3.40282e+38
local initializeFromCheckpointFlag = false;


--i/o parameters
local writePeriod         = 100 * displayPeriod;

--HyPerLCA parameters
local VThresh               = 0.025
local VWidth                = infinity
local learningRate          = 0
local dWMax                 = 10.0
local learningMomentumTau   = 500
local patchSize             = 16
local stride                = 8 --16
local tau                   = 400
local S1_numFeatures        = patchSize * patchSize * 3 * 2; -- (patchSize/stride)^2 Xs overcomplete (i.e. complete for orthonormal ICA basis for stride == patchSize)

--Ground Truth parameters
local numClasses            = 20
local nxScale_GroundTruth   = 0.015625 --0.03125 --0.0625;
local nyScale_GroundTruth   = 0.015625 --0.03125 --0.0625;


-- Base table variable to store
local pvParams = {
   column = {
      groupType = "HyPerCol"; --String values
      startTime                           = 0;
      dt                                  = 1;
      dtAdaptFlag                         = true;
      dtScaleMax                          = 10;
      dtScaleMin                          = 0.01;
      dtChangeMax                         = 0.01;
      dtChangeMin                         = -0.02;
      dtMinToleratedTimeScale             = 0.0001;
      stopTime                            = stopTime*numEpochs;
      progressInterval                    = 1000;
      writeProgressToErr                  = true;
      verifyWrites                        = false;
      outputPath                          = outputPath;
      printParamsFilename                 = experimentName .. "_" .. runName .. ".params";
      randomSeed                          = 1234567890;
      nx                                  = nxSize;
      ny                                  = nySize;
      filenamesContainLayerNames          = 2; --true;
      filenamesContainConnectionNames     = 2; --true;
      initializeFromCheckpointDir         = inputPath .. "/Checkpoints/Checkpoint" .. checkpointID;
      defaultInitializeFromCheckpointFlag = initializeFromCheckpointFlag;
      checkpointWrite                     = true;
      checkpointWriteDir                  = outputPath .. "/Checkpoints";
      checkpointWriteTriggerMode          = "step";
      checkpointWriteStepInterval         = 1*writePeriod;
      deleteOlderCheckpoints              = false;
      suppressNonplasticCheckpoints       = false;
      writeTimescales                     = true;
      errorOnNotANumber                   = false;
   };
} --End of pvParams

   pv.addGroup(pvParams, "Image", 
	       {
		  groupType = "Movie";
		  nxScale                             = 1;
		  nyScale                             = 1;
		  nf                                  = 3;
		  phase                               = 0;
		  mirrorBCflag                        = true;
		  initializeFromCheckpointFlag        = false;
		  writeStep                           = writePeriod;
		  initialWriteTime                    = writePeriod;
		  sparseLayer                         = false;
		  writeSparseValues                   = false;
		  updateGpu                           = false;
		  dataType                            = nil;
		  offsetAnchor                        = "tl";
		  offsetX                             = 0;
		  offsetY                             = 0;
		  writeImages                         = 0;
		  useImageBCflag                      = false;
		  autoResizeFlag                      = false;
		  inverseFlag                         = false;
		  normalizeLuminanceFlag              = true;
		  normalizeStdDev                     = true;
		  jitterFlag                          = 0;
		  padValue                            = 0;
		  inputPath                            = imageListPath;
		  displayPeriod                       = displayPeriod;
		  echoFramePathnameFlag               = true;
		  start_frame_index                   = 1;
		  skip_frame_index                    = 0;
		  writeFrameToTimestamp               = true;
		  flipOnTimescaleError                = true;
		  resetToStartOnLoop                  = false;
	       }
   )

   pv.addGroup(pvParams, "ImageReconS1Error",
	       {
		  groupType = "ANNNormalizedErrorLayer";
		  nxScale                             = 1;
		  nyScale                             = 1;	
		  nf                                  = 3;
		  phase                               = 1;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
		  InitVType                           = "ZeroV";
		  triggerLayerName                    = NULL;
		  writeStep                           = writePeriod;
		  initialWriteTime                    = writePeriod;
		  sparseLayer                         = false;
		  updateGpu                           = false;
		  dataType                            = nil;
		  VThresh                             = 0;
		  AMin                                = 0;
		  AMax                                = inf;
		  AShift                              = 0;
		  VWidth                              = 0;
		  clearGSynInterval                   = 0;
		  errScale                            = 1;
	       }
   ) 

local S1_Movie = true
if S1_Movie then
   local S1MoviePath                                   = inputPath .. "/" .. "a10_S1.pvp"
   pv.addGroup(pvParams, "ConstantS1",
	       {
		  groupType = "MoviePvp";
		  nxScale                             = 1.0/stride;
		  nyScale                             = 1.0/stride;
		  nf                                  = S1_numFeatures;
		  phase                               = 0;
		  mirrorBCflag                        = false;
		  initializeFromCheckpointFlag        = false;
		  writeStep                           = -1;
		  sparseLayer                         = true;
		  writeSparseValues                   = true;
		  updateGpu                           = false;
		  dataType                            = nil;
		  offsetAnchor                        = "tl";
		  offsetX                             = 0;
		  offsetY                             = 0;
		  writeImages                         = 0;
		  useImageBCflag                      = false;
		  autoResizeFlag                      = false;
		  inverseFlag                         = false;
		  normalizeLuminanceFlag              = false;
		  jitterFlag                          = 0;
		  padValue                            = 0;
		  inputPath                           = S1MoviePath;
		  displayPeriod                       = displayPeriod;
		  randomMovie                         = 0;
		  readPvpFile                         = true;
		  start_frame_index                   = 1;
		  skip_frame_index                    = 0;
		  writeFrameToTimestamp               = true;
		  flipOnTimescaleError                = true;
		  resetToStartOnLoop                  = false;
	       }
   )
else
   pv.addGroup(pvParams, "ConstantS1",
	       {
		  groupType = "ConstantLayer";
		  nxScale                             = 1.0/stride;
		  nyScale                             = 1.0/stride;
		  nf                                  = S1_numFeatures;
		  phase                               = 0;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
		  InitVType                           = "ConstantV";
		  valueV                              = VThresh;
		  writeStep                           = -1;
		  sparseLayer                         = false;
		  updateGpu                           = false;
		  dataType                            = nil;
		  VThresh                             = -inf;
		  AMin                                = -inf;
		  AMax                                = inf;
		  AShift                              = 0;
		  VWidth                              = 0;
		  clearGSynInterval                   = 0;
	       }
   )
end
   pv.addGroup(pvParams, "S1",
	       {
		  groupType = "HyPerLCALayer";
		  nxScale                             = 1.0/stride;
		  nyScale                             = 1.0/stride;
		  nf                                  = S1_numFeatures;
		  phase                               = 2;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
		  --InitVType                           = "InitVFromFile";
		  --Vfilename                           = inputPath .. "/Checkpoints/Checkpoint" .. checkpointID .. "/S1_V.pvp";
		  InitVType                           = "UniformRandomV";
		  minV                                = -1;
		  maxV                                = 0.05;
		  triggerLayerName                    = "Image";
		  triggerBehavior                     = "resetStateOnTrigger";
		  triggerResetLayerName               = "ConstantS1";
		  triggerOffset                       = 0;
		  writeStep                           = displayPeriod;
		  initialWriteTime                    = displayPeriod;
		  sparseLayer                         = true;
		  writeSparseValues                   = true;
		  updateGpu                           = false; --true;
		  dataType                            = nil;
		  VThresh                             = VThresh;
		  AMin                                = 0;
		  AMax                                = inf;
		  AShift                              = 0;
		  VWidth                              = 100;
		  clearGSynInterval                   = 0;
		  numChannels                         = 1;
		  timeConstantTau                     = tau;
		  numWindowX                          = 1;
		  numWindowY                          = 1;
		  selfInteract                        = true;
	       }
   )

   pv.addGroup(pvParams, "ImageReconS1",
	       {
		  groupType = "ANNLayer";
		  nxScale                             = 1;
		  nyScale                             = 1;
		  nf                                  = 3;
		  phase                               = 3;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
		  InitVType                           = "ZeroV";
		  triggerLayerName                    = NULL;
		  writeStep                           = writePeriod;
		  initialWriteTime                    = writePeriod;
		  sparseLayer                         = false;
		  updateGpu                           = false;
		  dataType                            = nil;
		  VThresh                             = -inf;
		  AMin                                = -inf;
		  AMax                                = inf;
		  AShift                              = 0;
		  VWidth                              = 0;
		  clearGSynInterval                   = 0;
	       }
   )



-- Ground Truth 

pv.addGroup(pvParams, "GroundTruthPixels",
	    {
	       groupType = "MoviePvp";
	       nxScale                             = 1;
	       nyScale                             = 1;
	       nf                                  = numClasses;
	       phase                               = 0;
	       mirrorBCflag                        = true;
	       initializeFromCheckpointFlag        = false;
	       writeStep                           = -1;
	       sparseLayer                         = true;
	       writeSparseValues                   = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       offsetAnchor                        = "tl";
	       offsetX                             = 0;
	       offsetY                             = 0;
	       writeImages                         = 0;
	       useImageBCflag                      = false;
	       autoResizeFlag                      = false;
	       inverseFlag                         = false;
	       normalizeLuminanceFlag              = false;
	       jitterFlag                          = 0;
	       padValue                            = 0;
	       inputPath                           = GroundTruthPath;
	       displayPeriod                       = displayPeriod;
	       randomMovie                         = 0;
	       readPvpFile                         = true;
	       start_frame_index                   = 1;
	       skip_frame_index                    = 0;
	       writeFrameToTimestamp               = true;
	       flipOnTimescaleError                = true;
	       resetToStartOnLoop                  = false;
	    }
)

pv.addGroup(pvParams, "GroundTruthNoBackground",
	    pvParams.ImageReconS1,
	    {
	       nxScale                             = nxScale_GroundTruth;
	       nyScale                             = nyScale_GroundTruth;
	       nf                                  = numClasses;
	       phase                               = 1;
	       writeStep                           = -1;
	       sparseLayer                         = true;
	    }
)
pvParams.GroundTruthNoBackground.triggerLayerName  = "GroundTruthPixels";
pvParams.GroundTruthNoBackground.triggerBehavior   = "updateOnlyOnTrigger";
pvParams.GroundTruthNoBackground.triggerOffset     = 0;
pvParams.GroundTruthNoBackground.writeSparseValues = false;

pv.addGroup(pvParams, "GroundTruth",
	    {
	       groupType = "BackgroundLayer";
	       nxScale                             = nxScale_GroundTruth;
	       nyScale                             = nyScale_GroundTruth;
	       nf                                  = numClasses + 1;
	       phase                               = 2;
	       mirrorBCflag                        = false;
	       valueBC                             = 0;
	       initializeFromCheckpointFlag        = false;
	       triggerLayerName                    = "GroundTruthPixels";
	       triggerBehavior                     = "updateOnlyOnTrigger";
	       triggerOffset                       = 0;
	       writeStep                           = displayPeriod;
	       initialWriteTime                    = displayPeriod;
	       sparseLayer                         = true;
	       writeSparseValues                   = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       originalLayerName                   = "GroundTruthNoBackground";
	       repFeatureNum                       = 1;
	    }
)

pv.addGroup(pvParams, "GroundTruthReconS1Error",
	    {
	       groupType = "ANNErrorLayer";
	       nxScale                             = nxScale_GroundTruth;
	       nyScale                             = nyScale_GroundTruth;
	       nf                                  = numClasses + 1;
	       phase                               = 10;
	       mirrorBCflag                        = false;
	       valueBC                             = 0;
	       initializeFromCheckpointFlag        = false;
	       InitVType                           = "ZeroV";
	       triggerFlag                         = true;
	       triggerLayerName                    = "GroundTruthPixels";
	       triggerBehavior                     = "updateOnlyOnTrigger";
	       triggerOffset                       = 1;
	       writeStep                           = displayPeriod;
	       initialWriteTime                    = displayPeriod;
	       sparseLayer                         = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       VThresh                             = 0;
	       AMin                                = 0;
	       AMax                                = inf;
	       AShift                              = 0;
	       VWidth                              = 0;
	       clearGSynInterval                   = 0;
	       errScale                            = 1;
	    }
)

pv.addGroup(pvParams, "GroundTruthReconS1",
	    {
	       groupType = "ANNLayer";
	       nxScale                             = nxScale_GroundTruth;
	       nyScale                             = nyScale_GroundTruth;
	       nf                                  = numClasses + 1;
	       phase                               = 9;
	       mirrorBCflag                        = false;
	       valueBC                             = 0;
	       initializeFromCheckpointFlag        = false;
	       InitVType                           = "ZeroV";
	       triggerFlag                         = true;
	       triggerLayerName                    = "GroundTruthPixels";
	       triggerBehavior                     = "updateOnlyOnTrigger";
	       triggerOffset                       = 1;
	       writeStep                           = displayPeriod;
	       initialWriteTime                    = displayPeriod;
	       sparseLayer                         = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       VThresh                             = -inf;
	       AMin                                = -inf;
	       AMax                                = inf;
	       AShift                              = 0;
	       VWidth                              = 0;
	       clearGSynInterval                   = 0;
	    }
)

pv.addGroup(pvParams, "BiasS1",
	    {
	       groupType = "ConstantLayer";
	       nxScale                             = nxScale_GroundTruth;
	       nyScale                             = nyScale_GroundTruth;
	       nf                                  = 1;
	       phase                               = 0;
	       mirrorBCflag                        = false;
	       valueBC                             = 0;
	       initializeFromCheckpointFlag        = false;
	       InitVType                           = "ConstantV";
	       valueV                              = 1;
	       writeStep                           = -1;
	       sparseLayer                         = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       VThresh                             = -inf;
	       AMin                                = -inf;
	       AMax                                = inf;
	       AShift                              = 0;
	       VWidth                              = 0;
	       clearGSynInterval                   = 0;
	    }
)

pv.addGroup(pvParams, "S1MaxPooled",
	    {
	       groupType = "ANNLayer";
	       nxScale                             = nxScale_GroundTruth;
	       nyScale                             = nyScale_GroundTruth;
	       nf                                  = 1536;
	       phase                               = 8;
	       mirrorBCflag                        = false;
	       valueBC                             = 0;
	       initializeFromCheckpointFlag        = false;
	       InitVType                           = "ZeroV";
	       triggerLayerName                    = "GroundTruthPixels";
	       triggerBehavior                     = "updateOnlyOnTrigger";
	       triggerOffset                       = 1;
	       writeStep                           = -1;
	       sparseLayer                         = false;
	       updateGpu                           = false;
	       dataType                            = nil;
	       VThresh                             = -inf;
	       AMin                                = -inf;
	       AMax                                = inf;
	       AShift                              = 0;
	       VWidth                              = 0;
	       clearGSynInterval                   = 0;
	    }
);


--connections 
pv.addGroup(pvParams, "ImageToImageReconS1Error",
	    {
	       groupType = "HyPerConn";
	       preLayerName                        = "Image";
	       postLayerName                       = "ImageReconS1Error";
	       channelCode                         = 0;
	       delay                               = {0.000000};
	       numAxonalArbors                     = 1;
	       plasticityFlag                      = false;
	       convertRateToSpikeCount             = false;
	       receiveGpu                          = false;
	       sharedWeights                       = true;
	       weightInitType                      = "OneToOneWeights";
	       initWeightsFile                     = nil;
	       weightInit                          = 0.032075; --((1/patchSize)*(1/patchSize)*(1/3))^(1/2); --
	       initializeFromCheckpointFlag        = false;
	       updateGSynFromPostPerspective       = false;
	       pvpatchAccumulateType               = "convolve";
	       writeStep                           = -1;
	       writeCompressedCheckpoints          = false;
	       selfFlag                            = false;
	       nxp                                 = 1;
	       nyp                                 = 1;
	       nfp                                 = 3;
	       shrinkPatches                       = false;
	       normalizeMethod                     = "none";
	    }
)

pv.addGroup(pvParams, "ImageReconS1ToImageReconS1Error",
	    {
	       groupType = "IdentConn";
	       preLayerName                        = "ImageReconS1";
	       postLayerName                       = "ImageReconS1Error";
	       channelCode                         = 1;
	       delay                               = {0.000000};
	       initWeightsFile                     = nil;
	       writeStep                           = -1;
	    }
)


pv.addGroup(pvParams, "S1ToImageReconS1Error",
	    {
	       groupType = "MomentumConn";
	       preLayerName                        = "S1";
	       postLayerName                       = "ImageReconS1Error";
	       channelCode                         = -1;
	       delay                               = {0.000000};
	       numAxonalArbors                     = 1;
	       plasticityFlag                      = true;
	       convertRateToSpikeCount             = false;
	       receiveGpu                          = false;
	       sharedWeights                       = true;
	       weightInitType                      = "FileWeight";
	       initWeightsFile                     = inputPath .. "/Checkpoints/Checkpoint" .. checkpointID .. "/S1ToImageReconS1Error_W.pvp";
	       useListOfArborFiles                 = false;
	       combineWeightFiles                  = false;    
	       --weightInitType                      = "UniformRandomWeight";
	       --initWeightsFile                     = nil;
	       --wMinInit                            = -1;
	       --wMaxInit                            = 1;
	       --sparseFraction                      = 0.9;
	       initializeFromCheckpointFlag        = false;
	       triggerFlag                         = true;
	       triggerLayerName                    = "Image";
	       triggerOffset                       = 1;
	       updateGSynFromPostPerspective       = true;
	       pvpatchAccumulateType               = "convolve";
	       writeStep                           = -1;
	       writeCompressedCheckpoints          = false;
	       selfFlag                            = false;
	       combine_dW_with_W_flag              = false;
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
	       momentumTau                         = momentumTau;
	       momentumMethod                      = "viscosity";
	       momentumDecay                       = 0;
	    }
)

pv.addGroup(pvParams, "ImageReconS1ErrorToS1",
	    {
	       groupType = "TransposeConn";
	       preLayerName                        = "ImageReconS1Error";
	       postLayerName                       = "S1";
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
	       originalConnName                    = "S1ToImageReconS1Error";
	    }
)

pv.addGroup(pvParams, "S1ToImageReconS1",
	    {
	       groupType = "CloneConn";
	       preLayerName                        = "S1";
	       postLayerName                       = "ImageReconS1";
	       channelCode                         = 0;
	       delay                               = {0.000000};
	       convertRateToSpikeCount             = false;
	       receiveGpu                          = false;
	       updateGSynFromPostPerspective       = false;
	       pvpatchAccumulateType               = "convolve";
	       writeStep                           = -1;
	       writeCompressedCheckpoints          = false;
	       selfFlag                            = false;
	       originalConnName                    = "S1ToImageReconS1Error";
	    }
)


-- Ground Truth connections

pv.addGroup(pvParams, "GroundTruthPixelsToGroundTruthNoBackground",
	    {
	       groupType = "PoolingConn";
	       preLayerName                        = "GroundTruthPixels";
	       postLayerName                       = "GroundTruthNoBackground";
	       channelCode                         = 0;
	       delay                               = {0.000000};
	       numAxonalArbors                     = 1;
	       convertRateToSpikeCount             = false;
	       receiveGpu                          = false;
	       sharedWeights                       = true;
	       initializeFromCheckpointFlag        = false;
	       updateGSynFromPostPerspective       = false;
	       pvpatchAccumulateType               = "maxpooling";
	       writeStep                           = -1;
	       writeCompressedCheckpoints          = false;
	       selfFlag                            = false;
	       nxp                                 = 1;
	       nyp                                 = 1;
	       shrinkPatches                       = false;
	       needPostIndexLayer                  = false;
	    }
)

pv.addGroup(pvParams, "GroundTruthToGroundTruthReconS1Error",
	    {
	       groupType = "IdentConn";
	       preLayerName                        = "GroundTruth";
	       postLayerName                       = "GroundTruthReconS1Error";
	       channelCode                         = 0;
	       delay                               = {0.000000};
	       initWeightsFile                     = nil;
	       writeStep                           = -1;
	    }
)

pv.addGroup(pvParams, "S1ToS1MaxPooled",
	    pvParams.GroundTruthPixelsToGroundTruthNoBackground,
	    {
	       preLayerName                        = "S1";
	       postLayerName                       = "S1MaxPooled";
	    }
)

pv.addGroup(pvParams, "S1MaxPooledToGroundTruthReconS1Error",
	    {
	       groupType = "MomentumConn";
	       preLayerName                        = "S1MaxPooled";
	       postLayerName                       = "GroundTruthReconS1Error";
	       channelCode                         = -1;
	       delay                               = {0.000000};
	       numAxonalArbors                     = 1;
	       plasticityFlag                      = true;
	       convertRateToSpikeCount             = false;
	       receiveGpu                          = false;
	       sharedWeights                       = true;
	       weightInitType                      = "FileWeight";
	       initWeightsFile                     = inputPath .. "/Checkpoints/Checkpoint" .. checkpointID .. "/S1MaxPooledToGroundTruthReconS1Error_W.pvp";
	       useListOfArborFiles                 = false;
	       combineWeightFiles                  = false;    
	       --weightInitType                      = "UniformRandomWeight";
	       --initWeightsFile                     = nil;
	       --wMinInit                            = -0;
	       --wMaxInit                            = 0;
	       sparseFraction                      = 0;
	       initializeFromCheckpointFlag        = false;
	       triggerFlag                         = true;
	       triggerLayerName                    = "Image";
	       triggerOffset                       = 1;
	       updateGSynFromPostPerspective       = false;
	       pvpatchAccumulateType               = "convolve";
	       writeStep                           = -1;
	       writeCompressedCheckpoints          = false;
	       selfFlag                            = false;
	       combine_dW_with_W_flag              = false;
	       nxp                                 = 1;
	       nyp                                 = 1;
	       shrinkPatches                       = false;
	       normalizeMethod                     = "none";
	       dWMax                               = 1.0; --0.5; --0.01;
	       keepKernelsSynchronized             = true;
	       useMask                             = false;
	       momentumTau                         = 1;
	       momentumMethod                      = "viscosity";
	       momentumDecay                       = 0;
	       batchPeriod                         = 1;
	    }
)

pv.addGroup(pvParams, "BiasS1ToGroundTruthReconS1Error",
	    pvParams.S1MaxPooledToGroundTruthReconS1Error,
	    {
	       preLayerName                        = "BiasS1";
	       postLayerName                       = "GroundTruthReconS1Error";
	       dWMax                               = 0.01;
	       initWeightsFile                     = inputPath .. "/Checkpoints/Checkpoint" .. checkpointID .. "/BiasS1ToGroundTruthReconS1Error_W.pvp";
	    }
)

pv.addGroup(pvParams, "S1MaxPooledToGroundTruthReconS1",
	    pvParams.S1ToImageReconS1,
	    {
	       preLayerName                        = "S1MaxPooled";
	       postLayerName                       = "GroundTruthReconS1";
	       originalConnName                    = "S1MaxPooledToGroundTruthReconS1Error";
	    }
)

pv.addGroup(pvParams, "BiasS1ToGroundTruthReconS1",
	    pvParams.S1MaxPooledToGroundTruthReconS1,
	    {
	       preLayerName                        = "BiasS1";
	       postLayerName                       = "GroundTruthReconS1";
	       originalConnName                    = "BiasS1ToGroundTruthReconS1Error";
	    }
)

pv.addGroup(pvParams, "GroundTruthReconS1ToGroundTruthReconS1Error",
	    {
	       groupType = "IdentConn";
	       preLayerName                        = "GroundTruthReconS1";
	       postLayerName                       = "GroundTruthReconS1Error";
	       channelCode                         = 1;
	       delay                               = {0.000000};
	       initWeightsFile                     = nil;
	       writeStep                           = -1;
	    }
)

-- Energy probe

pv.addGroup(pvParams, "S1EnergyProbe", 
	    {
	       groupType                           = "ColumnEnergyProbe";
	       probeOutputFile                     = "S1EnergyProbe.txt";
	    }
)

pv.addGroup(pvParams, "ImageReconS1ErrorL2NormEnergyProbe",
	    {
	       groupType                           = "L2NormProbe"
	       targetLayer                         = "ImageReconS1Error";
	       message                             = NULL;
	       textOutputFlag                      = true;
	       probeOutputFile                     = "ImageReconS1ErrorL2NormEnergyProbe.txt";
	       triggerFlag                         = true;
	       triggerLayerName                    = "Image";
	       triggerOffset                       = 1;
	       energyProbe                         = "S1EnergyProbe";
	       coefficient                         = 0.5;
	       maskLayerName                       = NULL;
	       exponent                            = 2;
	    }
)

L1NormProbe "S1L1NormEnergyProbe" = {
    targetLayer                         = "S1";
    message                             = NULL;
    textOutputFlag                      = true;
    probeOutputFile                     = "S1EnergyProbe.txt";
    triggerFlag                         = false;
    energyProbe                         = "S1L1NormEnergyProbe";
    coefficient                         = 0.025;
    maskLayerName                       = NULL;
};



-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParams)
