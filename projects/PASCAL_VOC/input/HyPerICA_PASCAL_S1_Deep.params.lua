-- PetaVision params file for combining sparse coding + deep learning

-- Load util module in PV trunk: NOTE this may need to change
--package.path = package.path .. ";" .. os.getenv("HOME") .. "/workspace/pv-core/parameterWrapper/PVModule.lua"
package.path = package.path .. ";" .. os.getenv("HOME") .. "/openpv/pv-core/parameterWrapper/PVModule.lua"
--package.path = package.path .. ";" .. "/nh/compneuro/Data" .. "/openpv/pv-core/parameterWrapper/PVModule.lua"
local pv = require "PVModule"

-- Global variable, for debug parsing
-- Needed by printConsole
debugParsing              = true

--HyPerLCA parameters
local VThresh               = 0.025
local VWidth                = infinity
local learningRate          = 0
local dWMax                 = 10.0
local learningMomentumTau   = 500
local patchSize             = 16
local tau                   = 400
local S1_numFeatures        = patchSize * patchSize * 3 * 2 -- (patchSize/stride)^2 Xs overcomplete (i.e. complete for orthonormal ICA basis for stride == patchSize)

-- User defined variables
local plasticityFlag      = true
local stride              = patchSize/4
local nxSize              = 192 --256
local nySize              = 256 --192
local experimentName      = "PASCAL_S1X" .. math.floor(patchSize*patchSize/(stride*stride)) .. "_" .. S1_numFeatures .. "_Deep" .. "_ICA"
local experimentNameTmp   = "PASCAL_S1X" .. math.floor(patchSize*patchSize/(stride*stride)) .. "_" .. S1_numFeatures .. "_ICA"
local runName             = "VOC2007_landscape" -- "VOC2007_portrait" --
local runVersion          = 2
local runVersionTmp       = 7
local machinePath         = "/Volumes/mountData" --"/nh/compneuro/Data" --"/home/ec2-user/mountData"
local databasePath        = "PASCAL_VOC"
local outputPath          = machinePath .. "/" .. databasePath .. "/" .. experimentName .. "/" .. runName .. runVersion
local inputPath           = machinePath .. "/" .. databasePath .. "/" .. experimentNameTmp .. "/" .. runName .. runVersionTmp
local inputPathSLP        = machinePath .. "/" .. databasePath .. "/" .. experimentName .. "/" .. runName .. runVersion-1
local numImages           = 7958 --1751 --
local displayPeriod       = 240
local numEpochs           = 1
local stopTime            = numImages * displayPeriod * numEpochs
local checkpointID        = stopTime
local checkpointIDSLP     = stopTime
local writeStep           = 100 * displayPeriod
local initialWriteTime    = writeStep
local checkpointWriteStepInterval = writeStep
local S1_Movie            = true --false
local movieVersion        = 1
if S1_Movie then
   outputPath              = outputPath .. "_S1_Movie" .. movieVersion
   inputPath               = inputPath -- .. "_S1_Movie" .. movieVersion
   inputPathSLP            = inputPathSLP .. "_S1_Movie" .. movieVersion
   displayPeriod           = 1
   numEpochs               = 10
   stopTime                = numImages * displayPeriod * numEpochs
   --checkpointID            = stopTime
   checkpointIDSLP         = numImages * numEpochs --stopTime
   writeStep               = 1
   initialWriteTime        = numImages*(numEpochs-1)+1
   checkpointWriteStepInterval = numImages
else -- not used if run version == 1
   inputPath               = inputPath .. runVersion
   inputPathSLP            = inputPath .. "_S1_Movie" .. movieVersion
   --checkpointID            = numImages * displayPeriod * 5
   --checkpointIDSLP         = numImages * 10
end
local inf                 = 3.40282e+38
local initializeFromCheckpointFlag = false

--i/o parameters
local imageListPath       = machinePath .. "/" .. databasePath .. "/" .. "VOC2007" .. "/" .. "VOC2007_" .. "landscape_192X256_list.txt" -- "portrait_256X192_list.txt" -- 
local GroundTruthPath     = machinePath .. "/" .. databasePath .. "/" .. "VOC2007" .. "/" .. "VOC2007_" .. "landscape_192X256.pvp" --"portrait_256X192.pvp" -- 
local startFrame          = 0

--HyPerCol parameters
local dtAdaptFlag              = not S1_Movie
local useAdaptMethodExp1stOrder = true
local dtAdaptController        = "S1EnergyProbe"
local dtAdaptTriggerLayerName  = "Image";
local dtScaleMax               = 0.25   --1.0     -- minimum value for the maximum time scale, regardless of tau_eff
local dtScaleMin               = 0.01  --0.01    -- default time scale to use after image flips or when something is wacky
local dtChangeMax              = 0.1   --0.1     -- determines fraction of tau_effective to which to set the time step, can be a small percentage as tau_eff can be huge
local dtChangeMin              = 0.01  --0.01    -- percentage increase in the maximum allowed time scale whenever the time scale equals the current maximum
local dtMinToleratedTimeScale  = 0.0001

--Ground Truth parameters
local numClasses            = 20
local nxScale_GroundTruth   = 0.015625 --0.03125 --0.0625;
local nyScale_GroundTruth   = 0.015625 --0.03125 --0.0625;


-- Base table variable to store
local pvParams = {
   column = {
      groupType                           = "HyPerCol"; 
      startTime                           = 0;
      dt                                  = 1;
      dtAdaptFlag                         = dtAdaptFlag;
      useAdaptMethodExp1stOrder           = useAdaptMethodExp1stOrder;
      dtAdaptController                   = dtAdaptController;
      dtAdaptTriggerLayerName             = dtAdaptTriggerLayerName;
      dtScaleMax                          = dtScaleMax;    
      dtScaleMin                          = dtScaleMin;
      dtChangeMax                         = dtChangeMax;
      dtChangeMin                         = dtChangeMin;
      dtMinToleratedTimeScale             = dtMinToleratedTimeScale;
      stopTime                            = stopTime;
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
      checkpointWriteStepInterval         = checkpointWriteStepInterval;
      deleteOlderCheckpoints              = false;
      suppressNonplasticCheckpoints       = false;
      writeTimescales                     = true;
      errorOnNotANumber                   = false;
   }
} --End of pvParams
if S1_Movie then
   pvParams.column.dtAdaptFlag                         = false;
   pvParams.column.dtAdaptController                   = nil;
   pvParams.column.dtAdaptTriggerLayerName             = nil;
   pvParams.column.dtScaleMax                          = nil;
   pvParams.column.dtScaleMin                          = nil;
   pvParams.column.dtChangeMax                         = nil;
   pvParams.column.dtChangeMin                         = nil;
   pvParams.column.dtMinToleratedTimeScale             = nil;


   local GroundTruthMoviePath                         = inputPath .. "/" .. "GroundTruth.pvp"
   pv.addGroup(pvParams, "GroundTruth",
	       {
		  groupType = "MoviePvp";
		  nxScale                             = nxScale_GroundTruth;
		  nyScale                             = nyScale_GroundTruth;
		  nf                                  = numClasses + 1;
		  phase                               = 0;
		  mirrorBCflag                        = true;
		  initializeFromCheckpointFlag        = false;
		  writeStep                           = writeStep;
		  initialWriteTime                    = initialWriteTime;
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
		  inputPath                           = GroundTruthMoviePath;
		  displayPeriod                       = displayPeriod;
		  randomMovie                         = 0;
		  readPvpFile                         = true;
		  start_frame_index                   = startFrame;
		  skip_frame_index                    = 0;
		  writeFrameToTimestamp               = true;
		  flipOnTimescaleError                = true;
		  resetToStartOnLoop                  = false;
	       }
   )

   local S1MoviePath                                   = inputPath .. "/" .. "S1.pvp"
   pv.addGroup(pvParams, "S1",
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
		  start_frame_index                   = startFrame;
		  skip_frame_index                    = 0;
		  writeFrameToTimestamp               = true;
		  flipOnTimescaleError                = true;
		  resetToStartOnLoop                  = false;
	       }
   )
else
   -- pv.addGroup(pvParams, "ConstantS1",
   -- 	       {
   -- 		  groupType = "ConstantLayer";
   -- 		  nxScale                             = 1.0/stride;
   -- 		  nyScale                             = 1.0/stride;
   -- 		  nf                                  = S1_numFeatures;
   -- 		  phase                               = 0;
   -- 		  mirrorBCflag                        = false;
   -- 		  valueBC                             = 0;
   -- 		  initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
   -- 		  InitVType                           = "ConstantV";
   -- 		  valueV                              = VThresh;
   -- 		  writeStep                           = -1;
   -- 		  sparseLayer                         = false;
   -- 		  updateGpu                           = false;
   -- 		  dataType                            = nil;
   -- 		  VThresh                             = -inf;
   -- 		  AMin                                = -inf;
   -- 		  AMax                                = inf;
   -- 		  AShift                              = 0;
   -- 		  VWidth                              = 0;
   -- 		  clearGSynInterval                   = 0;
   -- 	       }
   -- )

   pv.addGroup(pvParams, "Image", 
	       {
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
		  inputPath                           = imageListPath;
		  displayPeriod                       = displayPeriod;
		  echoFramePathnameFlag               = true;
		  start_frame_index                   = startFrame;
		  skip_frame_index                    = 0;
		  writeFrameToTimestamp               = true;
		  flipOnTimescaleError                = true;
		  resetToStartOnLoop                  = false;
	       }
   )

   pv.addGroup(pvParams, "ImageReconS1Error",
	       {
		  groupType = "ANNLayer";
		  nxScale                             = 1;
		  nyScale                             = 1;	
		  nf                                  = 3;
		  phase                               = 1;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
		  InitVType                           = "ZeroV";
		  triggerLayerName                    = NULL;
		  writeStep                           = writeStep;
		  initialWriteTime                    = initialWriteTime;
		  sparseLayer                         = false;
		  updateGpu                           = false;
		  dataType                            = nil;
		  VThresh                             = -inf;
		  AMin                                = -inf;
		  AMax                                = inf;
		  AShift                              = 0;
		  VWidth                              = 0;
		  clearGSynInterval                   = 0;
		  errScale                            = 1;
	       }
   ) 

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
		  -- InitVType                           = "UniformRandomV";
		  -- minV                                = -1;
		  -- maxV                                = 0.05;
		  InitVType                           = "ConstantV";
		  valueV                              = VThresh;
		  --triggerLayerName                    = "Image";
		  --triggerBehavior                     = "resetStateOnTrigger";
		  --triggerResetLayerName               = "ConstantS1";
		  --triggerOffset                       = 0;
		  writeStep                           = displayPeriod;
		  initialWriteTime                    = displayPeriod;
		  sparseLayer                         = true;
		  writeSparseValues                   = true;
		  updateGpu                           = true;
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
		  writeStep                           = writeStep;
		  initialWriteTime                    = initialWriteTime;
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
		  start_frame_index                   = startFrame;
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
   
end

-- Ground Truth 


pv.addGroup(pvParams, "GroundTruthReconS1Error",
	    {
	       groupType                           = "ANNLayer";
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
	       writeStep                           = writeStep;
	       initialWriteTime                    = initialWriteTime;
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
if S1_Movie then
   pvParams.GroundTruthReconS1Error.triggerLayerName  = "GroundTruth";
   pvParams.GroundTruthReconS1Error.triggerOffset  = 0;
end

pv.addGroup(pvParams, "GroundTruthReconS1",
	    {
	       groupType                           = "ANNLayer";
	       nxScale                             = nxScale_GroundTruth;
	       nyScale                             = nyScale_GroundTruth;
	       nf                                  = numClasses + 1;
	       phase                               = 8;
	       mirrorBCflag                        = false;
	       valueBC                             = 0;
	       initializeFromCheckpointFlag        = false;
	       InitVType                           = "ZeroV";
	       triggerLayerName                    = "GroundTruthPixels";
	       triggerBehavior                     = "updateOnlyOnTrigger";
	       triggerOffset                       = 1;
	       writeStep                           = writeStep;
	       initialWriteTime                    = initialWriteTime;
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
if S1_Movie then
   pvParams.GroundTruthReconS1.triggerLayerName  = "GroundTruth";
   pvParams.GroundTruthReconS1.triggerOffset  = 0;
end

pv.addGroup(pvParams, "S1MaxPooled1X1", pvParams.GroundTruthReconS1,
	    {
	       nf                                  = S1_numFeatures;
	       writeStep                           = -1;
	       VThresh                             = 0.0;
	       AMin                                = 0.0;
	       phase                               = 7;
	    }
)
pv.addGroup(pvParams, "S1MaxPooledIndex1X1", 
	    {
	       groupType                           = "PoolingIndexLayer";
	       nxScale                             = nxScale_GroundTruth;
	       nyScale                             = nyScale_GroundTruth;
	       nf                                  = S1_numFeatures;
	       triggerLayerName                    = "GroundTruthPixels";
	       triggerBehavior                     = "updateOnlyOnTrigger";
	       triggerOffset                       = 1;
	       writeStep                           = -1;
	       sparseLayer                         = false;
	       dataType                            = nil;
	       clearGSynInterval                   = 0;
	       phase                               = 8;
	    }
)
if S1_Movie then
   pvParams.S1MaxPooledIndex1X1.triggerLayerName  = "GroundTruth";
   pvParams.S1MaxPooledIndex1X1.triggerOffset  = 0;
end

pv.addGroup(pvParams, "S1Mask1X1", pvParams.S1MaxPooled1X1,
	    {
	       phase                               =  8;
	    }
)
pvParams.S1Mask1X1.groupType                       = "PtwiseLinearTransferLayer";
pvParams.S1Mask1X1.VThresh                         = nil;
pvParams.S1Mask1X1.AMin                            = nil;
pvParams.S1Mask1X1.AMax                            = nil;
pvParams.S1Mask1X1.AShift                          = nil;
pvParams.S1Mask1X1.VWidth                          = nil;
pvParams.S1Mask1X1.verticesA                       = {0.0, 1.0};
pvParams.S1Mask1X1.verticesV                       = {0.0, 0.0};
pvParams.S1Mask1X1.slopeNegInf                     = 0.0;
pvParams.S1Mask1X1.slopePosInf                     = 0.0;

pv.addGroup(pvParams, "S1Error1X1", pvParams.S1MaxPooled1X1,
	    {
	       groupType                           = "PtwiseProductLayer";
	       VThresh                             = -inf;
	       AMin                                = -inf;
	       phase                               = 10;
	    }
)

pv.addGroup(pvParams, "S1MaxPooled2X2", pvParams.S1MaxPooled1X1,
	    {
	       nxScale                             = nxScale_GroundTruth*2;
	       nyScale                             = nyScale_GroundTruth*2;
	       phase                               = 5;
	    }
)
pv.addGroup(pvParams, "S1MaxPooledIndex2X2", pvParams.S1MaxPooledIndex1X1,
	    {
	       nxScale                             = nxScale_GroundTruth*2;
	       nyScale                             = nyScale_GroundTruth*2;
	       phase                               = 6;
	    }
)
pv.addGroup(pvParams, "S1Hidden2X2", pvParams.S1MaxPooled1X1,
	    {
	       nxScale                             = nxScale_GroundTruth*2;
	       nyScale                             = nyScale_GroundTruth*2;
	       phase                               = 6;
	    }
)
pv.addGroup(pvParams, "S1Mask2X2", pvParams.S1Mask1X1,
	    {
	       nxScale                             = nxScale_GroundTruth*2;
	       nyScale                             = nyScale_GroundTruth*2;
	       phase                               = 6;
	    }
)
pv.addGroup(pvParams, "S1UnPooled2X2", pvParams.S1Error1X1,
	    {
	       nxScale                             = nxScale_GroundTruth*2;
	       nyScale                             = nyScale_GroundTruth*2;
	       phase                               = 11;
	    }
)
pv.addGroup(pvParams, "S1Error2X2", pvParams.S1Error1X1,
	    {
	       nxScale                             = nxScale_GroundTruth*2;
	       nyScale                             = nyScale_GroundTruth*2;
	       phase                               = 12;
	    }
)


pv.addGroup(pvParams, "S1MaxPooled4X4", pvParams.S1MaxPooled2X2,
	    {
	       nxScale                             = nxScale_GroundTruth*4;
	       nyScale                             = nyScale_GroundTruth*4;
	       phase                               = 3;
	    }
)
pv.addGroup(pvParams, "S1MaxPooledIndex4X4", pvParams.S1MaxPooledIndex2X2,
	    {
	       nxScale                             = nxScale_GroundTruth*4;
	       nyScale                             = nyScale_GroundTruth*4;
	       phase                               = 4;
	    }
)
pv.addGroup(pvParams, "S1Hidden4X4", pvParams.S1Hidden2X2,
	    {
	       nxScale                             = nxScale_GroundTruth*4;
	       nyScale                             = nyScale_GroundTruth*4;
	       phase                               = 4;
	    }
)
pv.addGroup(pvParams, "S1Mask4X4", pvParams.S1Mask2X2,
	    {
	       nxScale                             = nxScale_GroundTruth*4;
	       nyScale                             = nyScale_GroundTruth*4;
	       phase                               = 4;
	    }
)
pv.addGroup(pvParams, "S1UnPooled4X4", pvParams.S1UnPooled2X2,
	    {
	       nxScale                             = nxScale_GroundTruth*4;
	       nyScale                             = nyScale_GroundTruth*4;
	       phase                               = 13;
	    }
)
pv.addGroup(pvParams, "S1Error4X4", pvParams.S1Error2X2,
	    {
	       nxScale                             = nxScale_GroundTruth*4;
	       nyScale                             = nyScale_GroundTruth*4;
	       phase                               = 14;
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



--connections

if not S1_Movie then
   pv.addGroup(pvParams, "ImageToImageReconS1Error",
	       {
		  groupType                           = "HyPerConn";
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
		  weightInit                          = math.sqrt((1/patchSize)*(1/patchSize)*(1/3)); --
		  initializeFromCheckpointFlag        = false;
		  updateGSynFromPostPerspective       = false;
		  pvpatchAccumulateType               = "convolve";
		  writeStep                           = -1;
		  writeCompressedCheckpoints          = false;
		  selfFlag                            = false;
		  nxp                                 = 1;
		  nyp                                 = 1;
		  shrinkPatches                       = false;
		  normalizeMethod                     = "none";
	       }
   )

   pv.addGroup(pvParams, "ImageReconS1ToImageReconS1Error",
	       {
		  groupType                           = "IdentConn";
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
		  groupType                           = "MomentumConn";
		  preLayerName                        = "S1";
		  postLayerName                       = "ImageReconS1Error";
		  channelCode                         = -1;
		  delay                               = {0.000000};
		  numAxonalArbors                     = 1;
		  plasticityFlag                      = plasticityFlag;
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
		  updateGSynFromPostPerspective       = false; 
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
if not plasticityFlag then
   pvParams.S1ToImageReconS1Error.triggerLayerName    = NULL;
   pvParams.S1ToImageReconS1Error.triggerOffset       = nil;
end


   pv.addGroup(pvParams, "ImageReconS1ErrorToS1",
	       {
		  groupType                           = "TransposeConn";
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
		  groupType                           = "CloneConn";
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

   pv.addGroup(pvParams, "GroundTruthPixelsToGroundTruthNoBackground",
	       {
		  groupType                           = "PoolingConn";
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

end

-- Ground Truth connections

pv.addGroup(pvParams, "GroundTruthToGroundTruthReconS1Error",
	    {
	       groupType                           = "IdentConn";
	       preLayerName                        = "GroundTruth";
	       postLayerName                       = "GroundTruthReconS1Error";
	       channelCode                         = 0;
	       delay                               = {0.000000};
	       initWeightsFile                     = nil;
	       writeStep                           = -1;
	    }
)

pv.addGroup(pvParams, "S1ToS1MaxPooled4X4",
	    {
	       groupType                           = "PoolingConn";
	       preLayerName                        = "S1";
	       postLayerName                       = "S1MaxPooled4X4";
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
	       needPostIndexLayer                  = true;
	       postIndexLayerName                  = "S1MaxPooledIndex4X4";
	    }
)

pv.addGroup(pvParams, "S1MaxPooled4X4ToS1Hidden4X4",
	    {
	       groupType                           = "CloneConn";
	       preLayerName                        = "S1MaxPooled4X4";
	       postLayerName                       = "S1Hidden4X4";
	       channelCode                         = 0;
	       delay                               = {0.000000};
	       convertRateToSpikeCount             = false;
	       receiveGpu                          = true;
	       updateGSynFromPostPerspective       = true;
	       pvpatchAccumulateType               = "convolve";
	       writeStep                           = -1;
	       writeCompressedCheckpoints          = false;
	       selfFlag                            = false;
	       originalConnName                    = "S1MaxPooled4X4ToS1Error4X4";
	    }
)


pv.addGroup(pvParams, "S1Hidden4X4ToS1MaxPooled2X2", pvParams.S1ToS1MaxPooled4X4,
	    {
	       preLayerName                        = "S1Hidden4X4";
	       postLayerName                       = "S1MaxPooled2X2";
	       postIndexLayerName                  = "S1MaxPooledIndex2X2";
	    }
)

pv.addGroup(pvParams, "S1MaxPooled2X2ToS1Hidden2X2", pvParams.S1MaxPooled4X4ToS1Hidden4X4,
	    {
	       preLayerName                        = "S1MaxPooled2X2";
	       postLayerName                       = "S1Hidden2X2";
	       originalConnName                    = "S1MaxPooled2X2ToS1Error2X2";
	    }
)


pv.addGroup(pvParams, "S1Hidden2X2ToS1MaxPooled1X1", pvParams.S1ToS1MaxPooled4X4,
	    {
	       preLayerName                        = "S1Hidden2X2";
	       postLayerName                       = "S1MaxPooled1X1";
	       postIndexLayerName                  = "S1MaxPooledIndex1X1";
	    }
)

pv.addGroup(pvParams, "S1MaxPooled1X1ToGroundTruthReconS1", pvParams.S1MaxPooled4X4ToS1Hidden4X4,
	    {
	       preLayerName                        = "S1MaxPooled1X1";
	       postLayerName                       = "GroundTruthReconS1";
	       originalConnName                    = "S1MaxPooled1X1ToGroundTruthReconS1Error";
	    }
)

pv.addGroup(pvParams, "BiasS1ToGroundTruthReconS1", pvParams.S1MaxPooled4X4ToS1Hidden4X4,
	    {
	       preLayerName                        = "BiasS1";
	       postLayerName                       = "GroundTruthReconS1";
	       originalConnName                    = "BiasS1ToGroundTruthReconS1Error";
	    }
)

pv.addGroup(pvParams, "GroundTruthReconS1ToGroundTruthReconS1Error", 
	    {
	       groupType                           = "IdentConn";
	       preLayerName                        = "GroundTruthReconS1";
	       postLayerName                       = "GroundTruthReconS1Error";
	       channelCode                         = 1;
	       delay                               = {0.000000};
	       initWeightsFile                     = nil;
	       writeStep                           = -1;
	    }
)

pv.addGroup(pvParams, "S1MaxPooled1X1ToGroundTruthReconS1Error",
	    {
	       groupType                           = "HyPerConn";
	       preLayerName                        = "S1MaxPooled1X1";
	       postLayerName                       = "GroundTruthReconS1Error";
	       channelCode                         = -1;
	       delay                               = {0.000000};
	       numAxonalArbors                     = 1;
	       plasticityFlag                      = plasticityFlag;
	       convertRateToSpikeCount             = false;
	       receiveGpu                          = true; --false;
	       sharedWeights                       = true;
	       weightInitType                      = "FileWeight";
	       initWeightsFile                     = inputPathSLP .. "/Checkpoints/Checkpoint" .. checkpointIDSLP .. "/S1MaxPooled1X1ToGroundTruthReconS1Error_W.pvp";
	       --initWeightsFile                     = inputPathSLP .. "/Checkpoints/Checkpoint" .. checkpointIDSLP .. "/S1MaxPooledToGroundTruthReconS1Error_W.pvp";
	       useListOfArborFiles                 = false;
	       combineWeightFiles                  = false;    
	       --weightInitType                      = "UniformRandomWeight";
	       --initWeightsFile                     = nil;
	       --wMinInit                            = -0;
	       --wMaxInit                            = 0;
	       --sparseFraction                      = 0;
	       initializeFromCheckpointFlag        = false;
	       triggerLayerName                    = "Image";
	       triggerBehavior                     = "updateOnlyOnTrigger";
	       triggerOffset                       = 1;
	       updateGSynFromPostPerspective       = true;
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
	       -- momentumTau                         = 1;
	       -- momentumMethod                      = "viscosity";
	       -- momentumDecay                       = 0;
	       Batchperiod                         = 1;
	    }
)
if S1_Movie then
   pvParams.S1MaxPooled1X1ToGroundTruthReconS1Error.triggerLayerName  = "GroundTruth";
   pvParams.S1MaxPooled1X1ToGroundTruthReconS1Error.triggerOffset     = 0;
end

pv.addGroup(pvParams, "BiasS1ToGroundTruthReconS1Error",
	    pvParams.S1MaxPooled1X1ToGroundTruthReconS1Error,
	    {
	       preLayerName                        = "BiasS1";
	       postLayerName                       = "GroundTruthReconS1Error";
	       initWeightsFile                     = inputPathSLP .. "/Checkpoints/Checkpoint" .. checkpointIDSLP .. "/BiasS1ToGroundTruthReconS1Error_W.pvp";
	    }
)
if plasticityFlag then
   pvParams.BiasS1ToGroundTruthReconS1Error.dWMax = 0.01;
end

if not plasticityFlag then
   pvParams.S1MaxPooled1X1ToGroundTruthReconS1Error.plasticityFlag     = false;
   pvParams.S1MaxPooled1X1ToGroundTruthReconS1Error.triggerLayerName   = NULL;
   pvParams.S1MaxPooled1X1ToGroundTruthReconS1Error.triggerOffset      = nil;
   pvParams.S1MaxPooled1X1ToGroundTruthReconS1Error.triggerBehavior    = nil;
   pvParams.S1MaxPooled1X1ToGroundTruthReconS1Error.dWMax              = nil;
   pvParams.BiasS1ToGroundTruthReconS1Error.plasticityFlag             = false;
   pvParams.BiasS1ToGroundTruthReconS1Error.triggerLayerName           = NULL;
   pvParams.BiasS1ToGroundTruthReconS1Error.triggerOffset              = nil;
   pvParams.BiasS1ToGroundTruthReconS1Error.triggerBehavior            = nil;
   pvParams.BiasS1ToGroundTruthReconS1Error.dWMax                      = nil;
end

pv.addGroup(pvParams, "GroundTruthReconS1ErrorToS1Error1X1", 
	    {
	       groupType                           = "TransposeConn";
	       preLayerName                        = "GroundTruthReconS1Error";
	       postLayerName                       = "S1Error1X1";
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
	       originalConnName                    = "S1MaxPooled1X1ToGroundTruthReconS1Error";
	    }
)

pv.addGroup(pvParams, "S1Mask1X1ToS1Error1X1",
	    {
	       groupType                           = "IdentConn";
	       preLayerName                        = "S1Mask1X1";
	       postLayerName                       = "S1Error1X1";
	       channelCode                         = 1;
	       delay                               = {0.000000};
	       initWeightsFile                     = nil;
	       writeStep                           = -1;
	    }
)

pv.addGroup(pvParams, "S1MaxPooled1X1ToS1Mask1X1",
	    {
	       groupType                           = "IdentConn";
	       preLayerName                        = "S1MaxPooled1X1";
	       postLayerName                       = "S1Mask1X1";
	       channelCode                         = 0;
	       delay                               = {0.000000};
	       initWeightsFile                     = nil;
	       writeStep                           = -1;
	    }
)

pv.addGroup(pvParams, "S1Error1X1ToS1UnPooled2X2",
	    {
	       groupType                           = "TransposePoolingConn";
	       preLayerName                        = "S1Error1X1";
	       postLayerName                       = "S1UnPooled2X2";
	       channelCode                         = 0;
	       updateGSynFromPostPerspective       = false;
	       pvpatchAccumulateType               = "maxpooling";
	       writeStep                           = -1;
	       writeCompressedCheckpoints          = false;
	       selfFlag                            = false;
	       delay                               = 0;
	       originalConnName                    = "S1Hidden2X2ToS1MaxPooled1X1";
	    }
)

pv.addGroup(pvParams, "S1UnPooled2X2ToS1Error2X2",
	    pvParams.GroundTruthReconS1ErrorToS1Error1X1,
	    {
	       preLayerName                        = "S1UnPooled2X2";
	       postLayerName                       = "S1Error2X2";
	       originalConnName                    = "S1MaxPooled2X2ToS1Error2X2";
	    }
)

pv.addGroup(pvParams, "S1Mask2X2ToS1Error2X2",
	    {
	       groupType                           = "IdentConn";
	       preLayerName                        = "S1Mask2X2";
	       postLayerName                       = "S1Error2X2";
	       channelCode                         = 1;
	       delay                               = {0.000000};
	       initWeightsFile                     = nil;
	       writeStep                           = -1;
	    }
)

pv.addGroup(pvParams, "S1MaxPooled2X2ToS1Mask2X2",
	    {
	       groupType                           = "IdentConn";
	       preLayerName                        = "S1MaxPooled2X2";
	       postLayerName                       = "S1Mask2X2";
	       channelCode                         = 0;
	       delay                               = {0.000000};
	       initWeightsFile                     = nil;
	       writeStep                           = -1;
	    }
)

pv.addGroup(pvParams, "S1MaxPooled2X2ToS1Error2X2",
	    pvParams.S1MaxPooled1X1ToGroundTruthReconS1Error,
	    {
	       preLayerName                        = "S1MaxPooled2X2";
	       postLayerName                       = "S1Error2X2";
	       initWeightsFile                     = inputPathSLP .. "/Checkpoints/Checkpoint" .. checkpointIDSLP .. "/S1MaxPooled2X2ToS1Error2X2_W.pvp";
	    }
)
local identityFlag2X2  = false --true
if identityFlag2X2 then
   pvParams.S1MaxPooled2X2ToS1Error2X2.plasticityFlag     = false;
   pvParams.S1MaxPooled2X2ToS1Error2X2.triggerLayerName   = NULL;
   pvParams.S1MaxPooled2X2ToS1Error2X2.triggerOffset      = nil;
   pvParams.S1MaxPooled2X2ToS1Error2X2.triggerBehavior    = nil;
   pvParams.S1MaxPooled2X2ToS1Error2X2.dWMax              = nil;
   pvParams.S1MaxPooled2X2ToS1Error2X2.weightInitType     = "OneToOneWeights";
   pvParams.S1MaxPooled2X2ToS1Error2X2.weightInit         = 1.0;
   pvParams.S1MaxPooled2X2ToS1Error2X2.initWeightsFile    = NULL;
end

pv.addGroup(pvParams, "S1Error2X2ToS1UnPooled4X4",
	    pvParams.S1Error1X1ToS1UnPooled2X2,
	    {
	       preLayerName                        = "S1Error2X2";
	       postLayerName                       = "S1UnPooled4X4";
	       originalConnName                    = "S1Hidden4X4ToS1MaxPooled2X2";
	    }
)

pv.addGroup(pvParams, "S1UnPooled4X4ToS1Error4X4",
	    pvParams.S1UnPooled2X2ToS1Error2X2,
	    {
	       preLayerName                        = "S1UnPooled4X4";
	       postLayerName                       = "S1Error4X4";
	       originalConnName                    = "S1MaxPooled4X4ToS1Error4X4";
	    }
)

pv.addGroup(pvParams, "S1Mask4X4ToS1Error4X4", pvParams.S1Mask2X2ToS1Error2X2, 
	    {
	       preLayerName                        = "S1Mask4X4";
	       postLayerName                       = "S1Error4X4";
	    }
)

pv.addGroup(pvParams, "S1MaxPooled4X4ToS1Mask4X4", pvParams.S1MaxPooled2X2ToS1Mask2X2, 
	    {
	       preLayerName                        = "S1MaxPooled4X4";
	       postLayerName                       = "S1Mask4X4";
	    }
)

pv.addGroup(pvParams, "S1MaxPooled4X4ToS1Error4X4",
	    pvParams.S1MaxPooled2X2ToS1Error2X2,
	    {
	       preLayerName                        = "S1MaxPooled4X4";
	       postLayerName                       = "S1Error4X4";
	       initWeightsFile                     = inputPathSLP .. "/Checkpoints/Checkpoint" .. checkpointIDSLP .. "/S1MaxPooled4X4ToS1Error4X4_W.pvp";
	    }
)
local identityFlag4X4  = true
if identityFlag4X4 then
   pvParams.S1MaxPooled4X4ToS1Error4X4.plasticityFlag     = false;
   pvParams.S1MaxPooled4X4ToS1Error4X4.triggerLayerName   = NULL;
   pvParams.S1MaxPooled4X4ToS1Error4X4.triggerOffset      = nil;
   pvParams.S1MaxPooled4X4ToS1Error4X4.triggerBehavior    = nil;
   pvParams.S1MaxPooled4X4ToS1Error4X4.dWMax              = nil;
   pvParams.S1MaxPooled4X4ToS1Error4X4.weightInitType     = "OneToOneWeights";
   pvParams.S1MaxPooled4X4ToS1Error4X4.weightInit         = 1.0;
   pvParams.S1MaxPooled4X4ToS1Error4X4.initWeightsFile    = NULL;
end


-- Energy probe

if not S1_Movie then
   
   
   --pv.addGroup(pvParams, "ImageEnergyProbe", 
   --	       {
   --		  groupType                           = "ColumnEnergyProbe";
   --		  probeOutputFile                     = "ImageEnergyProbe.txt";
   --	       }
   --)
   
   --pv.addGroup(pvParams, "TimeScaleProbe",
   --	       {
   --		  groupType                           = "QuotientColProbe";
   --		  numerator                           = "S1EnergyProbe";
   --		  denominator                         = "ImageEnergyProbe";
   --		  probeOutputFile                     = "TimeScaleProbe.txt";
   --		  valueDescription                    = "TimeScale";
   --	       }
   --)


   pv.addGroup(pvParams, "S1EnergyProbe", 
	       {
		  groupType                           = "ColumnEnergyProbe";
		  probeOutputFile                     = "S1EnergyProbe.txt";
	       }
   )
   
   pv.addGroup(pvParams, "ImageReconS1ErrorL2NormEnergyProbe",
	       {
		  groupType                           = "L2NormProbe";
		  targetLayer                         = "ImageReconS1Error";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "ImageReconS1ErrorL2NormEnergyProbe.txt";
		  triggerLayerName                    = NULL; --"Image";
		  --triggerOffset                       = 1;
		  energyProbe                         = "S1EnergyProbe";
		  coefficient                         = 0.5;
		  maskLayerName                       = NULL;
		  exponent                            = 2;
	       }
   )

   pv.addGroup(pvParams, "S1FirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S1";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S1FirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "S1EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   --pv.addGroup(pvParams, "ImageL2NormEnergyProbe", 
   --	       {
   --		  groupType                           = "L2NormProbe";
   --		  targetLayer                         = "Image";
   --		  textOutputFlag                      = true;
   --		  message                             = NULL;
   --		  probeOutputFile                     = "ImageL2NormEnergyProbe.txt";
   --		  energyProbe                         = "ImageEnergyProbe";
   --		  triggerLayerName                    = "Image";
   --		  triggerOffset                       = 0;
   --		  coefficient                         = 0.5;
   --		  maskLayerName                       = NULL;
   --		  exponent                            = 2;
   --	       }
   --)

   
end


-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParams)
