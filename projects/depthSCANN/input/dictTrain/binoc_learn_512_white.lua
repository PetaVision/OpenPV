-------------------------------------------------------------------
-- Learning binocular dictionaries with SCANN
--
-- Sheng Lundquist 6/9/15
--
-- Implements binocular SCANN model as described in
-- Lundquist et al., "Emergence of Depth-Tuned Hidden Units Through
-- Sparse Encoding of Binocular Images"
-------------------------------------------------------------------

--Util module
package.path = package.path .. ";" .. os.getenv("HOME") .. "/workspace/PetaVision/parameterWrapper/?.lua;"
local pv = require("PVModule")

------------------------------------------------
-- Parameters --
------------------------------------------------
-- User defined variables
local nxSize = 1200 --Size of image
local nySize = 360 
local outputPath = "/home/ec2-user/output/dictTrain/binoc_512_white/" --Output directory
local writePeriod = 200 --How often to write out
local checkpointWriteStepInterval = 50000 --How often to checkpoint
local stopTime= 100000000 --Stopping time
local progressStep = 5000; --How often to print out a status report

--Image parameters
local leftImageListPath = "/home/ec2-user/dataset/list/raw_left.txt" --List of images, raw
local rightImageListPath = "/home/ec2-user/dataset/list/raw_right.txt"
local displayPeriod = 200 --Display period for sparse approximation
local startFrame = 1 --Starting at frame 1

--LCA parameters
local stride = 2 --Stride of LCA
local numDictElements = 512 --Total number of dictionary elements
local dictPatchSize = 66 --Square patch, with this as the dimension
local VThresh = 0.006 --Threshold parameter
local VWidth = .05 --Threshold parameter

local learningRate = 0.05 --Learning rate
local useMomentum = true  --Using momentum
local learningMomentumTau = 100; --Learning Omega

--nil for new dict, a checkpoint directory for loading weights
local V1DictDir = nil
--local V1DictDir = "/home/ec2-user/saved_output/dictTrain/saved_binoc_512_white/"
------------------------------------------------


local pvParams = {
   --base column
   column = {
      groupType = "HyPerCol";
      nx = nxSize;
      ny = nySize;
      dt = 1.0;
      randomSeed = 1234567890;
      startTime = 0;
      stopTime= stopTime;
      progressStep = progressStep;
      outputPath = outputPath;
      filenamesContainLayerNames = 2;
      filenamesContainConnectionNames = 2;
      checkpointRead = false;
      checkpointWrite = true;
      checkpointWriteDir = outputPath .. "/Checkpoints";
      checkpointWriteStepInterval = checkpointWriteStepInterval;
      deleteOlderCheckpoints = true;
      writeProgressToErr = true;
      outputNamesOfLayersAndConns = "LayerAndConnNames.txt";
      dtAdaptFlag = true;
      dtScaleMax = 5.0;
      dtScaleMin = 0.25;
      dtChangeMax = 0.05;
      dtChangeMin = 0.0;
   };

   --Left layers
   LeftImage = {
      groupType = "Movie";
      restart = 0;
      nxScale = 1;
      nyScale = 1;
      readPvpFile = false;
      imageListPath = leftImageListPath;
      writeFrameToTimestamp = true;
      nf = 1;
      writeStep = writePeriod;
      initialWriteTime = writePeriod;
      writeSparseActivity = false;
      displayPeriod = displayPeriod;
      start_frame_index = startFrame;
      skip_frame_index = 1;
      echoFramePathnameFlag = false;
      mirrorBCflag = true;
      jitterFlag = 0;
      useImageBCflag = false;
      inverseFlag = false;
      normalizeLuminanceFlag = true;
      writeImages = false;
      offsetAnchor = "br"; 
      offsetX = 0;
      offsetY = 0;
      autoResizeFlag = 0;
      randomMovie = 0;
      phase = 0;
   };

   LeftBipolar = {
      groupType = "ANNLayer";
      nxScale = .5;
      nyScale = .5;
      nf = 1;
      phase = 1;
      mirrorBCflag = true;
      InitVType = "ZeroV";
      triggerFlag = true;
      triggerLayerName = "LeftImage";
      triggerOffset = 0;
      writeStep = -1;
      sparseLayer = false;
      updateGpu = false;
      VThresh = -3.40282e+38;
      AMin = -3.40282e+38;
      AMax = 3.40282e+38;
      AShift = 0;
      VWidth = 0;
   };

   LeftGanglion = {
      groupType = "ANNLayer";
      nxScale = .5;
      nyScale = .5;
      nf = 1;
      phase = 2;
      mirrorBCflag = true;
      InitVType = "ZeroV";
      triggerFlag = true;
      triggerLayerName = "LeftImage";
      triggerOffset = 0;
      writeStep = -1;
      initialWriteTime = writePeriod;
      sparseLayer = false;
      updateGpu = false;
      VThresh = -3.40282e+38;
      AMin = -3.40282e+38;
      AMax = 3.40282e+38;
      AShift = 0;
      VWidth = 0;
   };

   LeftRescale = {
      groupType = "RescaleLayer";
      restart = false;
      originalLayerName = "LeftGanglion";
      nxScale = .5;
      nyScale = .5;
      nf = 1;
      mirrorBCflag = true;
      writeStep = writePeriod;
      initialWriteTime = writePeriod;
      writeSparseActivity = false;
      rescaleMethod = "l2";
      patchSize = dictPatchSize * dictPatchSize;
      valueBC = 0;
      phase = 3;
      triggerFlag = true;
      triggerLayerName = "LeftImage";
   };

   LeftError = {
      groupType = "ANNNormalizedErrorLayer";
      restart = 0;
      nxScale = .5;
      nyScale = .5;
      nf = 1;
      writeStep = writePeriod;
      initialWriteTime = writePeriod;
      mirrorBCflag = 0;
      writeSparseActivity = 0;
      InitVType = "ZeroV";
      VThresh = 0;
      VMax =  INFINITY;
      VMin = -INFINITY;
      VShift = 0;
      VWidth = 0;
      valueBC = 0;
      errScale = 1;
      phase = 4;  
   };

   LeftRecon = {
      groupType = "ANNLayer";
      restart = 0;
      nxScale = .5;
      nyScale = .5;
      nf = 1;
      writeStep = writePeriod;
      initialWriteTime = writePeriod;
      mirrorBCflag = 0;
      writeSparseActivity = 0;
      InitVType = "ZeroV";
      VThresh = -INFINITY;
      VMax = INFINITY;
      VMin = -INFINITY; 
      VShift = 0;
      VWidth = 0;
      valueBC = 0;
      phase = 6;
      triggerFlag = 1;
      triggerLayerName = "LeftImage";
      triggerOffset = 1;
   };
};

--Right eye layers
pv.addGroup(pvParams, "RightImage", pvParams["LeftImage"],
   {
      imageListPath = rightImageListPath;
   }
)

pv.addGroup(pvParams, "RightBipolar", pvParams["LeftBipolar"],
   {
      triggerLayerName = "RightImage";
   }
)

pv.addGroup(pvParams, "RightGanglion", pvParams["LeftGanglion"],
   {
      triggerLayerName = "RightImage";
   }
)

pv.addGroup(pvParams, "RightRescale", pvParams["LeftRescale"],
   {
      originalLayerName = "RightGanglion";
   }
)

pv.addGroup(pvParams, "RightError", pvParams["LeftError"])

pv.addGroup(pvParams, "RightRecon", pvParams["LeftRecon"],
   {
      triggerLayerName = "RightImage";
   }
)

pv.addGroup(pvParams, "V1",
{
   groupType = "HyPerLCALayer";
   restart = 0;
   nxScale = .5/stride;
   nyScale = .5/stride;
   nf = numDictElements;
   numChannels = 1;
   numWindowX = 1;
   numWindowY = 1;
   writeStep = writePeriod;
   initialWriteTime = writePeriod;
   mirrorBCflag = 0;
   writeSparseActivity = 1;
   writeSparseValues   = 1;
   InitVType = "UniformRandomV";
   minV = -1.0;
   maxV = .02; 
   timeConstantTau = 200.0;
   slopeErrorStd = 0.01;
   dVThresh = 0;
   VThresh = VThresh;
   VMax = infinity;
   VMin = 0;
   VShift = 0.0;  
   VWidth = VWidth; 
   updateGpu = true;
   phase = 5;
});

--Connections
pv.addMultiGroups(pvParams, 
   {
   LeftImageToLeftBipolarCenter = {
      groupType                           = "KernelConn";
      preLayerName                        = "LeftImage";
      postLayerName                       = "LeftBipolar";
      channelCode                         = 0;
      delay                               = 0.000000;
      numAxonalArbors                     = 1;
      plasticityFlag                      = false;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      weightInitType                      = "Gauss2DWeight";
      aspect                              = 1;
      sigma                               = .5;
      rMax                                = 3;
      rMin                                = 0;
      strength                            = 1;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 3;
      nyp                                 = 3;
      nfp                                 = 1;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeSum";
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = true;
      minSumTolerated                     = 0;
   };

   LeftBipolarToLeftGanglionCenter = {
      groupType                           = "KernelConn";
      preLayerName                        = "LeftBipolar";
      postLayerName                       = "LeftGanglion";
      channelCode                         = 0;
      delay                               = 0.000000;
      numAxonalArbors                     = 1;
      plasticityFlag                      = false;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      weightInitType                      = "Gauss2DWeight";
      aspect                              = 1;
      sigma                               = 1;
      rMax                                = 3;
      rMin                                = 0;
      strength                            = 1;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 1;
      nyp                                 = 1;
      nfp                                 = 1;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeSum";
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = true;
      minSumTolerated                     = 0;
   };

   LeftBipolarToLeftGanglionSurround = {
      groupType                           = "KernelConn";
      preLayerName                        = "LeftBipolar";
      postLayerName                       = "LeftGanglion";
      channelCode                         = 1;
      delay                               = 0.000000;
      numAxonalArbors                     = 1;
      plasticityFlag                      = false;
      convertRateToSpikeCount             = false;
      receiveGpu                          = false;
      weightInitType                      = "Gauss2DWeight";
      aspect                              = 1;
      sigma                               = 5.5;
      rMax                                = 7.5;
      rMin                                = 0.5;
      strength                            = 1;
      updateGSynFromPostPerspective       = false;
      pvpatchAccumulateType               = "convolve";
      writeStep                           = -1;
      writeCompressedCheckpoints          = false;
      selfFlag                            = false;
      nxp                                 = 11;
      nyp                                 = 11;
      nfp                                 = 1;
      shrinkPatches                       = false;
      normalizeMethod                     = "normalizeSum";
      normalizeArborsIndividually         = false;
      normalizeOnInitialize               = true;
      normalizeOnWeightUpdate             = true;
      rMinX                               = 0;
      rMinY                               = 0;
      nonnegativeConstraintFlag           = false;
      normalize_cutoff                    = 0;
      normalizeFromPostPerspective        = true;
      minSumTolerated                     = 0;
   };

   LeftRescaleToLeftError = {
      groupType                           = "IdentConn";
      preLayerName                        = "LeftRescale";
      postLayerName                       = "LeftError";
      channelCode                         = 0;
      delay                               = 0.000000;
      writeStep                           = -1;
   };
   }
)

--Right eye connections 
pv.addGroup(pvParams, "RightImageToRightBipolarCenter", pvParams["LeftImageToLeftBipolarCenter"],
   {
      preLayerName = "RightImage";
      postLayerName = "RightBipolar";
   }
)

pv.addGroup(pvParams, "RightBipolarToRightGanglionCenter", pvParams["LeftBipolarToLeftGanglionCenter"],
   {
      preLayerName = "RightBipolar";
      postLayerName = "RightGanglion";
   }
)

pv.addGroup(pvParams, "RightBipolarToRightGanglionSurround", pvParams["LeftBipolarToLeftGanglionSurround"],
   {
      preLayerName = "RightBipolar";
      postLayerName = "RightGanglion";
   }
)

pv.addGroup(pvParams, "RightRescaleToRightError", pvParams["LeftRescaleToLeftError"],
   {
      preLayerName = "RightRescale";
      postLayerName = "RightError";
   }
)

--LCA connections
pv.addGroup(pvParams, "V1ToLeftError", 
   {
      groupType = useMomentum and "MomentumConn" or "KernelConn";
      preLayerName = "V1";
      postLayerName = "LeftError";
      channelCode = 1;
      nxp = dictPatchSize;
      nyp = dictPatchSize;
      shrinkPatches = false;
      numAxonalArbors = 1;
      initFromLastFlag = 0;
      sharedWeights = true;
      strength = 1;
      normalizeMethod = "normalizeL2";
      minL2NormTolerated = 0;
      normalizeArborsIndividually = 0;
      normalize_cutoff = 0.0;
      normalizeFromPostPerspective = false;
      symmetrizeWeights = false;
      preActivityIsNotRate = false;
      keepKernelsSynchronized = true; 
      combine_dW_with_W_flag = false; 
      writeStep = -1;
      writeCompressedWeights = false;
      writeCompressedCheckpoints = false;
      plasticityFlag = true;
      triggerFlag = true;
      triggerLayerName = "LeftImage";
      triggerOffset = 1;
      initialWriteTime = 0.0;
      dWMax = learningRate;
      momentumTau = learningMomentumTau;
      momentumMethod = "viscosity";
      selfFlag = false;
      shmget_flag = false;
      delay = 0;
      useWindowPost = false;
      updateGSynFromPostPerspective = false;
      pvpatchAccumulateType = "convolve";
   }
)

--This is by reference, so changing things in connGroup will change things in pvParams
local v1ConnLeft = pvParams["V1ToLeftError"]
--These flags are based on external variables
if(V1DictDir == nil) then
   v1ConnLeft["weightInitType"] = "UniformRandomWeight"
   v1ConnLeft["wMinInit"] = -1
   v1ConnLeft["wMaxInit"] = 1
   v1ConnLeft["sparseFraction"] = .9;
else
   v1ConnLeft["weightInitType"] = "FileWeight"
   v1ConnLeft["initWeightsFile"] = V1DictDir.."/V1ToToLeftError_W.pvp";
end

--Copy lefterror connection to righterror connection
pv.addGroup(pvParams, "V1ToRightError",
   pvParams["V1ToLeftError"],
   {
      postLayerName = "RightError";
      normalizeMethod = "normalizeGroup";
   }
)

--Add extra overwrite parameters if nessessary
local v1ConnRight = pvParams["V1ToRightError"]
if(V1DictDir ~= nil) then
   v1ConnRight["initWeightsFile"] = V1DictDir.."/V1ToRightError_W.pvp";
end
v1ConnRight["normalizeGroupName"] = "V1ToLeftError"

--Recon and transpose conns
pv.addMultiGroups(pvParams, 
{
   LeftErrorToV1 = {
      groupType = "TransposeConn";
      preLayerName = "LeftError";
      postLayerName = "V1";
      channelCode = 0;
      originalConnName = "V1ToLeftError";
      selfFlag = false;
      preActivityIsNotRate = false;
      writeStep = -1;
      writeCompressedCheckpoints = false;
      shmget_flag = false;
      delay = 0;
      pvpatchAccumulateType = "convolve";
      updateGSynFromPostPerspective = true;
      receiveGpu = true;
   };

   V1ToLeftRecon = {
      groupType = "CloneKernelConn";
      preLayerName = "V1";
      postLayerName = "LeftRecon";
      channelCode = 0;
      writeStep = -1;
      originalConnName = "V1ToLeftError";
      selfFlag = false;
      delay = 0;
      preActivityIsNotRate = false;
      useWindowPost = false;
      updateGSynFromPostPerspective = false;
      pvpatchAccumulateType = "convolve";
   };
}
)

pv.addGroup(pvParams, "RightErrorToV1", pvParams["LeftErrorToV1"], 
   {
      preLayerName = "RightError";
      originalConnName = "V1ToRightError";
   }
)

pv.addGroup(pvParams, "V1ToRightRecon", pvParams["V1ToLeftRecon"], 
   {
      postLayerName = "RightRecon";
      originalConnName = "V1ToRightError";
   }
)

--Prints out a PetaVision approved parameter file to the console
pv.printConsole(pvParams)

