-------------------------------------------------------------------
-- Encoding benchmark dataset with SCANN
--
-- Sheng Lundquist 6/9/15
--
-- Implements encoding of benchmark training set with SCANN model as described in
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
local outputPath = "/home/ec2-user/output/benchmarkEncoding/encode_RELU/" --Output directory
local writePeriod = 1 --How often to write out
local checkpointWriteStepInterval = 5000 --How often to checkpoint
local stopTime = 195 --Stopping time, 195 images (rounded)
local progressStep = 5000; --How often to print out a status report

--Image parameters
local leftImageListPath = "/home/ec2-user/dataset/list/benchmark_left.txt" --List of images, raw
local rightImageListPath = "/home/ec2-user/dataset/list/benchmark_right.txt"
local displayPeriod = 1 --Display period for sparse approximation
local startFrame = 1 --Starting at frame 1

--LCA parameters
local stride = 2 --Stride of LCA
local numDictElements = 512 --Total number of dictionary elements
local dictPatchSize = 66 --Square patch, with this as the dimension
local VThresh = 0 --Threshold parameter, 0 for RELU
local VWidth = 0 --Threshold parameter

--nil for new dict, a checkpoint directory for loading weights
--local V1DictDir = nil
local V1DictDir = "/home/ec2-user/saved_output/dictTrain/saved_binoc_512_white/"

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
      groupType = "ANNErrorLayer";
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


pv.addGroup(pvParams, "V1_RELU",
{
   groupType = "ANNLayer";
   restart = 0;
   nxScale = .5/stride;
   nyScale = .5/stride;
   nf = numDictElements;
   writeStep = writePeriod;
   initialWriteTime = writePeriod;
   mirrorBCflag = 0;
   writeSparseActivity = 1;
   writeSparseValues   = 1;
   InitVType = "ZeroV";
   VThresh = VThresh;
   VMax = infinity;
   VMin = 0;
   VShift = 0.0;  
   VWidth = VWidth; 
   phase = 5;
});

--Connections
pv.addMultiGroups(pvParams, 
   {
   LeftImageToLeftBipolarCenter = {
      groupType                           = "HyPerConn";
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
      groupType = "HyPerConn";
      preLayerName = "V1_RELU";
      postLayerName = "LeftError";
      channelCode = -1;
      nxp = dictPatchSize;
      nyp = dictPatchSize;
      shrinkPatches = false;
      numAxonalArbors = 1;
      initFromLastFlag = 0;
      sharedWeights = true;
      weightInitType = "FileWeight";
      initWeightsFile = V1DictDir.."/V1ToLeftError_W.pvp";
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
      plasticityFlag = false;
      initialWriteTime = 0.0;
      selfFlag = false;
      shmget_flag = false;
      delay = 0;
      useWindowPost = false;
      updateGSynFromPostPerspective = false;
      pvpatchAccumulateType = "convolve";
   }
)

--Copy lefterror connection to righterror connection
pv.addGroup(pvParams, "V1ToRightError",
   pvParams["V1ToLeftError"],
   {
      postLayerName = "RightError";
      normalizeMethod = "normalizeGroup";
      initWeightsFile = V1DictDir.."/V1ToRightError_W.pvp";
   }
)
pvParams["V1ToRightError"]["normalizeGroupName"] = "V1ToLeftError"

--Recon and transpose conns
pv.addMultiGroups(pvParams, 
{
   LeftErrorToV1 = {
      groupType = "TransposeConn";
      preLayerName = "LeftError";
      postLayerName = "V1_RELU";
      channelCode = 1;
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
}
)

pv.addGroup(pvParams, "RightErrorToV1", pvParams["LeftErrorToV1"], 
   {
      preLayerName = "RightError";
      originalConnName = "V1ToRightError";
   }
)

--Prints out a PetaVision approved parameter file to the console
pv.printConsole(pvParams)


