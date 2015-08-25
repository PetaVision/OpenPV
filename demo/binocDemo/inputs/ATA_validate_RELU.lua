-------------------------------------------------------------------
-- ATA validate for RELU
--
-- Sheng Lundquist 6/9/15
--
-- Implements training of ATA method with SCANN model as described in
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
local outputPath = "/home/ec2-user/output/depthInference/ATA_validate_RELU/" --Output directory
local writePeriod = 1 --How often to write out
local stopTime = 93 --Stopping time, last 93 
local progressStep = 5; --How often to print out a status report

--Image parameters
local V1PVP = "/home/ec2-user/saved_output/benchmarkEncoding/V1_RELU.pvp" --V1 activity file
local depthImageListPath = "/home/ec2-user/dataset/list/benchmark_depth_noc.txt" --List of images, raw
local numDepthBins = 128;

--LCA parameters
local stride = 2 --Stride of LCA
local numDictElements = 512 --Total number of dictionary elements
local dictPatchSize = 66 --Square patch, with this as the dimension

--ATA previous weights
local ATAWeightsDir = "/home/ec2-user/saved_output/depthInference/saved_ATA_RELU/"

local pvParams = {
   --base column
   column = {
      groupType = "HyPerCol";
      nx = nxSize;
      ny = nySize;
      dt = 1.0;
      randomSeed = 1234567890;
      startTime = 0;
      stopTime = stopTime;
      progressStep = progressStep;
      outputPath = outputPath;
      filenamesContainLayerNames = 2;
      filenamesContainConnectionNames = 2;
      checkpointRead = false;
      checkpointWrite = false;
      suppressLastOutput = false;
      deleteOlderCheckpoints = true;
      writeProgressToErr = true;
      outputNamesOfLayersAndConns = "LayerAndConnNames.txt";
      dtAdaptFlag = true;
      dtScaleMax = 5.0;
      dtScaleMin = 0.25;
      dtChangeMax = 0.05;
      dtChangeMin = 0.0;
   };

   V1_RELU = {
      groupType = "Movie";
      restart = 0;
      nxScale = .5/stride;
      nyScale = .5/stride;
      nf = numDictElements;
      readPvpFile = true;
      imageListPath = V1PVP;
      writeFrameToTimestamp = true;
      writeStep = -1;
      sparseLayer = true;
      writeSparseValues = true;
      displayPeriod = 1;
      start_frame_index = 101;
      skip_frame_index = 1;
      echoFramePathnameFlag = true;
      mirrorBCflag = true;
      jitterFlag = 0;
      useImageBCflag = false;
      inverseFlag = false;
      normalizeLuminanceFlag = false;
      writeImages = false;
      offsetX = 0;
      offsetY = 0;
      autoResizeFlag = 0;
      randomMovie = 0;
      phase = 0;
   };

   DepthImage = {
      groupType = "Movie";
       restart = 0;
       nxScale = 1;
       nyScale = 1;
       readPvpFile = false; 
       imageListPath = depthImageListPath;
       writeFrameToTimestamp = true;
       nf = 1;
       writeStep = -1;
       sparseLayer = false;
       displayPeriod = 1;
       start_frame_index = 100;
       skip_frame_index = 1;
       echoFramePathnameFlag = true;
       mirrorBCflag = false;
       jitterFlag = 0;
       useImageBCflag = false;
       inverseFlag = false;
       normalizeLuminanceFlag = false;
       writeImages = false;
       offsetAnchor = "br"; 
       offsetX = 0; 
       offsetY = 0; 
       randomMovie = 0;
       autoResizeFlag = 0;
       phase = 0;
   };

   DepthDownsample = {
      groupType = "ANNLayer";
       restart = 0;
       nxScale = .25;
       nyScale = .25;
       nf = 1;
       writeStep = writePeriod;
       initialWriteTime = writePeriod;
       mirrorBCflag = false;
       writeSparseActivity = 0;
       InitVType = "ZeroV";
       VThresh = -INFINITY;
       VMax =  INFINITY;
       VMin = -INFINITY;
       VShift = 0;
       VWidth = 0;
       phase = 1;  
   };

   RCorrBuf = {
      groupType = "ANNLayer";
      restart = 0;
      nxScale = .5/stride;
      nyScale = .5/stride;
      nf = numDepthBins;
      writeStep = -1.0;
      initialWriteTime = 1.0;
      mirrorBCflag = 0;
      writeSparseActivity = 0;
      InitVType = "ZeroV";
      VThresh = -INFINITY;
      VMax = INFINITY;
      VMin = -INFINITY; 
      VShift = 0;
      VWidth = 0;
      valueBC = 0;
      phase = 5;
   };

   RCorrRecon = {
      groupType = "WTALayer";
      restart = 0;
      nxScale = .5/stride; 
      nyScale = .5/stride;
      nf = 1; 
      writeStep = 1.0;
      initialWriteTime = 1.0;
      mirrorBCflag = false;
      writeSparseActivity = false;
      delay = 0;
      originalLayerName = "RCorrBuf";
      phase = 6;
   };

   DepthImageToDepthDownsample = {
      groupType = "PoolingConn";
       preLayerName = "DepthImage";
       postLayerName = "DepthDownsample";
       channelCode = 0;
       sharedWeights = true;
       nxp = 1; 
       nyp = 1; 
       numAxonalArbors = 1;
       initFromLastFlag = 0;
       writeStep = -1;
       initialWriteTime = 0.0;
       writeCompressedWeights = false;
       normalizeMethod                     = "none";
       shrinkPatches = false;
       writeCompressedCheckpoints = false;
       plasticityFlag = 0;
       pvpatchAccumulateType = "maxpooling";
       delay = 0;
       convertRateToSpikeCount = false;
       selfFlag = false;
       updateGSynFromPostPerspective = false;
       useWindowPost = false;
       keepKernelsSynchronized             = true;
   };

   V1ToDepthGT = {
       groupType = "HyPerConn";
       preLayerName = "V1_RELU";
       postLayerName = "RCorrBuf";
       channelCode = -1;
       nxp = 33;
       nyp = 33;
       shrinkPatches = false;
       numAxonalArbors = 1;
       initFromLastFlag = 0;
       weightInitType = "FileWeight";
       initWeightsFile = ATAWeightsDir.."/V1ToDepthGT_W.pvp";
       strength = 1;
       normalizeMethod = "none";
       preActivityIsNotRate = false;
       keepKernelsSynchronized = true; 
       combine_dW_with_W_flag = false; 
       writeStep = -1;
       writeCompressedWeights = false;
       writeCompressedCheckpoints = false;
       plasticityFlag = false;
       weightUpdatePeriod = 1.0;
       initialWeightUpdateTime = 1.0;
       initialWriteTime = 0.0;
       dWMax = 1;
       selfFlag = false;
       shmget_flag = false;
       delay = 0;
       useWindowPost = false;
       updateGSynFromPostPerspective = false;
       pvpatchAccumulateType = "convolve";
       useMask = true;
       maskLayerName = "DepthDownsample";
       sharedWeights = true;
   };
}

--Prints out a PetaVision approved parameter file to the console
pv.printConsole(pvParams)

