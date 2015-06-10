-------------------------------------------------------------------
-- Generates a full downsample depth pvp file for depth tuning
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
local outputPath = "/home/ec2-user/output/depthTuning/downsample_depth/" --Output directory
local writePeriod = 1 --How often to write out
local stopTime = 195 --Stopping time, first 100 
local progressStep = 5; --How often to print out a status report

--Image parameters
local depthImageListPath = "/home/ec2-user/dataset/list/benchmark_depth_noc.txt" --List of images, raw

--LCA parameters
local stride = 2 --Stride of LCA
local numDictElements = 512 --Total number of dictionary elements
local dictPatchSize = 66 --Square patch, with this as the dimension

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
      start_frame_index = 1;
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
      normalizeMethod = "none";
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
}

--Prints out a PetaVision approved parameter file to the console
pv.printConsole(pvParams)

