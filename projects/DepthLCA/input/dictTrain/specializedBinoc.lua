--Util module
package.path = package.path .. ";" .. os.getenv("HOME") .. "/workspace/PetaVision/parameterWrapper/?.lua;"
--Parameter includes

local pv = require("PVModule")
local whiten = require("params.preprocess")

-- Global variable, for debug parsing
-- Needed by printConsole
debugParsing = true

-- User defined variables
local nxSize = 1200
local nySize = 360 
local outputPath = "/nh/compneuro/Data/Depth/LCA/dictLearn/specializedBinoc/"

--Image parameters

local leftImageListPath = "/nh/compneuro/Data/KITTI/list/image_02.txt"
local rightImageListPath = "/nh/compneuro/Data/KITTI/list/image_03.txt"
local depthImageListPath = "/nh/compneuro/Data/KITTI/list/depth.txt"
local displayPeriod = 200
local writePeriod = 200 
local startFrame = 350

--Depth parameters
local numDepthBins = 64

--LCA parameters
local stride = 2
local numDictElementsPerClass = 16
local dictPatchSize = 66
local VThresh = 0.005
local VWidth = .05

local learningRate = 0.05
local useMomentum = true 
local learningMomentumTau = 100;

--nil for new dict, a checkpoint directory for loading weights
--local V1DictDir = nil
local V1DictDir = "/nh/compneuro/Data/Depth/LCA/dictLearn/spec_saved/"

--Table constructor
--This is where we construct the basic table for the parameter. The constructor is your
--typical way to define a basic parameter file.
--Note that this is an lua array, therefore, the keys must be iterated in order starting
--from 1, with no other keys allowed
local pvParams = {
   column = {  
      groupType = "HyPerCol"; --String values
      nx = nxSize;  --Using user defined variables 
      ny = nySize;
      dt = 1.0;
      randomSeed = 1234567890;
      startTime = 0.0;
      stopTime = 100000000.0; 
      progressInterval = 5000.0;
      writeProgressToErr = true;
      outputPath = outputPath;
      printParamsFilename = "pv.params";
      filenamesContainLayerNames = 2;  
      filenamesContainConnectionNames = 2;
      initializeFromCheckpointDir = nil; --"NULL" variable
      checkpointWrite = true;
      checkpointWriteDir = outputPath .. "/Checkpoints";
      checkpointWriteStepInterval = 10000;
      suppressLastOutput = false;
      errorOnNotANumber = true;
      outputNamesOfLayersAndConns = "LayerAndConnNames.txt";
      dtAdaptFlag = true;
      dtScaleMax = 5.0;
      dtScaleMin = 0.25;
      dtChangeMax = 0.05;
      dtChangeMin = 0.0;
   };

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
      VMin = 0;
      VShift = 0;
      VWidth = 0;
      valueBC = 0;
      errScale = 1;
      phase = 4;  
      triggerLayerName = "LeftImage";
   }
} --End of table constructor

--Include a rightImage and rightWhitened
pv.addGroup(pvParams, "RightImage", pvParams["LeftImage"], 
   {
      imageListPath = rightImageListPath;
   }
)

pv.addGroup(pvParams, "RightError", pvParams["LeftError"], 
   {
      triggerLayerName = "RightImage";
   }
)

--Include preprocessing layers
--Left eye
local leftWhiteGroups = whiten.ds2_white_rescale(
   "Left", --Prefix
   "LeftImage",
   pvParams["LeftImage"], --Input layer
   "LeftError",
   pvParams["LeftError"], --Output layer
   dictPatchSize * dictPatchSize, -- Patch size to feed to l2 rescale
   writePeriod
)

--Right eye
local rightWhiteGroups = whiten.ds2_white_rescale(
   "Right", --Prefix
   "RightImage", --Input layer
   pvParams["RightImage"],
   "RightError",
   pvParams["RightError"], --Output layer
   dictPatchSize * dictPatchSize, -- Patch size to feed to l2 rescale
   writePeriod
)

pv.addMultiGroups(pvParams, leftWhiteGroups)
pv.addMultiGroups(pvParams, rightWhiteGroups)

--Depth groups
pv.addMultiGroups(pvParams, 
{
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
       displayPeriod = displayPeriod;
       start_frame_index = startFrame;
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
      nxScale = .5;
      nyScale = .5;
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
      triggerFlag = 1;
      triggerLayerName = "DepthImage";
      triggerOffset = 0;
      phase = 1;  
   };

   --Bin depth into multiple features
   DepthBinned = {
      groupType = "BinningLayer";
      restart = 0;
      nxScale = .5; 
      nyScale = .5;
      nf = numDepthBins;
      writeStep = -1.0;
      initialWriteTime = 1.0;
      mirrorBCflag = false;
      sparseLayer = true;
      binMax = 1;
      binMin = 0;
      binSigma = 3;
      zeroNeg = false;
      zeroDCR = true;
      normalDist = false;
      delay = 0;
      originalLayerName = "DepthDownsample";
      phase = 2;
      triggerFlag = 1;
      triggerLayerName = "DepthImage";
      triggerOffset = 0;
   };

   --Add a DNC class to GT
   DepthGT = {
      groupType = "BackgroundLayer";
      restart = 0;
      nxScale = .5; 
      nyScale = .5;
      nf = numDepthBins + 1;
      writeStep = -1.0;
      initialWriteTime = 1.0;
      mirrorBCflag = false;
      sparseLayer = true;
      delay = 0;
      originalLayerName = "DepthBinned";
      phase = 3;
      triggerFlag = 1;
      triggerLayerName = "DepthImage";
      triggerOffset = 0;
   };

   --Connections
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
      shrinkPatches = false;
      writeCompressedCheckpoints = false;
      plasticityFlag = 0;
      pvpatchAccumulateType = "maxpooling";
      delay = 0;
      convertRateToSpikeCount = false;
      selfFlag = false;
      updateGSynFromPostPerspective = true;
      useWindowPost = false;
      keepKernelsSynchronized = true;
   };
} --End table constructor
) --End addMultiGroup function call


--LCA linked dictionary connections
local v1Groups = {}
-- +1 for dnc class
-- lua is 1 indexed, so starting from 0 means running 1 more than # of depth bins
for i=0,numDepthBins do
   --LCA Layers
   pv.addGroup(v1Groups, "V1_spec_" .. i,
   {
      groupType = "HyPerLCALayer";
      restart = 0;
      nxScale = .5/stride;
      nyScale = .5/stride;
      nf = numDictElementsPerClass;
      numChannels = 1;
      numWindowX = 1;
      numWindowY = 1;
      writeStep = writePeriod;
      initialWriteTime = writePeriod;
      mirrorBCflag = 0;
      writeSparseActivity = 1;
      writeSparseValues   = 1;
      timeConstantTau = 200.0;
      slopeErrorStd = 0.01;
      dVThresh = 0;
      VThresh = VThresh;
      VMax = INFINITY;
      VMin = 0;
      VShift = 0.0;
      VWidth = VWidth;
      updateGpu = true;
      phase = 5;
      InitVType = "UniformRandomV";
      minV = -1.0;
      maxV = .02;
   } --End table constructor
   ) --End addGroup function call

   --Recon layers, one for left, one for right per V1
   pv.addGroup(v1Groups, "LeftRecon_spec_" .. i,
   {
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
      VMax = INFINTIY;
      VMin = -INFINITY; 
      VShift = 0;
      VWidth = 0;
      valueBC = 0;
      phase = 6;
      triggerFlag = 1;
      triggerLayerName = "LeftImage";
      triggerOffset = 1;
   } --End table constructor
   ) --End addGroup function call

   pv.addGroup(v1Groups, "RightRecon_spec_" .. i,
      v1Groups["LeftRecon_spec_"..i],
      {
         triggerLayerName = "RightImage";
      }
   )

   --Connections
   --Plasticity connection
   pv.addGroup(v1Groups, "V1_spec_" .. i .. "ToLeftError", 
   {
      --This is shorthand equivelent for useMomentum ? "MomentumConn" : "KernelConn"
      --http://www.lua.org/pil/3.3.html
      groupType = useMomentum and "MomentumConn" or "KernelConn"; 
      preLayerName = "V1_spec_" .. i;
      postLayerName = "LeftError";
      channelCode = 1; --Inhib connection to error
      nxp = dictPatchSize;
      nyp = dictPatchSize;
      shrinkPatches = false;
      numAxonalArbors = 1;
      initFromLastFlag = 0;
      sharedWeights = true;
      strength = 1;
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

      normalizeMethod = "normalizeL2";
      minL2NormTolerated = 0;
      normalizeArborsIndividually = 0;
      normalize_cutoff = 0.0;
      normalizeFromPostPerspective = false;

      useMask = true;
      maskLayerName = "DepthGT";
      maskFeatureIdx = i; --Masking out based on what dict is part of
      
   }
   )

   --This is by reference, so changing things in connGroup will change things in v1Groups
   local v1ConnGroup = v1Groups["V1_spec_"..i.."ToLeftError"]
   --These flags are based on external variables
   if(V1DictDir == nil) then
      v1ConnGroup["weightInitType"] = "UniformRandomWeight"
      v1ConnGroup["wMinInit"] = -1
      v1ConnGroup["wMaxInit"] = 1
      v1ConnGroup["sparseFraction"] = .9;
   else
      v1ConnGroup["weightInitType"] = "FileWeight"
      v1ConnGroup["initWeightsFile"] = V1DictDir.."V1_spec_"..i.."ToLeftError_W.pvp";
   end

   --Copy lefterror connection to righterror connection
   pv.addGroup(v1Groups, "V1_spec_"..i.."ToRightError",
      v1Groups["V1_spec_"..i.."ToLeftError"],
      {
         postLayerName = "RightError";
         normalizeMethod = "normalizeGroup";
      }
   )

   --Add extra overwrite parameters if nessessary
   local v1ConnGroup = v1Groups["V1_spec_"..i.."ToRightError"]
   if(V1DictDir ~= nil) then
      v1ConnGroup["initWeightsFile"] = V1DictDir.."V1_spec_"..i.."ToRightError_W.pvp";
   end

   v1ConnGroup["normalizeGroupName"] = "V1_spec_"..i.."ToLeftError"

   --Add connections for transpose and clone
   pv.addGroup(v1Groups, "LeftErrorToV1_spec_"..i, 
   {
      groupType = "TransposeConn";
      preLayerName = "LeftError";
      postLayerName = "V1_spec_"..i;
      channelCode = 0;
      originalConnName = "V1_spec_"..i.."ToLeftError";
      selfFlag = false;
      preActivityIsNotRate = false;
      writeStep = -1;
      writeCompressedCheckpoints = false;
      shmget_flag = false;
      delay = 0;
      pvpatchAccumulateType = "convolve";
      updateGSynFromPostPerspective = true;
      receiveGpu = true;
   })

   pv.addGroup(v1Groups, "RightErrorToV1_spec_"..i, v1Groups["LeftErrorToV1_spec_"..i], 
   {
      preLayerName = "RightError";
      originalConnName = "V1_spec_"..i.."ToRightError";
   })

   pv.addGroup(v1Groups, "V1_spec_"..i.."ToLeftRecon_spec_"..i,
   {
      groupType = "CloneKernelConn";
      preLayerName = "V1_spec_"..i;
      postLayerName = "LeftRecon_spec_"..i;
      channelCode = 0;
      writeStep = -1;
      originalConnName = "V1_spec_"..i.."ToLeftError";
      selfFlag = false;
      delay = 0;
      preActivityIsNotRate = false;
      useWindowPost = false;
      updateGSynFromPostPerspective = false;
      pvpatchAccumulateType = "convolve";
   })

   pv.addGroup(v1Groups, "V1_spec_"..i.."ToRightRecon_spec_"..i,
   v1Groups["V1_spec_"..i.."ToLeftRecon_spec_"..i],
   {
      postLayerName = "RightRecon_spec_"..i;
      originalConnName = "V1_spec_"..i.."ToRightError";
   }
   )

end --end for loop across numDepthBins

--Add generated V1 groups to base list
pv.addMultiGroups(pvParams, v1Groups)

--Prints out a PetaVision approved parameter file to the console
pv.printConsole(pvParams)

