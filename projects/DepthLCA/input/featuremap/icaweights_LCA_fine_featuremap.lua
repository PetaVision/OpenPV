--Load packages
package.path = package.path .. ";" 
            .. os.getenv("HOME") 
            .. "/workspace/OpenPV/pv-core/parameterWrapper/?.lua";
local pv = require "PVModule";
--local subnets = require "PVSubnets";

local params = {};

local numDictElements = 512;
local leftDictFile = "/home/sheng/mountData/benchmark/icaweights_binoc_LCA_fine/Checkpoints/Checkpoint194000/V1ToLeftError_W.pvp";
local rightDictFile = "/home/sheng/mountData/benchmark/icaweights_binoc_LCA_fine/Checkpoints/Checkpoint194000/V1ToRightError_W.pvp";
local V1File = "~/mountData/benchmark/icaweights_binoc_LCA_fine/a12_V1.pvp";

params["column"] = {
   groupType = "HyPerCol";
   nx = 1200; 
   ny = 360;  
   dt = 1.0;
   randomSeed = 1234567890;
   startTime = 0;
   stopTime = 1; 
   progressStep = 10;
   outputPath = "/home/sheng/mountData/benchmark/featuremap/icaweights_binoc_LCA_fine/";
   filenamesContainLayerNames = 2;
   filenamesContainConnectionNames = 2;
   checkpointRead = false;
   checkpointWrite = false;
   deleteOlderCheckpoints = true;
   suppressLastOutput = true;
   writeProgressToErr = true;
   outputNamesOfLayersAndConns = "LayerAndConnNames.txt";
   dtAdaptFlag = false;
};

params["V1"] = {
   groupType = "MoviePvp";
   restart = 0;
   nxScale = .25;
   nyScale = .25;
   nf = numDictElements;
   inputPath = V1File;
   writeFrameToTimestamp = true;
   writeStep = -1;
   sparseLayer = true;
   writeSparseValues = true;
   displayPeriod = 1;
   start_frame_index = 150; 
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

--params["LeftRecon"] = {
--    groupType = "ANNLayer";
--    nxScale                             = .5;
--    nyScale                             = .5;
--    nf                                  = 1;
--    phase                               = 3;
--    mirrorBCflag                        = false;
--    valueBC                             = 0;
--    initializeFromCheckpointFlag        = false;
--    InitVType                           = "ZeroV";
--    triggerFlag                         = false;
--    writeStep                           = 1;
--    initialWriteTime                    = 1;
--    sparseLayer                         = false;
--    updateGpu                           = false;
--    dataType                            = nil;
--    VThresh                             = -infinity;
--    AMin                                = -infinity;
--    AMax                                = infinity;
--    AShift                              = 0;
--    VWidth                              = 0;
--    clearGSynInterval                   = 0;
--};
--
--pv.addGroup(params, "RightRecon",
--   params["LeftRecon"]
--)

--Full recon
params["V1ToLeftRecon"] = {
   groupType = "HyPerConn";
   preLayerName = "V1";
   postLayerName = "LeftRecon_slice";
   channelCode = -1;
   nxp = 66;
   nyp = 66;
   shrinkPatches = false;
   numAxonalArbors = 1;
   initFromLastFlag = 0;
   sharedWeights = true;
   weightInitType = "FileWeight";
   initWeightsFile = leftDictFile;
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
};

pv.addGroup(params, "V1ToRightRecon",
   params["V1ToLeftRecon"],
   {
      postLayerName = "RightRecon_slice";
      initWeightsFile = rightDictFile;
   }
)

--Mask each element to a new layer
--for i_map = 0, numDictElements, 1 do

params["V1_slice"] = {
   groupType          = "MaskLayer";
   nxScale            = .25;
   nyScale            = .25;
   nf                 = numDictElements;
   writeStep          = -1;
   mirrorBCflag       = false;
   valueBC            = 0.0;
   sparseLayer        = 1;
   writeSparseValues  = 1;
   InitVType          = "ZeroV";
   VThresh            = -infinity;
   AMax               = infinity;     
   AMin               = -infinity; 
   phase              = 1; 
   triggerFlag        = false;
   maskMethod         = "noMaskFeatures";
   --featureIdxs        = i_map;
}

sweep = {}
for i = 0, numDictElements-1, 1 do
   sweep[i+1] = i;
end

--sweep feature indexes
pv.paramSweep(params, "V1_slice", "featureIdxs", sweep);

params["V1ToV1_slice"] = {
   groupType                           = "IdentConn";
   preLayerName                        = "V1";
   postLayerName                       = "V1_slice";
   channelCode                         = 0;
   delay                               = 0;
   writeStep                           = -1;
}

params["LeftRecon_slice"] = {
    groupType = "ANNLayer";
    nxScale                             = .5;
    nyScale                             = .5;
    nf                                  = 1;
    phase                               = 3;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerFlag                         = false;
    writeStep                           = 1;
    initialWriteTime                    = 1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -infinity;
    AMin                                = -infinity;
    AMax                                = infinity;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

--pv.addGroup(params, "LeftRecon_slice",
--   params["LeftRecon"], 
--)

pv.addGroup(params, "RightRecon_slice",
   params["LeftRecon_slice"]
)

params["V1_slice_ToLeftRecon_slice"] = {
   groupType                           = "CloneConn";
   preLayerName                        = "V1_slice";
   postLayerName                       = "LeftRecon_slice";
   channelCode                         = 0;
   delay                               = 0;
   convertRateToSpikeCount             = false;
   receiveGpu                          = false;
   updateGSynFromPostPerspective       = false;
   pvpatchAccumulateType               = "convolve";
   writeStep                           = -1;
   writeCompressedCheckpoints          = false;
   selfFlag                            = false;
   originalConnName                    = "V1ToLeftRecon";
}

pv.addGroup(params, "V1_slice_ToRightRecon_slice",
   params["V1_slice_ToLeftRecon_slice"],
   {
      postLayerName = "RightRecon_slice";
      originalConnName = "V1ToRightRecon";
   }
)


--end

-- Print out PetaVision approved parameter file to the console
pv.printConsole(params)
