--Util module
--package.path = package.path .. ";" .. os.getenv("HOME") .. "/workspace/PetaVision/parameterWrapper/PVModule.lua;"
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

--TODO change this parameter
local leftImageListPath = "/home/ec2-user/mountData/kitti/list/image_02_aws.txt"
local rightImageListPath = "/home/ec2-user/mountData/kitti/list/image_03_aws.txt"
local displayPeriod = 200
local startFrame = 1
local dictPatchSize = 66

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
      filenamesContainLayerNames = false;  
      filenamesContainConnectionNames = false;
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
      writeStep = displayPeriod;
      initialWriteTime = displayPeriod;
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

   LeftWhitened = {
      groupType = "ANNLayer";
      nxScale                             = .5;
      nyScale                             = .5;
      nf                                  = nf;
      phase                               = 4;
      mirrorBCflag                        = true;
      InitVType                           = "ZeroV";
      triggerFlag                         = true;
      triggerLayerName                    = "LeftImage";
      triggerOffset                       = 0;
      writeStep                           = displayPeriod;
      initialWriteTime                    = displayPeriod;
      sparseLayer                         = false; --How do we specify if the layer is sparse?
      updateGpu                           = false;
      VThresh                             = -INFINITY;
      AMin                                = -INFINITY;
      AMax                                = INFINITY;
      AShift                              = 0;
      VWidth                              = 0;
   };
} --End of table constructor

--Include a rightImage and rightWhitened
pv.addGroup(pvParams, "RightImage", pvParams["LeftImage"], 
   {
      imageListPath = rightImageListPath;
   }
)

pv.addGroup(pvParams, "RightWhitened", pvParams["LeftWhitened"], 
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
   "LeftWhitened",
   pvParams["LeftWhitened"], --Output layer
   dictPatchSize * dictPatchSize -- Patch size to feed to l2 rescale
)

--Right eye
local rightWhiteGroups = whiten.ds2_white_rescale(
   "Right", --Prefix
   "RightImage", --Input layer
   pvParams["RightImage"],
   "RightWhitened",
   pvParams["RightWhitened"], --Output layer
   dictPatchSize * dictPatchSize -- Patch size to feed to l2 rescale
)

pv.addGroup(pvParams, leftWhiteGroups)
pv.addGroup(pvParams, rightWhiteGroups)

--Prints out a PetaVision approved parameter file to the console
pv.printConsole(pvParams)

