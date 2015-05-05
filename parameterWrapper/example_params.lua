--
-- example_params.lua
--
-- created by slundquist
--
-- A parameter showcasing lua syntax and examples
-- Derived from BasicSystemTests
--


--Util module
local pv = require "PVModule"

-- Global variable, for debug parsing
-- Needed by printConsole
debugParsing = false;

-- User defined variables
local nxSize = 32
local nySize = 32

--Table constructor
--This is where we construct the basic table for the parameter. The constructor is your
--typical way to define a basic parameter file.
--Note that this is an lua array, therefore, the keys must be iterated in order starting
--from 1, with no other keys allowed
local basicParams = {
   --Implicit key of 1; since lua is 1 indexed and no key is specified
   --Each key must be an integer
   --Value is a table of key/value pairs
   --HyPerCol object
   --"groupType" (note the quotes; no quotes means variable) is key, "HyPerCol" is value
   {  
      groupType = "HyPerCol"; --String values
      groupName = "column";
      nx = nxSize;  --Using user defined variables 
      ny = nySize;
      dt = 1.0;
      dtAdaptFlag = false; --Boolean values 
      randomSeed = 1234567890;
      startTime = 0.0;
      stopTime = 10.0; 
      errorOnNotANumber = true;
      progressInterval = 10.0;
      writeProgressToErr = false;
      verifyWrites = false;
      outputPath = "output/";
      printParamsFilename = "pv.params";
      filenamesContainLayerNames = false;  
      filenamesContainConnectionNames = false;
      initializeFromCheckpointDir = nil; --"NULL" variable
      checkpointWrite = false;
      suppressLastOutput = false
   };

   --
   -- layers
   --
   -- Implicit key of 2 for same reason as above. This counter will keep incrementing for every non-keyed value specified
   {
      groupType = "Image";
      groupName = "Input";
      nxScale = 1;
      nyScale = 1;
      imagePath = "input/sampleimage.png";
      nf = 1;
      phase = 0;
      writeStep = -1;
      sparseLayer = false;
      mirrorBCflag = false;
      valueBC = 0.0;
      useImageBCflag = false;
      inverseFlag = false ;
      normalizeLuminanceFlag = false;
      autoResizeFlag = false;
      writeImages = false;
      offsetAnchor = "tl";
      offsetX = 0;
      offsetY = 0;
      jitterFlag = false;
      padValue = false
   };

   --an output layer
   {
      groupType = "ANNLayer";
      groupName = "Output";
      nxScale = 1;
      nyScale = 1;
      nf = 8;
      phase = 1;
      triggerFlag = false;
      writeStep = 1.0;
      initialWriteTime = 0.0;
      mirrorBCflag = 1;
      sparseLayer = false;
      InitVType = "ZeroV";
      VThresh = INFINITY; --Infinity, user defined variable
      AMax = INFINITY;
      AMin = -INFINITY;
      AShift = 0.0;
      VWidth = 0.0;
      clearGSynInterval = 0.0;
   };

   --a connection
   {
      groupType = "HyPerConn";
      groupName = "InputToOutput";
      preLayerName = "Input";
      postLayerName = "Output";
      channelCode = 0;
      nxp = 7;
      nyp = 7;
      nfp = 8;
      numAxonalArbors = 1;
      sharedWeights = true;
      writeStep = -1;
      weightInitType = "Gauss2DWeight";
      deltaThetaMax = 6.283185;
      thetaMax = 1.0;
      numFlanks = 1;
      flankShift = 0;
      rotate = false;
      bowtieFlag = false;
      aspect = 3;
      sigma = 1;
      rMax  = INFINITY;
      rMin = 0;
      numOrientationsPost = 8;
      strength = 4.0;
      normalizeMethod = "normalizeSum";
      normalizeArborsIndividually = false;
      normalizeOnInitialize = true;
      normalizeOnWeightUpdate = true;
      normalize_cutoff = 0;
      convertRateToSpikeCount = false;
      minSumTolerated = 0.0;
      normalizeFromPostPerspective = false;
      rMinX = 0.0;
      rMinY = 0.0;
      nonnegativeConstraintFlag = false;
      writeCompressedCheckpoints = false;
      plasticityFlag = false;
      selfFlag = false;
      delay = 0;

      pvpatchAccumulateType = "Convolve";
      shrinkPatches = false;
      updateGSynFromPostPerspective = false;
   };
} --End of table constructor

--Adding a connection to the parameters
--PVModule.addGroup(baseParameterTable, group)
--Calling the pv module to add group to baseParameterTable
pv.addGroup(
   basicParams, --Base parameter table variable 
   {
      groupType = "ANNLayer";
      groupName = "Output2";
      nxScale = 1;
      nyScale = 1;
      nf = 8;
      phase = 1;
      triggerFlag = false;
      writeStep = 1.0;
      initialWriteTime = 0.0;
      mirrorBCflag = 1;
      sparseLayer = false;
      InitVType = "ZeroV";
      VThresh = INFINITY; --Infinity, user defined variable
      AMax = INFINITY;
      AMin = -INFINITY;
      AShift = 0.0;
      VWidth = 0.0;
      clearGSynInterval = 0.0;
   }
) --End of function call

--This is the same as adding a group using the module
--DO NOT USE THIS CODE, use the addGroup interface instead
--Note that the table basicParams must be a list, aka, integer keys from 1 to numGroups
--continuous, with no other keys
--#basicParams means number of items in the LIST. +1 for 1 indexing
--
-- basicParams[#basicParams+1] = 
-- {
--    groupType = "ANNLayer";
--    groupName = "Output2";
--    nxScale = 1;
--    nyScale = 1;
--    nf = 8;
--    phase = 1;
--    triggerFlag = false;
--    writeStep = 1.0;
--    initialWriteTime = 0.0;
--    mirrorBCflag = 1;
--    sparseLayer = false;
--    InitVType = "ZeroV";
--    VThresh = INFINITY; --Infinity, user defined variable
--    AMax = INFINITY;
--    AMin = -INFINITY;
--    AShift = 0.0;
--    VWidth = 0.0;
--    clearGSynInterval = 0.0;
-- }

--Function to include a previously defined group
--Note that both are addGroup, only depends on if you specify a new group or
--if you call the function "getGroupFromName"
--Third parameter is a group that overwrites/adds parameters
--Make a new ANNLayer named "Output3", including from "Output2"
--Change nxScale and nyScale to .5
pv.addGroup(basicParams,
   pv.getGroupFromName(basicParams, "Output2"), 
   --Overwriting params for this parameter group
   {
      groupName = "Output3";
      nxScale = .5;
      nyScale = .5;
   }
)

--Prints out a PetaVision approved parameter file to the console
pv.printConsole(basicParams)

