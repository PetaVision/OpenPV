--
-- example_params.lua
--
-- created by slundquist
--
-- A parameter showcasing lua syntax and examples
-- Derived from BasicSystemTests
--

--Util module
package.path = package.path .. ";" .. os.getenv("HOME") .. "/workspace/OpenPV/pv-core/parameterWrapper/?.lua;"
local pv = require "PVModule"

-- Global variable, for debug parsing
-- Needed by printConsole
debugParsing = true;

-- User defined variables
local nxSize = 32
local nySize = 32

--Table constructor
--This is where we explicitly construct the basic table for the parameter. The constructor is your
--typical way to define a basic parameter file.
local basicParams = {
   --Implicit key of 1; since lua is 1 indexed and no key is specified
   --Each key must be an integer
   --Value is a table of key/value pairs
   --HyPerCol object
   --"groupType" (note the quotes; no quotes means variable) is key, "HyPerCol" is value
   column = {  
      groupType = "HyPerCol"; --String values
      nx = nxSize;  --Using user defined variables 
      ny = nySize;
      dt = 1.0;
      randomSeed = 1234567890;
      stopTime = 10.0; 
      errorOnNotANumber = true;
      progressInterval = 10.0;
      writeProgressToErr = false;
      verifyWrites = false;
      outputPath = "output/";
      printParamsFilename = "pv.params";
      initializeFromCheckpointDir = nil; --"NULL" variable
      checkpointWrite = false;
      suppressLastOutput = false
   };
}

--Adding multiple groups, outermost group must be an array
pv.addMultiGroups(basicParams, 
--Start of groups to add
{
   --
   -- layers
   --
   -- Implicit key of 2 for same reason as above. This counter will keep incrementing for every non-keyed value specified
   Input = {
      groupType = "ImageLayer";
      nxScale = 1;
      nyScale = 1;
      inputPath = "input/sampleimage.png";
      nf = 1;
      phase = 0;
      writeStep = -1;
      sparseLayer = false;
      mirrorBCflag = false;
      valueBC = 0.0;
      useInputBCflag = false;
      inverseFlag = false ;
      normalizeLuminanceFlag = false;
      autoResizeFlag = false;
      offsetAnchor = "tl";
      offsetX = 0;
      offsetY = 0;
      padValue = false
   };

   --an output layer
   Output = {
      groupType = "ANNLayer";
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
      VThresh = infinity; --Infinity, user defined variable
      AMax = infinity;
      AMin = -infinity;
      AShift = 0.0;
      VWidth = 0.0;
   };

   --a connection
   InputToOutput = {
      groupType = "HyPerConn";
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
      rMax  = infinity;
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
      delay = 0;

      pvpatchAccumulateType = "Convolve";
      updateGSynFromPostPerspective = false;
   };
} --End of table constructor
) --End function call

--Adding a layer to the parameters
--PVModule.addGroup(baseParameterTable, group)
--Calling the pv module to add group to baseParameterTable
pv.addGroup(
   basicParams, --Base parameter table variable 
   "Output2", --Key value of the added group
   { --Parameters for the group
      groupType = "ANNLayer";
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
      VThresh = infinity; --Infinity, user defined variable
      AMax = infinity;
      AMin = -infinity;
      AShift = 0.0;
      VWidth = 0.0;
   }
) --End of function call

--This is the same as adding a group using the module
--However, accessing basicParams directly will provide no protection from clobbering
--This code will silently clobber the previously defined Output2 group
basicParams["Output2"] = 
{
   groupType = "ANNLayer";
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
   VThresh = infinity; --Infinity, user defined variable
   AMax = infinity;
   AMin = -infinity;
   AShift = 0.0;
   VWidth = 0.0;
}

--Function to include a previously defined group
--Third parameter is a group that overwrites/adds parameters
--Make a new ANNLayer named "Output3", including from "Output2"
--Change nxScale and nyScale to .5
pv.addGroup(basicParams,
   "Output3", --New group key/name
   basicParams["Output2"], --Previously defined group 
   {
      nxScale = .5; --Overwrite parameters
      nyScale = .5;
   }
)

--Parameter Sweep
pv.paramSweep(basicParams, "Output2", "nf", {2, 4, 6, 8});



--Prints out a PetaVision approved parameter file to the console
pv.printConsole(basicParams)

