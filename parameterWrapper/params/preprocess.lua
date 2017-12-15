local preprocess = {}
--A parameter generator that spits out a downsampled by 2 (each dimension), whitened, and rescaled (based on LCA rescaling) parameter file
--

-- Returns a group 
-- inputLayer and outputLayer will not get changed, but will be used to get information
-- 3 layers, so out phase should be at least phase + 4
function preprocess.ds2_white_rescale(prefix, inputLayerName, inputLayer, outputLayerName, outputLayer, normPatchSize)
   --Parameter handeling 
   assert(type(prefix) == "string")
   --Input layer must be specified
   assert(type(inputLayer) == "table")
   assert(type(outputLayer) == "table")
   --Default values for parameters
   --if(rcenter == nil) then rcenter = 1 end
   --if(rsurround == nil) then rsurround = 11 end
   if(normPatchSize == nil) then
      print("normPatchSize must be specified")
      os.exit()
   end

   local rcenter = 1
   local rsurround = 11

   --Grab various necessary parameters from inputLayer
   --These are all parameters that are required to exist
   local nxScaleDownsample = inputLayer["nxScale"]/2
   local nyScaleDownsample = inputLayer["nyScale"]/2
   local nf = inputLayer["nf"]
   local startingPhase = inputLayer["phase"]

   --Check outputLayer dimensions and variables
   local endingPhase = outputLayer["phase"]
   if(endingPhase ~= startingPhase + 4) then
      print("outputLayer phase mismatch. Target outputLayer phase: " .. startingPhase + 4 .. "  Actual outputLayerphase: " .. endingPhase)
      os.exit()
   end

   local outNxScale = outputLayer["nxScale"]
   local outNyScale = outputLayer["nyScale"]
   if(outNxScale ~= nxScaleDownsample) then
      print("outputLayer nxScale mismatch. target outputLayer nxScale: " .. nxScaleDownsample .. "  actual outputLayer nxScale: " .. outNxScale)
      os.exit()
   end
   if(outNyScale ~= nyScaleDownsample) then
      print("outputLayer nyScale mismatch. target outputLayer nyScale: " .. nyScaleDownsample .. "  actual outputLayer nyScale: " .. outNyScale)
      os.exit()
   end

   --Type and nil checks
   assert(type(inputLayerName) == "string")
   assert(type(outputLayerName) == "string")
   assert(type(nxScaleDownsample) == "number")
   assert(type(nyScaleDownsample) == "number")
   assert(type(nf) == "number")
   assert(type(startingPhase) == "number")

   local whiteParams = {}
      --layers
   whiteParams[prefix .. "Bipolar"] = {
      groupType = "ANNLayer";
      nxScale                             = nxScaleDownsample;
      nyScale                             = nyScaleDownsample;
      nf                                  = nf;
      phase                               = startingPhase + 1;
      mirrorBCflag                        = true;
      InitVType                           = "ZeroV";
      triggerFlag                         = true;
      triggerLayerName                    = inputLayerName;
      triggerOffset                       = 0;
      writeStep                           = -1;
      sparseLayer                         = false; --How do we specify if the layer is sparse?
      updateGpu                           = false;
      VThresh                             = -INFINITY;
      AMin                                = -INFINITY;
      AMax                                = INFINITY;
      AShift                              = 0;
      VWidth                              = 0;
   }

   whiteParams[prefix .. "Ganglion"] = {
      groupType = "ANNLayer";
      nxScale                             = nxScaleDownsample;
      nyScale                             = nyScaleDownsample;
      nf                                  = nf;
      phase                               = startingPhase + 2;
      mirrorBCflag                        = true;
      InitVType                           = "ZeroV";
      triggerFlag                         = true;
      triggerLayerName                    = inputLayerName;
      triggerOffset                       = 0;
      writeStep                           = -1;
      sparseLayer                         = false;
      updateGpu                           = false;
      VThresh                             = -INFINITY;
      AMin                                = -INFINITY;
      AMax                                = INFINITY;
      AShift                              = 0;
      VWidth                              = 0;
   }

   whiteParams[prefix .. "Rescale"] = {
      groupType = "RescaleLayer";
      restart                         = false;
      originalLayerName               = prefix .. "Ganglion";
      nxScale                         = nxScaleDownsample;
      nyScale                         = nyScaleDownsample;
      nf                              = nf;
      mirrorBCflag                    = true;
      writeStep                       = -1;
      initialWriteTime                = -1;
      writeSparseActivity             = false;
      rescaleMethod                   = "l2";
      patchSize                       = normPatchSize;
      valueBC = 0;
      phase                           = startingPhase + 3;
      triggerFlag                         = true;
      triggerLayerName = inputLayerName;
   }

   --Connections
   whiteParams[inputLayerName .. "To" .. prefix .. "Bipolar"] = {
      groupType = "HyPerConn";
      preLayerName                        = inputLayerName;
      postLayerName                       = prefix .. "Bipolar";
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
      nxp                                 = 3;
      nyp                                 = 3;
      nfp                                 = 1;
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
      sharedWeights                       = true;
   }

   whiteParams[prefix .. "BipolarTo" .. prefix .. "GanglionCenter"] = {
      groupType = "HyPerConn";
      preLayerName                        = prefix .. "Bipolar";
      postLayerName                       = prefix .. "Ganglion";
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
      nxp                                 = rcenter;
      nyp                                 = rcenter;
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
      sharedWeights = true;
   }

   whiteParams[prefix .. "BipolarTo" .. prefix .. "GanglionSurround"] = {
      groupType = "HyPerConn";
      preLayerName                        = prefix .. "Bipolar";
      postLayerName                       = prefix .. "Ganglion";
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
      nxp                                 = rsurround;
      nyp                                 = rsurround;
      nfp                                 = 1;
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
      sharedWeights                       = true;
   }

      -- Connection for connection to output layer
   whiteParams[prefix .. "RescaleTo" .. outputLayerName] = {
      groupType                           = "IdentConn";
      preLayerName                        = prefix .. "Rescale";
      postLayerName                       = outputLayerName;
      channelCode                         = 0;
      delay                               = 0.000000;
      writeStep                           = -1;
   }
   return whiteParams

end -- endFunction

return preprocess
