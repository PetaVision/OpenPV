local pv = require "PVModule";

local PVSubnets = {};

local
  addLCASubnet,
  addScaleValueConn,
  addActivityMask,
  addMaxPoolingLayer,
  singleLayerPerceptron,
  backPropStep,
  deconvPath;

--------------
-- Wrappers --
--------------

-- These wrapper functions serve as a convenient interface to the
-- functions below, because all of the parameters can be specified
-- by name in any order. For example:
--
--  local subnets = require "PVSubnets";
--  subnets.addLCASubnet
--    { pvParams                      = params
--    , lcaLayerName                  = "S1"
--    , inputLayerName                = "Image"
--    , inputValueScale               = 1/math.sqrt(5 * 5)
--    , stride                        = 1
--
--    , lcaParams = { nf              = 96
--                  , VThresh         = 0.010
--                  , AShift          = 0.010
--                  , AMin            = 0
--                  , AMax            = INFINITY
--                  , timeConstantTau = 100
--                  , InitVType       = "UniformRandomV"
--                  , minV            = -1
--                  , maxV            = 0.05 
--                  }
--
--    , connParams = { nxp            = 5
--                   , nyp            = 5
--                   , plasticityFlag = true
--                   , momentumTau    = 100
--                   , dWMax          = 0.5 
--                   , weightInitType = "UniformRandomWeight"
--                   , wMinInit       = -1
--                   , wMaxInit        = 1
--                   , sparseFraction = 0.7
--                   , normalizeMethod = "normalizeL2"
--                   , strength       = 1
--                   }
--
--    , triggerLayerName              = "Image"
--    };


-- creates an error layer, lca layer, recon layer, and all
-- of the appropriate connections
function PVSubnets.addLCASubnet(args)
  addLCASubnet
    ( args.pvParams          -- your params table
    , args.lcaLayerName      -- desired LCA layer name
    , args.inputLayerName    -- the layer to be reconstructed
    , args.inputValueScale or 1 -- scale the input values
    , args.lcaParams         -- nf, threshold, InitV, tau, etc
    , args.stride            -- lca.scale = input.scale. / stride
    , args.connParams        -- patch size, plasticity, dW, momentum, etc
    , args.triggerLayerName  -- for weight update, displayPeriod, etc
    );
end

-- creates a layer and connection to scale the values of an input
function PVSubnets.addScaleValueConn(args)
  addScaleValueConn
    ( args.pvParams          -- params table
    , args.inputLayerName    -- the layer to value-scale

    , args.scaleLayerName    -- desired name for scaled layer
      or args.inputLayerName .. 'Scaled' -- default

    , args.scaleFactor       -- the scale factor
    , args.writeStep or -1   -- write step
    );
end

-- creates a masked version of 'unmaskedLayer' that only contains
-- activity where 'maskingLayer' is non-zero.
-- returns name of masked layer.
function PVSubnets.addActivityMask(args)
  return addActivityMask
    ( args.pvParams          -- params table
    , args.unmaskedLayerName -- layer to mask
    , args.maskingLayerName  -- layer to pull mask from
    , args.triggerLayerName  -- trigger layer (usually image)
    , args.writeStep or -1
    );
end

-- creates a max-pooled version of the input layer with the given stride
function PVSubnets.addMaxPoolingLayer(args)
  addMaxPoolingLayer
    ( args.pvParams          -- params table
    , args.inputLayerName    -- layer to max pool

    , args.poolingLayerName  -- desired name for pooling layer
      or args.inputLayerName .. 'MaxPool' -- default

    , args.stride            -- stride at which to max-pool
    , args.writeStep         -- writeStep for pooled layer
    );
end

-- creates a single-layer perceptron from given input to given ground-truth
function PVSubnets.singleLayerPerceptron(args)
  singleLayerPerceptron
    ( args.pvParams             -- params table
    , args.inputLayerNames      -- input to perceptron
    , args.groundTruthLayerName -- desired output of perceptron
    , args.connParams           -- learning / connection parameters
    , args.biasConnParams       -- bias connection parameters (if different)
    , args.triggerLayerName     -- trigger layer (usually image)
    , args.addDeltaInputLayer   -- whether to begin backpropagating the error
    , args.deltaErrScale        -- err scale for delta layer
    );
end

-- creates a single step in a backprop hierarchy, including activity mask
-- returns new masked delta layer name
function PVSubnets.backPropStep(args)
  return backPropStep
    ( args.pvParams                    -- params table
    , args.currentDeltaLayerName       -- name of current delta layer in back-net
    , args.forwardPostLayerName        -- name of current post layer in forward-net
    , args.forwardPreLayerName         -- name of current pre layer in forward-net
    , args.connParams or {}            -- learning / connection parameters
    , args.triggerLayerName            -- trigger layer name
    , args.learningDirection           -- "forward" (e.g. for convnet)
                                       -- or "backward" (e.g. for LCA)
    , args.errScale or 1               -- errScale for delta layer
    , args.createMask == nil and true  -- create mask unless otherwise specified
    )
end

-- create a deconvolutional reconstruction pathway from the start layer,
-- down through the path layers
function PVSubnets.deconvPath(args)
  deconvPath
    ( args.pvParams           -- params table
    , args.start              -- start layer
    , args.path               -- layer path
    , args.triggerLayerName   -- trigger layer name
    );
end






---------------------
-- Implementations --
---------------------

function addLCASubnet
  ( pvParams
  , lcaLayerName
  , inputLayerName
  , inputValueScale
  , lcaParams
  , stride
  , connParams
  , triggerLayerName
  )

  local displayPeriod = pvParams[triggerLayerName]['displayPeriod'];


  local inputLayer = pvParams[inputLayerName];
  local errorLayerName = inputLayerName .. 'Error_' .. lcaLayerName;
  local reconLayerName = inputLayerName .. 'Recon_' .. lcaLayerName;

  if inputValueScale ~= 1 then
    PVSubnets.addScaleValueConn
      { pvParams       = pvParams
      , inputLayerName = inputLayerName
      , scaleFactor    = inputValueScale
      , writeStep      = displayPeriod
      }
    inputLayerName = inputLayerName .. 'Scaled';
  end


  local inputToError = {
    groupType = "IdentConn";
    preLayerName = inputLayerName;
    postLayerName = errorLayerName;
    channelCode = 0;
    writeStep = -1;
  }
  pv.addGroup(pvParams, inputLayerName .. 'To' .. errorLayerName, inputToError);

  local errorLayer = {
    groupType        = "ANNNormalizedErrorLayer";
    nxScale          = inputLayer['nxScale'];
    nyScale          = inputLayer['nyScale'];
    nf               = inputLayer['nf'];

    phase            = inputLayer['phase'] + 1;

    InitVType        = "ZeroV";

    writeStep        = displayPeriod;
    initialWriteTime = displayPeriod;
  };
  pv.addGroup(pvParams, errorLayerName, errorLayer);

  local reconLayer = {
    groupType        = "ANNLayer";
    nxScale          = inputLayer['nxScale'];
    nyScale          = inputLayer['nyScale'];
    nf               = inputLayer['nf'];

    phase            = inputLayer['phase'] + 1;

    InitVType        = "ZeroV";

    writeStep        = displayPeriod;
    initialWriteTime = displayPeriod;

    triggerFlag = true;
    triggerLayerName = triggerLayerName;
    triggerOffset = 1;
  };
  pv.addGroup(pvParams, reconLayerName, reconLayer);


  local lcaLayer = {
    groupType = "HyPerLCALayer";
    nxScale = inputLayer['nxScale'] / stride;
    nyScale = inputLayer['nyScale'] / stride;

    phase = errorLayer['phase'] + 1;

    sparseLayer = true;
    updateGpu  = true;

    writeStep        = displayPeriod;
    initialWriteTime = displayPeriod;
  };
  for k,v in pairs(lcaParams) do lcaLayer[k] = v; end
  pv.addGroup(pvParams, lcaLayerName, lcaLayer);

  local lcaToError = {
    groupType = "MomentumConn";

    preLayerName = lcaLayerName;
    postLayerName = errorLayerName;
    channelCode = 1;

    nxp = connParams['nxp'];
    nyp = connParams['nyp'];
    nfp = errorLayer['nf'];

    momentumMethod = "viscosity";
    momentumTau = connParams['momentumTau'];

    triggerFlag = connParams['plasticityFlag'];
    triggerLayerName = connParams['plasticityFlag']
                   and triggerLayerName or nil;

    triggerOffset    = connParams['plasticityFlag'] and 0 or nil;

    writeStep = -1;
  };
  for k,v in pairs(connParams) do lcaToError[k] = v; end
  pv.addGroup(pvParams, lcaLayerName .. 'To' .. errorLayerName, lcaToError);

  local errorToLca = {
    groupType = "TransposeConn";
    preLayerName = errorLayerName;
    postLayerName = lcaLayerName;
    originalConnName = lcaLayerName .. 'To' .. errorLayerName;
    channelCode = 0;
    writeStep = -1;
    receiveGpu = true;
    updateGSynFromPostPerspective = true;
  };
  pv.addGroup(pvParams, errorLayerName .. 'To' .. lcaLayerName, errorToLca);

  local lcaToRecon = {
    groupType = "CloneConn";
    preLayerName = lcaLayerName;
    postLayerName = reconLayerName;
    originalConnName = lcaLayerName .. 'To' .. errorLayerName;
    channelCode = 0;
    writeStep = -1;
  };
  pv.addGroup(pvParams, lcaLayerName .. 'To' .. reconLayerName, lcaToRecon);

end


function addScaleValueConn
  ( pvParams
  , inputLayerName
  , scaledLayerName
  , scaleFactor
  , writeStep
  )
  local inputLayer = pvParams[inputLayerName];

  local scaledLayer = {
    groupType        = "ANNLayer";
    nxScale          = inputLayer['nxScale'];
    nyScale          = inputLayer['nyScale'];
    nf               = inputLayer['nf'];

    phase            = inputLayer['phase'] + 1;

    InitVType        = "ZeroV";

    writeStep        = writeStep;
    initialWriteTime = writeStep;

    triggerFlag      = true;
    triggerLayerName = inputLayerName;
  };

  local scaleConn = {
    groupType        = "HyPerConn";
    preLayerName     = inputLayerName;
    postLayerName    = scaledLayerName;
    weightInitType   = "OneToOneWeights";
    weightInit       = scaleFactor;
    nxp              = 1;
    nyp              = 1;
    nfp              = inputLayer['nf'];
    normalizeMethod  = "none";
    channelCode      = 0;
    plasticityFlag  = false;
    writeStep        = -1;
  };

  pv.addGroup(pvParams, scaledLayerName, scaledLayer);
  pv.addGroup(pvParams, inputLayerName .. 'To' .. scaledLayerName, scaleConn);

end


function addActivityMask
  ( pvParams
  , unmaskedLayerName
  , maskingLayerName
  , triggerLayerName
  , writeStep
  )

  local unmaskedLayer = pvParams[unmaskedLayerName];
  local maskingLayer  = pvParams[maskingLayerName];

  local maskLayerName   = maskingLayerName .. 'Mask';
  local maskedLayerName = unmaskedLayerName .. 'MaskedBy' .. maskingLayerName;

  local displayPeriod = pvParams[triggerLayerName]['displayPeriod'];

  local maskLayer = {
    groupType = "ANNLayer";
    nxScale   = maskingLayer['nxScale'];
    nyScale   = maskingLayer['nyScale'];
    nf        = maskingLayer['nf'];

    phase     = maskingLayer['phase'] + 1;

    InitVType        = "ZeroV";

    writeStep        = -1;

    triggerFlag = true;
    triggerLayerName = triggerLayerName;
    triggerOffset = 1;

    verticesV = {0,0};
    verticesA = {1,0};
    slopeNegInf = 0;
    slopePosInf = 0;

    sparseLayer = maskingLayer['sparseLayer'];
  };
  pv.addGroup(pvParams, maskLayerName, maskLayer);


  local maskingLayerToMaskLayer = {
    groupType = "IdentConn";
    preLayerName  = maskingLayerName;
    postLayerName = maskLayerName;
    channelCode = 1;
    writeStep = -1;
  };
  pv.addGroup( pvParams
             , maskingLayerName .. 'To' .. maskLayerName
             , maskingLayerToMaskLayer
             );


  local maskedLayer = {
    groupType = "PtwiseProductLayer";
    nxScale = unmaskedLayer['nxScale'];
    nyScale = unmaskedLayer['nyScale'];
    nf      = unmaskedLayer['nf'];

    phase = math.max(maskLayer['phase'], unmaskedLayer['phase']) + 1;

    InitVType = "ZeroV";

    writeStep = writeStep;
    initalWriteTime = writeStep;
  };
  pv.addGroup(pvParams, maskedLayerName, maskedLayer);


  for i,layer in ipairs( { unmaskedLayerName, maskLayerName }) do

    local channel = i - 1;
    local operandConn = {
      groupType = "IdentConn";
      preLayerName = layer;
      postLayerName = maskedLayerName;
      channelCode = channel;
      writeStep = -1;
    };
    pv.addGroup(pvParams, layer .. 'To' .. maskedLayerName, operandConn);

  end
  return maskedLayerName;
end


function addMaxPoolingLayer
  ( pvParams
  , inputLayerName
  , poolingLayerName
  , stride
  , writeStep
  );

  local inputLayer = pvParams[inputLayerName];

  local poolingLayer = {
    groupType = "ANNLayer";
    nxScale   = inputLayer['nxScale'] / stride;
    nyScale   = inputLayer['nyScale'] / stride;
    nf        = inputLayer['nf'];

    phase     = inputLayer['phase'] + 1;

    InitVType = "ZeroV";

    writeStep = writeStep;
    initialWriteTime = writeStep;

    sparseLayer = inputLayer['sparseLayer'];
  };
  pv.addGroup(pvParams, poolingLayerName, poolingLayer);

  local poolingConn = {
    groupType = "PoolingConn";
    preLayerName = inputLayerName;
    postLayerName = poolingLayerName;
    nxp = 1;
    nyp = 1;
    pvpatchAccumulateType = "maxpooling";

    channelCode = 0;

    updateGSynFromPostPerspective = true;

    needPostIndexLayer = true;
    postIndexLayerName = poolingLayerName .. "Index";

    writeStep = -1;
  };
  pv.addGroup(pvParams, inputLayerName .. 'To' .. poolingLayerName, poolingConn);

  local poolingIdxLayer = {
    groupType = "PoolingIndexLayer";

    nxScale = poolingLayer['nxScale'];
    nyScale = poolingLayer['nyScale'];
    nf      = poolingLayer['nf'];

    writeStep = -1;

    phase = poolingLayer['phase'] + 1;
  };
  pv.addGroup(pvParams, poolingLayerName .. "Index", poolingIdxLayer);

end


function singleLayerPerceptron
  ( pvParams
  , inputLayerNames
  , groundTruthLayerName
  , inputConnParams
  , biasConnParams
  , triggerLayerName
  , addDeltaInputLayer
  , deltaErrScale
  )

  -- table-ize non-table values for single input case
  if  type(inputConnParams[1]) ~= "table" and 
      type(inputLayerNames)    ~= "table" then

    local connParams = pv.deepCopy(inputConnParams);
    inputConnParams = {};
    inputConnParams[inputLayerNames] = connParams;
    inputLayerNames = { inputLayerNames };

  -- replicate params if only one passed in
  elseif type(inputLayerNames)    == "table" and
         type(inputConnParams[1]) ~= "table" then

    local connParams = pv.deepCopy(inputConnParams);
    inputConnParams = {};
    for _,inputLayerName in pairs(inputLayerNames) do
      inputConnParams[inputLayerName] = pv.deepCopy(connParams);
    end

  -- throw error if more connections then layers
  elseif type(inputLayerNames) ~= "table" and
         type(inputConnParams) == "table" then

    error('type mismatch: multiple connection params passed in for only one layer');
  end 

  -- replicate input params for bias if not passed in
  if biasConnParams == nil then
    biasConnParams = pv.deepCopy(inputConnParams[inputLayerNames[1]]);
  end

  allConnParams = pv.deepCopy(inputConnParams)
  allConnParams['biasConn'] = biasConnParams;

  -- Error checking
  for _,connParams in pairs(allConnParams) do
    if connParams['normalizeMethod'] ~= nil    and
       connParams['normalizeMethod'] ~= 'none' then
      error('perceptron connections are not normalized!');
    end

    if connParams['plasticityFlag'] and not connParams['dWMax'] then
      error('plasticity on but no dWMax given');
    end

    if connParams['groupType'] ~= "CloneConn" and
       connParams['groupType'] ~= "PlasticCloneConn" then

      for _,required in pairs({'nxp', 'nyp', 'weightInitType', 'plasticityFlag'}) do
        if connParams[required] == nil then
          error(string.format('%s is required', required))
        end
      end

    end

  end


  local displayPeriod    = pvParams[triggerLayerName]['displayPeriod'];
  local groundTruthLayer = pvParams[groundTruthLayerName];


  local reconLayerName   = groundTruthLayerName .. 'Recon_';
  local maxPhase = 0;
  for _,inputLayerName in pairs(inputLayerNames) do

    local inputLayer       = pvParams[inputLayerName];
    if inputLayer['phase'] > maxPhase then
      maxPhase = inputLayer['phase'];
    end

    reconLayerName = reconLayerName .. inputLayerName;

  end

  local deltaLayerName   = 'Delta' .. reconLayerName;
  local biasLayerName    = groundTruthLayerName .. 'Bias';


  -- Layers
  local reconLayer = {
    groupType = "ANNLayer";
    nxScale   = groundTruthLayer['nxScale'];
    nyScale   = groundTruthLayer['nyScale'];
    nf        = groundTruthLayer['nf'];

    phase = maxPhase + 1;

    InitVType = "ZeroV";

    writeStep        = displayPeriod;
    initialWriteTime = displayPeriod;

    triggerFlag = true;
    triggerLayerName = triggerLayerName;
    triggerOffset = 1;
  };
  pvParams[reconLayerName] = reconLayer;

  local deltaLayer = pv.deepCopy(reconLayer);
  deltaLayer['groupType'] = "ANNErrorLayer";
  deltaLayer['phase']     = reconLayer['phase'] + 1;
  pvParams[deltaLayerName] = deltaLayer;

  local biasLayer = {
    groupType = "ConstantLayer";
    nxScale   = 1/pvParams['column']['nx'];
    nyScale   = 1/pvParams['column']['ny'];
    nf        = 1;

    valueV    = 1;
    writeStep = -1;
    phase     = 0;
  };
  pvParams[biasLayerName] = biasLayer;

  -- Connections
  local groundTruthToDelta = {
    groupType     = "IdentConn";
    preLayerName  = groundTruthLayerName;
    postLayerName = deltaLayerName;
    channelCode   = 0;
    writeStep     = -1;
  };
  pvParams[groundTruthLayerName .. 'To' .. deltaLayerName] = groundTruthToDelta;

  local reconToDelta = {
    groupType     = "IdentConn";
    preLayerName  = reconLayerName;
    postLayerName = deltaLayerName;
    channelCode   = 1;
    writeStep     = -1;
  };
  pvParams[reconLayerName .. 'To' .. deltaLayerName] = reconToDelta;

  for _,inputLayerName in pairs(inputLayerNames) do
    local inputConn = inputConnParams[inputLayerName];

    local inputToDelta = {
      groupType = "HyPerConn";
      preLayerName = inputLayerName;
      postLayerName = deltaLayerName;
      nfp = deltaLayer['nfp'];

      channelCode = -1;
      normalizeMethod  = inputConn['normalizeMethod'] or 'none';

      triggerFlag      = inputConn['plasticityFlag'];
      triggerLayerName = inputConn['plasticityFlag'] and triggerLayerName or nil;
      triggerOffset    = inputConn['plasticityFlag'] and 0 or nil;

      writeStep = -1;
    };
    for k,v in pairs(inputConn) do inputToDelta[k] = v end
    pvParams[inputLayerName .. 'To' .. deltaLayerName] = inputToDelta;

    local inputToRecon = {
      groupType = "CloneConn";
      preLayerName = inputLayerName;
      postLayerName = reconLayerName;
      originalConnName = inputLayerName .. 'To' .. deltaLayerName;

      channelCode = 0;

      writeStep = -1;
    };
    pvParams[inputLayerName .. 'To' .. reconLayerName] = inputToRecon;
  end


  local biasToDelta = {
    groupType = "HyPerConn";
    preLayerName = biasLayerName;
    postLayerName = deltaLayerName;
    nfp = deltaLayer['nfp'];

    channelCode = -1;
    normalizeMethod  = biasConnParams['normalizeMethod'] or 'none';

    triggerFlag      = biasConnParams['plasticityFlag'];
    triggerLayerName = biasConnParams['plasticityFlag'] and triggerLayerName or nil;
    triggerOffset    = biasConnParams['plasticityFlag'] and 0 or nil;

    writeStep = -1;
  }
  for k,v in pairs(biasConnParams) do biasToDelta[k] = v end
  pvParams[biasLayerName .. 'To' .. deltaLayerName] = biasToDelta;

  local biasToRecon  = {
    groupType = "CloneConn";
    preLayerName = biasLayerName;
    postLayerName = reconLayerName;
    originalConnName = biasLayerName .. 'To' .. deltaLayerName;

    channelCode = 0;

    writeStep = -1;
  };
  pvParams[biasLayerName .. 'To' .. reconLayerName] = biasToRecon;

  -- Add extra layer and connection if perceptron will be used to start backprop
  if addDeltaInputLayer then

    for _,inputLayerName in pairs(inputLayerNames) do

      local inputLayer = pvParams[inputLayerName];
      local deltaInputLayerName = 'Delta' .. inputLayerName;

      local deltaInputLayer = {
        groupType = "ANNErrorLayer";
        nxScale   = inputLayer['nxScale'];
        nyScale   = inputLayer['nyScale'];
        nf        = inputLayer['nf'];

        phase = deltaLayer['phase'] + 1;

        InitVType = "ZeroV";

        writeStep = displayPeriod;
        initialWriteTime = displayPeriod;

        triggerFlag = true;
        triggerLayerName = triggerLayerName;
        triggerOffset = 1;

        errScale = deltaErrScale;
      };
      pvParams[deltaInputLayerName] = deltaInputLayer;

      pvParams[deltaLayerName .. 'To' .. deltaInputLayerName] = {
        groupType = "TransposeConn";
        preLayerName = deltaLayerName;
        postLayerName = deltaInputLayerName;
        originalConnName = inputLayerName .. 'To' .. deltaLayerName;

        channelCode = 1;
        writeStep = -1;

        receiveGpu = true;
        updateGSynFromPostPerspective = true;
      };

    end

  end

end

function backPropStep
  ( pvParams
  , currentDeltaLayerName
  , forwardPostLayerName
  , forwardPreLayerName
  , connParams
  , triggerLayerName
  , learningDirection
  , errScale
  , createMask
  )


  local displayPeriod = pvParams[triggerLayerName]['displayPeriod'];

  local currentDeltaLayer     = pvParams[currentDeltaLayerName];
  local forwardPostLayer      = pvParams[forwardPostLayerName];
  local forwardPreLayer       = pvParams[forwardPreLayerName];
  local newDeltaLayerName     = 'Delta' .. forwardPreLayerName;

  local originalConnName = forwardPreLayerName .. 'To' .. forwardPostLayerName;
  local originalConn = pvParams[originalConnName];
  local isPoolingConn =
    originalConn and originalConn['groupType'] == "PoolingConn"

  if not isPoolingConn then
    if connParams['plasticityFlag'] and not connParams['dWMax'] then
      error('plasticity on but no dWMax given');
    end

    for _,required in pairs(
      {'nxp', 'nyp', 'weightInitType', 'plasticityFlag','normalizeMethod'}
    ) do

      if connParams[required] == nil then
        error(string.format('%s is required', required))
      end
    end


  end

  -- create delta version of forward pre layer
  local newDeltaLayer = {
    groupType = "ANNErrorLayer";
    nxScale   = forwardPreLayer['nxScale'];
    nyScale   = forwardPreLayer['nyScale'];
    nf        = forwardPreLayer['nf'];

    phase = currentDeltaLayer['phase'] + 1;

    writeStep = displayPeriod;
    initialWriteTime = displayPeriod;

    InitVType = "ZeroV";

    triggerFlag = true;
    triggerLayerName = triggerLayerName;
    triggerOffset = 1;

    errScale = errScale;

    sparseLayer = isPoolingConn;
  };
  pv.addGroup(pvParams, newDeltaLayerName, newDeltaLayer);

  local newLearningConn, newBackwardConn, newForwardConn;
  -- if forward connection is pooling connection, don't learn, just
  -- invert pooling operation
  if isPoolingConn then

    newBackwardConn = {
      groupType = "TransposePoolingConn";
      preLayerName = currentDeltaLayerName;
      postLayerName = newDeltaLayerName;
      channelCode = 0;

      originalConnName = originalConnName;

      writeStep = -1;

      pvpatchAccumulateType = "maxpooling";
    };

  -- learning connection proceeds up the hierarchy (e.g. for convnet)
  elseif learningDirection == "forward" then

    newLearningConn = {
      groupType        = "HyPerConn";
      preLayerName     = forwardPreLayerName;
      postLayerName    = currentDeltaLayerName;

      triggerFlag      = connParams['plasticityFlag'];
      triggerLayerName = connParams['plasticityFlag'] and triggerLayerName or nil;
      triggerOffset    = connParams['plasticityFlag'] and 0 or nil;

      channelCode = -1;

      writeStep = -1;
    };
    pvParams[forwardPreLayerName .. 'To' .. currentDeltaLayerName] =
      newLearningConn;

    newBackwardConn = {
      groupType = "TransposeConn";
      preLayerName = currentDeltaLayerName;
      postLayerName = newDeltaLayerName;
      originalConnName = forwardPreLayerName .. 'To' .. currentDeltaLayerName;

      receiveGpu = forwardPreLayer['sparseLayer'];
      updateGSynFromPostPerspective = forwardPreLayer['sparseLayer'];
      channelCode = 0;
      writeStep = -1;
    };

    newForwardConn = {
      groupType = "CloneConn";
      preLayerName = forwardPreLayerName;
      postLayerName = forwardPostLayerName;
      originalConnName = forwardPreLayerName .. 'To' .. currentDeltaLayerName;
      channelCode = 0;

      writeStep = -1;
    };

  -- learning connection proceeds down the hierarchy (e.g. for LCA)
  elseif learningDirection == "backward" then

    newLearningConn = {
      groupType        = "HyPerConn";
      preLayerName     = forwardPostLayerName;
      postLayerName    = newDeltaLayerName;

      triggerFlag      = connParams['plasticityFlag'];
      triggerLayerName = connParams['plasticityFlag'] and triggerLayerName or nil;
      triggerOffset    = connParams['plasticityFlag'] and 0 or nil;

      channelCode = -1;

      writeStep = -1;
    };
    pvParams[forwardPostLayerName .. 'To' .. newDeltaLayerName] =
      newLearningConn;

    newBackwardConn = {
      groupType = "CloneConn";
      preLayerName = currentDeltaLayerName;
      postLayerName = newDeltaLayerName;
      originalConnName = forwardPostLayerName .. 'To' .. newDeltaLayerName;
      channelCode = 0;

      writeStep = -1;
    };

    newForwardConn = {
      groupType = "TransposeConn";
      preLayerName = forwardPreLayerName;
      postLayerName = forwardPostLayerName;
      originalConnName = forwardPostLayerName .. 'To' .. newDeltaLayerName;

      receiveGpu = forwardPostLayer['sparseLayer'];
      updateGSynFromPostPerspective = forwardPostLayer['sparseLayer'];
      channelCode = 0;
      writeStep = -1;
    };

  else
    error("invalid learning direction:" ..
          "valid directions are 'forward' and 'backward'");
  end

  for k,v in pairs(connParams) do newLearningConn[k] = v; end

  pvParams[currentDeltaLayerName .. 'To' .. newDeltaLayerName] =
    newBackwardConn;

  if not isPoolingConn then
    pvParams[forwardPreLayerName .. 'To' .. forwardPostLayerName] =
      newForwardConn;
  end

  if createMask then
    local maskedDeltaLayerName = PVSubnets.addActivityMask
      { pvParams          = pvParams
      , unmaskedLayerName = newDeltaLayerName
      , maskingLayerName  = forwardPreLayerName
      , triggerLayerName  = triggerLayerName
      , writeStep         = displayPeriod
      }
    return maskedDeltaLayerName;
  else
    return newDeltaLayerName;
  end
end



local function deconvStep
  ( pvParams
  , currentReconName
  , originalPreName
  , originalPostName
  , triggerLayerName
  , writeOut
  )
  local displayPeriod = pvParams[triggerLayerName]['displayPeriod'];

  local currentReconLayer  = pvParams[currentReconName];

  local originalConnectionName = originalPreName .. 'To' .. originalPostName
  local originalConnection = pvParams[originalConnectionName];
  local originalPreLayer   = pvParams[originalPreName];

  local newPreName  = currentReconName;
  local newPostName = originalPreName .. 'Recon_' .. currentReconName;
  local newConnName = newPreName .. 'To' .. newPostName;


  local newPostLayer = {
    groupType = "ANNLayer";
    nxScale = originalPreLayer['nxScale'];
    nyScale = originalPreLayer['nyScale'];
    nf      = originalPreLayer['nf'];

    phase = currentReconLayer['phase'] + 1;

    triggerFlag = true;
    triggerLayerName = triggerLayerName;
    triggerOffset = 1;

    InitVType = "ZeroV";

    writeStep = writeOut and displayPeriod or -1;
    initialWriteTime = writeOut and displayPeriod or nil;
  };
  pv.addGroup(pvParams, newPostName, newPostLayer);

  isPoolingConn = originalConnection['groupType'] == "PoolingConn";
  isIdentConn   = originalConnection['groupType'] == "IdentConn";

  local deconvConn = {
    preLayerName  = newPreName;
    postLayerName = newPostName;

    groupType =
         isPoolingConn and "TransposePoolingConn"
      or isIdentConn and "CloneConn"
      or "TransposeConn";

    receiveGpu = not (isPoolingConn or isIdentConn);

    updateGSynFromPostPerspective = not (isPoolingConn or isIdentConn);

    channelCode = 0;

    originalConnName = originalConnectionName;

    writeStep = -1;

    pvpatchAccumulateType = originalConnection['pvpatchAccumulateType'];
  };
  pv.addGroup(pvParams, newConnName, deconvConn);

  return newPostName
end

function deconvPath(pvParams, start, path, trigger)
  for idx,_ in ipairs(path) do

    if path[idx+2] then
      start = deconvStep
        ( pvParams     -- params
        , start        -- start
        , path[idx+1] -- origpre
        , path[idx]   -- origpost
        , trigger      -- trigger
        , false        -- writeout
        );
    else
      deconvStep
        ( pvParams     -- params
        , start        -- start
        , path[idx+1] -- origpre
        , path[idx]   -- origpost
        , trigger      -- trigger
        , true        -- writeout
        );
      break;
    end

  end

end


return PVSubnets;
