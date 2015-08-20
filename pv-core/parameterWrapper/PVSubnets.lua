local pv = require "PVModule";

local PVSubnets = {};

local 
  addLCASubnet,
  addScaleValueConn, 
  addActivityMask,
  addMaxPoolingLayer;

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
--    , inputLayerName                = "ImageScaled"
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
    , args.scaleFactor       -- the scale factor
    );
end

-- creates a masked version of 'unmaskedLayer' that only contains
-- activity where 'maskingLayer' is non-zero.
function PVSubnets.addActivityMask(args)
  addActivityMask
    ( args.pvParams          -- params table
    , args.unmaskedLayerName -- layer to mask
    , args.maskingLayerName  -- layer to pull mask from
    , args.triggerLayerName  -- trigger layer (usually image)
    );
end

-- creates a max-pooled version of the input layer with the given stride
function PVSubnets.addMaxPoolingLayer(args)
  addMaxPoolingLayer
    ( args.pvParams          -- params table
    , args.inputLayerName    -- layer to max pool
    , args.stride            -- stride at which to max-pool
    , args.writeStep         -- writeStep for pooled layer
    );
end



---------------------
-- Implementations --
---------------------

function addLCASubnet
  ( pvParams
  , lcaLayerName
  , inputLayerName
  , lcaParams
  , stride
  , connParams
  , triggerLayerName
  ) 

  local displayPeriod = pvParams[triggerLayerName]['displayPeriod'];

  local inputLayer = pvParams[inputLayerName];
  local errorLayerName = inputLayerName .. 'Error_' .. lcaLayerName;
  local reconLayerName = inputLayerName .. 'Recon_' .. lcaLayerName;

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

    triggerOffset    = connParams['plasticityFlag'] and 1 or nil;

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


function addScaleValueConn(pvParams, inputLayerName, scaleFactor)
  local inputLayer = pvParams[inputLayerName];

  local scaledLayerName = inputLayerName .. "Scaled";

  local scaledLayer = {
    groupType        = "ANNLayer";
    nxScale          = inputLayer['nxScale'];
    nyScale          = inputLayer['nyScale'];
    nf               = inputLayer['nf'];

    phase            = inputLayer['phase'] + 1; 
    
    InitVType        = "ZeroV";

    writeStep        = -1;

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
  )

  local unmaskedLayer = pvParams[unmaskedLayerName];
  local maskingLayer  = pvParams[maskingLayerName];

  local maskLayerName   = maskingLayerName .. 'Mask';
  local maskedLayerName = unmaskedLayerName .. 'MaskedBy' .. maskingLayerName;

  local displayPeriod = pvParams[triggerLayerName]['displayPeriod'];

  local maskLayer = {
    groupType = "PtwiseLinearTransferLayer";
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

    writeStep = -1; 
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
 
end


function addMaxPoolingLayer
  ( pvParams
  , inputLayerName
  , stride
  , writeStep
  );

  local poolingLayerName = inputLayerName .. "MaxPool";
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



-- this really needs to be cleaned up -- I'd avoid using it for now
function PVSubnets.deconvStep
  ( pvParams
  , currentReconName
  , originalConnectionName
  , triggerLayerName
  , writeOut
  )
  local displayPeriod = pvParams[triggerLayerName]['displayPeriod'];

  local currentReconLayer  = pvParams[currentReconName];

  local originalConnection = pvParams[originalConnectionName];
  local originalPostName   = originalConnection['postLayerName'];
  local originalPreName    = originalConnection['preLayerName'];
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
    
  local deconvConn = {
    preLayerName = newPreName;
    postLayerName = newPostName;

    groupType = originalConnection['groupType'] == "PoolingConn"
             and "TransposePoolingConn" or "TransposeConn";
    
    receiveGpu = not originalConnection['groupType'] == "PoolingConn";
    updateGSynFromPostPerspective = 
      not originalConnection['groupType'] == "PoolingConn";
   
    channelCode = 0; 

    originalConnName = originalConnectionName;

    writeStep = -1;

    pvpatchAccumulateType = originalConnection['pvpatchAccumulateType'];
  };
  pv.addGroup(pvParams, newConnName, deconvConn);

 
end

return PVSubnets;
