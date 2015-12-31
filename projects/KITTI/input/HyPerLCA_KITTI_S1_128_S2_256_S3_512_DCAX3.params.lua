
-- Sun Nov 15 13:39:49 2015
package.path = package.path .. ";" .. "/home/gkenyon/openpv/pv-core/parameterWrapper/?.lua"
--package.path = package.path .. ";" .. "/Users/gkenyon/openpv/pv-core/parameterWrapper/?.lua"
local pv = require "PVModule"

--HyPerCol parameters
local dtAdaptFlag              = not S1_Movie
local useAdaptMethodExp1stOrder = true
local dtAdaptController        = "EnergyProbe"
local dtAdaptTriggerLayerName  = "ImageLeft";
local dtScaleMax               = 0.05    --1.0     -- minimum value for the maximum time scale, regardless of tau_eff
local dtScaleMin               = 0.0005  --0.01    -- default time scale to use after image flips or when something is wacky
local dtChangeMax              = 0.005   --0.1     -- determines fraction of tau_effective to which to set the time step, can be a small percentage as tau_eff can be huge
local dtChangeMin              = 0.0005  --0.01    -- percentage increase in the maximum allowed time scale whenever the time scale equals the current maximum
local dtMinToleratedTimeScale  = 0.00001

-- Base table variable to store
local pvParameters = {
column = {
groupType = "HyPerCol";
    startTime                           = 0;
    dt                                  = 1;
      dtAdaptFlag                         = dtAdaptFlag;
      useAdaptMethodExp1stOrder           = useAdaptMethodExp1stOrder;
      dtAdaptController                   = dtAdaptController;
      dtAdaptTriggerLayerName             = dtAdaptTriggerLayerName;
      dtScaleMax                          = dtScaleMax;    
      dtScaleMin                          = dtScaleMin;
      dtChangeMax                         = dtChangeMax;
      dtChangeMin                         = dtChangeMin;
      dtMinToleratedTimeScale             = dtMinToleratedTimeScale;
    stopTime                            = 1.90608e+07;
    progressInterval                    = 1200;
    writeProgressToErr                  = true;
    verifyWrites                        = false;
    outputPath                          = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train7";
    printParamsFilename                 = "KITTI_S1_128_S2_256_S3_512_DCA_train7.params";
    randomSeed                          = 1234567890;
    nx                                  = 512;
    ny                                  = 152;
    nbatch                              = 1;
    filenamesContainLayerNames          = 2;
    filenamesContainConnectionNames     = 2;
    initializeFromCheckpointDir         = ""; --"/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train7/Checkpoints/Checkpoint";
    defaultInitializeFromCheckpointFlag = false;
    checkpointWrite                     = true;
    checkpointWriteDir                  = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train7/Checkpoints";
    checkpointWriteTriggerMode          = "step";
    checkpointWriteStepInterval         = 12000;
    deleteOlderCheckpoints              = false;
    suppressNonplasticCheckpoints       = false;
    writeTimescales                     = true;
    errorOnNotANumber                   = false;
};

ImageLeft = {
groupType = "Movie";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 0;
    mirrorBCflag                        = true;
    initializeFromCheckpointFlag        = false;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    inputPath                           = "/nh/compneuro/Data/KITTI/list/image_02.txt";
    offsetAnchor                        = "tl";
    offsetX                             = 0;
    offsetY                             = 0;
    writeImages                         = 0;
    inverseFlag                         = false;
    normalizeLuminanceFlag              = true;
    normalizeStdDev                     = true;
    jitterFlag                          = 0;
    useImageBCflag                      = false;
    padValue                            = 0;
    autoResizeFlag                      = true;
    displayPeriod                       = 1200;
    echoFramePathnameFlag               = true;
    batchMethod                         = "bySpecified";
    start_frame_index                   = {0.000000};
    skip_frame_index                    = {0.000000};
    writeFrameToTimestamp               = true;
    flipOnTimescaleError                = true;
    resetToStartOnLoop                  = false;
};

ImageRight = {
groupType = "Movie";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 0;
    mirrorBCflag                        = true;
    initializeFromCheckpointFlag        = false;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    inputPath                           = "/nh/compneuro/Data/KITTI/list/image_03.txt";
    offsetAnchor                        = "tl";
    offsetX                             = 0;
    offsetY                             = 0;
    writeImages                         = 0;
    inverseFlag                         = false;
    normalizeLuminanceFlag              = true;
    normalizeStdDev                     = true;
    jitterFlag                          = 0;
    useImageBCflag                      = false;
    padValue                            = 0;
    autoResizeFlag                      = true;
    displayPeriod                       = 1200;
    echoFramePathnameFlag               = true;
    batchMethod                         = "bySpecified";
    start_frame_index                   = {0.000000};
    skip_frame_index                    = {0.000000};
    writeFrameToTimestamp               = true;
    flipOnTimescaleError                = true;
    resetToStartOnLoop                  = false;
};

ImageLeftDecon = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 11;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageRightDecon = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 11;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageLeftDeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 1;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
    useMask                             = false;
};

ImageRightDeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 1;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
    useMask                             = false;
};

S1DeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 2;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

S1LeftDeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 2;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

S1RightDeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 2;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

S1 = {
groupType = "HyPerLCALayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 3;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "InitVFromFile";
    Vfilename                           = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S1_V.pvp";
    --InitVType                           = "UniformRandomV";
    --minV                                = -1;
    --maxV                                = 0.05;
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
    VThresh                             = 0.00625;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = 400;
    selfInteract                        = true;
};

S1Left = {
groupType = "HyPerLCALayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 3;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "InitVFromFile";
    Vfilename                           = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S1Left_V.pvp";
    --InitVType                           = "UniformRandomV";
    --minV                                = -1;
    --maxV                                = 0.05;
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
    VThresh                             = 0.00625;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = 400;
    selfInteract                        = true;
};

S1Right = {
groupType = "HyPerLCALayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 3;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "InitVFromFile";
    Vfilename                           = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S1Right_V.pvp";
    --InitVType                           = "UniformRandomV";
    --minV                                = -1;
    --maxV                                = 0.05;
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
    VThresh                             = 0.00625;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = 400;
    selfInteract                        = true;
};

ImageLeftDeconS1 = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 4;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageLeftDeconS1Left = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 4;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageRightDeconS1 = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 4;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageRightDeconS1Right = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 4;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S2DeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 4;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

S2LeftDeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 4;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

S2RightDeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 4;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

S2 = {
groupType = "HyPerLCALayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 5;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "InitVFromFile";
    Vfilename                           = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S2_V.pvp";
    --InitVType                           = "UniformRandomV";
    --minV                                = -1;
    --maxV                                = 0.05;
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
VThresh                             = 0.00625; --0.009375;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = 800;
    selfInteract                        = true;
};

S2Left = {
groupType = "HyPerLCALayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 5;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "InitVFromFile";
    Vfilename                           = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S2Left_V.pvp";
    --InitVType                           = "UniformRandomV";
    --minV                                = -1;
    --maxV                                = 0.05;
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
VThresh                             = 0.00625; --0.009375;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = 800;
    selfInteract                        = true;
};

S2Right = {
groupType = "HyPerLCALayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 5;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "InitVFromFile";
    Vfilename                           = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S2Right_V.pvp";
    --InitVType                           = "UniformRandomV";
    --minV                                = -1;
    --maxV                                = 0.05;
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
VThresh                             = 0.00625; --0.009375;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = 800;
    selfInteract                        = true;
};

S1DeconS2 = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 6;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1LeftDeconS2 = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 6;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1RightDeconS2 = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 6;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1LeftDeconS2Left = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 6;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1RightDeconS2Right = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 6;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageLeftDeconS2 = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 7;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageLeftDeconS2Left = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 7;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageRightDeconS2 = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 7;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageRightDeconS2Right = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 7;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S3DeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 512;
    phase                               = 5;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

S3LeftDeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 512;
    phase                               = 5;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

S3RightDeconError = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 512;
    phase                               = 5;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

S3 = {
groupType = "HyPerLCALayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 512;
    phase                               = 7;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "InitVFromFile";
    Vfilename                           = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S3_V.pvp";
    --InitVType                           = "UniformRandomV";
    --minV                                = -1;
    --maxV                                = 0.05;
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
VThresh                             = 0.00625; --0.0125; --0.025;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = 1200;
    selfInteract                        = true;
};

S3Left = {
groupType = "HyPerLCALayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 512;
    phase                               = 7;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "InitVFromFile";
    Vfilename                           = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S3Left_V.pvp";
    --InitVType                           = "UniformRandomV";
    --minV                                = -1;
    --maxV                                = 0.05;
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
VThresh                             = 0.00625; --0.0125; --0.025;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = 1200;
    selfInteract                        = true;
};

S3Right = {
groupType = "HyPerLCALayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 512;
    phase                               = 7;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "InitVFromFile";
    Vfilename                           = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S3Right_V.pvp";
    --InitVType                           = "UniformRandomV";
    --minV                                = -1;
    --maxV                                = 0.05;
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
VThresh                             = 0.00625; --0.0125; --0.025;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = 1200;
    selfInteract                        = true;
};

S2DeconS3 = {
groupType = "ANNLayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 8;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S2LeftDeconS3 = {
groupType = "ANNLayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 8;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S2RightDeconS3 = {
groupType = "ANNLayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 8;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S2LeftDeconS3Left = {
groupType = "ANNLayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 8;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S2RightDeconS3Right = {
groupType = "ANNLayer";
    nxScale                             = 0.25;
    nyScale                             = 0.25;
    nf                                  = 256;
    phase                               = 8;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1DeconS3 = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 9;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1LeftDeconS3 = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 9;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1RightDeconS3 = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 9;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1LeftDeconS3Left = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 9;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1RightDeconS3Right = {
groupType = "ANNLayer";
    nxScale                             = 0.5;
    nyScale                             = 0.5;
    nf                                  = 128;
    phase                               = 9;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageLeftDeconS3 = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 10;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageLeftDeconS3Left = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 10;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageRightDeconS3 = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 10;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageRightDeconS3Right = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 10;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

GroundTruthPixels = {
groupType = "Movie";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 1;
    phase                               = 0;
    mirrorBCflag                        = true;
    initializeFromCheckpointFlag        = false;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    inputPath                           = "/nh/compneuro/Data/KITTI/list/depth.txt";
    offsetAnchor                        = "tl";
    offsetX                             = 0;
    offsetY                             = 0;
    writeImages                         = 0;
    inverseFlag                         = false;
    normalizeLuminanceFlag              = false;
    jitterFlag                          = 0;
    useImageBCflag                      = false;
    padValue                            = 0;
    autoResizeFlag                      = true;
    displayPeriod                       = 1200;
    echoFramePathnameFlag               = true;
    batchMethod                         = "bySpecified";
    start_frame_index                   = {0.000000};
    skip_frame_index                    = {0.000000};
    writeFrameToTimestamp               = true;
    flipOnTimescaleError                = true;
    resetToStartOnLoop                  = false;
};

GroundTruthDownsample = {
groupType = "ANNLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 1;
    phase                               = 1;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 0;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

GroundTruth = {
groupType = "BinningLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 32;
    phase                               = 2;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 0;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    originalLayerName                   = "GroundTruthDownsample";
    binMax                              = 1;
    binMin                              = 0;
    delay                               = 0;
    binSigma                            = 1;
    zeroNeg                             = false;
    zeroDCR                             = true;
    normalDist                          = false;
};

GroundTruthReconS3Error = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 32;
    phase                               = 10;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

GroundTruthReconS3 = {
groupType = "ANNLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 32;
    phase                               = 9;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

BiasS3 = {
groupType = "ConstantLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 1;
    phase                               = 0;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ConstantV";
    valueV                              = 1;
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S3MaxPooled = {
groupType = "ANNLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 512;
    phase                               = 8;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

GroundTruthReconS2Error = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 32;
    phase                               = 10;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

GroundTruthReconS2 = {
groupType = "ANNLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 32;
    phase                               = 9;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

BiasS2 = {
groupType = "ConstantLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 1;
    phase                               = 0;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ConstantV";
    valueV                              = 1;
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S2MaxPooled = {
groupType = "ANNLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 256;
    phase                               = 8;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

GroundTruthReconS1Error = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 32;
    phase                               = 10;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

GroundTruthReconS1 = {
groupType = "ANNLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 32;
    phase                               = 9;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

BiasS1 = {
groupType = "ConstantLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 1;
    phase                               = 0;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ConstantV";
    valueV                              = 1;
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

S1MaxPooled = {
groupType = "ANNLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 128;
    phase                               = 8;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

GroundTruthReconS1S2S3Error = {
groupType = "ANNErrorLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 32;
    phase                               = 10;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
};

GroundTruthReconS1S2S3 = {
groupType = "ANNLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 32;
    phase                               = 9;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerLayerName                    = "GroundTruthPixels";
    triggerOffset                       = 1;
    triggerBehavior                     = "updateOnlyOnTrigger";
    writeStep                           = 1200;
    initialWriteTime                    = 1200;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

BiasS1S2S3 = {
groupType = "ConstantLayer";
    nxScale                             = 0.125;
    nyScale                             = 0.125;
    nf                                  = 1;
    phase                               = 0;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ConstantV";
    valueV                              = 1;
    triggerLayerName                    = nil;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -3.40282e+38;
    AMin                                = -3.40282e+38;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
};

ImageLeftToImageLeftDeconError = {
groupType = "HyPerConn";
    preLayerName                        = "ImageLeft";
    postLayerName                       = "ImageLeftDeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = false;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "OneToOneWeights";
    initWeightsFile                     = nil;
    weightInit                          = 0.032075;
    initializeFromCheckpointFlag        = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 3;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
};

ImageRightToImageRightDeconError = {
groupType = "HyPerConn";
    preLayerName                        = "ImageRight";
    postLayerName                       = "ImageRightDeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = false;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "OneToOneWeights";
    initWeightsFile                     = nil;
    weightInit                          = 0.032075;
    initializeFromCheckpointFlag        = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 3;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
};

ImageLeftDeconToImageLeftDeconError = {
groupType = "IdentConn";
    preLayerName                        = "ImageLeftDecon";
    postLayerName                       = "ImageLeftDeconError";
    channelCode                         = 1;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageRightDeconToImageRightDeconError = {
groupType = "IdentConn";
    preLayerName                        = "ImageRightDecon";
    postLayerName                       = "ImageRightDeconError";
    channelCode                         = 1;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S1ToImageLeftDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S1";
    postLayerName                       = "ImageLeftDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S1ToImageLeftDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 18;
    nyp                                 = 18;
    nfp                                 = 3;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeL2";
    strength                            = 1;
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minL2NormTolerated                  = 0;
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S1LeftToImageLeftDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S1Left";
    postLayerName                       = "ImageLeftDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S1LeftToImageLeftDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 18;
    nyp                                 = 18;
    nfp                                 = 3;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeL2";
    strength                            = 1;
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minL2NormTolerated                  = 0;
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S1ToImageRightDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S1";
    postLayerName                       = "ImageRightDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S1ToImageRightDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 18;
    nyp                                 = 18;
    nfp                                 = 3;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeGroup";
    normalizeGroupName                  = "S1ToImageLeftDeconError";
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S1RightToImageRightDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S1Right";
    postLayerName                       = "ImageRightDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S1RightToImageRightDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 18;
    nyp                                 = 18;
    nfp                                 = 3;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeL2";
    strength                            = 1;
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minL2NormTolerated                  = 0;
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

ImageLeftDeconErrorToS1DeconError = {
groupType = "TransposeConn";
    preLayerName                        = "ImageLeftDeconError";
    postLayerName                       = "S1DeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1ToImageLeftDeconError";
};

ImageLeftDeconErrorToS1LeftDeconError = {
groupType = "TransposeConn";
    preLayerName                        = "ImageLeftDeconError";
    postLayerName                       = "S1LeftDeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1LeftToImageLeftDeconError";
};

ImageRightDeconErrorToS1DeconError = {
groupType = "TransposeConn";
    preLayerName                        = "ImageRightDeconError";
    postLayerName                       = "S1DeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1ToImageRightDeconError";
};

ImageRightDeconErrorToS1RightDeconError = {
groupType = "TransposeConn";
    preLayerName                        = "ImageRightDeconError";
    postLayerName                       = "S1RightDeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1RightToImageRightDeconError";
};

S1DeconErrorToS1 = {
groupType = "IdentConn";
    preLayerName                        = "S1DeconError";
    postLayerName                       = "S1";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S1LeftDeconErrorToS1Left = {
groupType = "IdentConn";
    preLayerName                        = "S1LeftDeconError";
    postLayerName                       = "S1Left";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S1RightDeconErrorToS1Right = {
groupType = "IdentConn";
    preLayerName                        = "S1RightDeconError";
    postLayerName                       = "S1Right";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S1ToImageLeftDeconS1 = {
groupType = "CloneConn";
    preLayerName                        = "S1";
    postLayerName                       = "ImageLeftDeconS1";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S1ToImageLeftDeconError";
};

S1LeftToImageLeftDeconS1Left = {
groupType = "CloneConn";
    preLayerName                        = "S1Left";
    postLayerName                       = "ImageLeftDeconS1Left";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S1LeftToImageLeftDeconError";
};

S1ToImageRightDeconS1 = {
groupType = "CloneConn";
    preLayerName                        = "S1";
    postLayerName                       = "ImageRightDeconS1";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S1ToImageRightDeconError";
};

S1RightToImageRightDeconS1Right = {
groupType = "CloneConn";
    preLayerName                        = "S1Right";
    postLayerName                       = "ImageRightDeconS1Right";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S1RightToImageRightDeconError";
};

ImageLeftDeconS1ToImageLeftDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageLeftDeconS1";
    postLayerName                       = "ImageLeftDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageLeftDeconS1LeftToImageLeftDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageLeftDeconS1Left";
    postLayerName                       = "ImageLeftDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageRightDeconS1ToImageRightDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageRightDeconS1";
    postLayerName                       = "ImageRightDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageRightDeconS1TRightoImageRightDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageRightDeconS1Right";
    postLayerName                       = "ImageRightDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S2ToS1DeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S2";
    postLayerName                       = "S1DeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S2ToS1DeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 128;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeL2";
    strength                            = 1;
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minL2NormTolerated                  = 0;
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S2ToS1LeftDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S2";
    postLayerName                       = "S1LeftDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S2ToS1LeftDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 128;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeGroup";
    normalizeGroupName                  = "S2ToS1DeconError";
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S2ToS1RightDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S2";
    postLayerName                       = "S1RightDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S2ToS1RightDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 128;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeGroup";
    normalizeGroupName                  = "S2ToS1DeconError";
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S2LeftToS1LeftDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S2Left";
    postLayerName                       = "S1LeftDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S2LeftToS1LeftDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 128;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeL2";
    strength                            = 1;
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minL2NormTolerated                  = 0;
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S2RightToS1RightDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S2Right";
    postLayerName                       = "S1RightDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S2RightToS1RightDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 128;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeL2";
    strength                            = 1;
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minL2NormTolerated                  = 0;
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S1DeconErrorToS2DeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S1DeconError";
    postLayerName                       = "S2DeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2ToS1DeconError";
};

S1LeftDeconErrorToS2DeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S1LeftDeconError";
    postLayerName                       = "S2DeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2ToS1LeftDeconError";
};

S1RightDeconErrorToS2DeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S1RightDeconError";
    postLayerName                       = "S2DeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2ToS1RightDeconError";
};

S1LeftDeconErrorToS2LeftDeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S1LeftDeconError";
    postLayerName                       = "S2LeftDeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2LeftToS1LeftDeconError";
};

S1RightDeconErrorToS2RightDeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S1RightDeconError";
    postLayerName                       = "S2RightDeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2RightToS1RightDeconError";
};

S2DeconErrorToS2 = {
groupType = "IdentConn";
    preLayerName                        = "S2DeconError";
    postLayerName                       = "S2";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S2LeftDeconErrorToS2Left = {
groupType = "IdentConn";
    preLayerName                        = "S2LeftDeconError";
    postLayerName                       = "S2Left";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S2RightDeconErrorToS2Right = {
groupType = "IdentConn";
    preLayerName                        = "S2RightDeconError";
    postLayerName                       = "S2Right";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S2ToS1DeconS2 = {
groupType = "CloneConn";
    preLayerName                        = "S2";
    postLayerName                       = "S1DeconS2";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S2ToS1DeconError";
};

S2ToS1LeftDeconS2 = {
groupType = "CloneConn";
    preLayerName                        = "S2";
    postLayerName                       = "S1LeftDeconS2";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S2ToS1LeftDeconError";
};

S2ToS1RightDeconS2 = {
groupType = "CloneConn";
    preLayerName                        = "S2";
    postLayerName                       = "S1RightDeconS2";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S2ToS1RightDeconError";
};

S2LeftToS1LeftDeconS2Left = {
groupType = "CloneConn";
    preLayerName                        = "S2Left";
    postLayerName                       = "S1LeftDeconS2Left";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S2LeftToS1LeftDeconError";
};

S2RightToS1RightDeconS2Right = {
groupType = "CloneConn";
    preLayerName                        = "S2Right";
    postLayerName                       = "S1RightDeconS2Right";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S2RightToS1RightDeconError";
};

S1DeconS2ToImageLeftDeconS2 = {
groupType = "CloneConn";
    preLayerName                        = "S1DeconS2";
    postLayerName                       = "ImageLeftDeconS2";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1ToImageLeftDeconError";
};

S1LeftDeconS2ToImageLeftDeconS2 = {
groupType = "CloneConn";
    preLayerName                        = "S1LeftDeconS2";
    postLayerName                       = "ImageLeftDeconS2";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1LeftToImageLeftDeconError";
};

S1LeftDeconS2LeftToImageLeftDeconS2Left = {
groupType = "CloneConn";
    preLayerName                        = "S1LeftDeconS2Left";
    postLayerName                       = "ImageLeftDeconS2Left";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1LeftToImageLeftDeconError";
};

S1DeconS2ToImageRightDeconS2 = {
groupType = "CloneConn";
    preLayerName                        = "S1DeconS2";
    postLayerName                       = "ImageRightDeconS2";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1ToImageRightDeconError";
};

S1RightDeconS2ToImageRightDeconS2 = {
groupType = "CloneConn";
    preLayerName                        = "S1RightDeconS2";
    postLayerName                       = "ImageRightDeconS2";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1RightToImageRightDeconError";
};

S1RightDeconS2RightToImageRightDeconS2Right = {
groupType = "CloneConn";
    preLayerName                        = "S1RightDeconS2Right";
    postLayerName                       = "ImageRightDeconS2Right";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1RightToImageRightDeconError";
};

ImageLeftDeconS2ToImageLeftDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageLeftDeconS2";
    postLayerName                       = "ImageLeftDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageLeftDeconS2LeftToImageLeftDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageLeftDeconS2Left";
    postLayerName                       = "ImageLeftDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageRightDeconS2ToImageRightDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageRightDeconS2";
    postLayerName                       = "ImageRightDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageRightDeconS2RightToImageRightDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageRightDeconS2Right";
    postLayerName                       = "ImageRightDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S3ToS2DeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S3";
    postLayerName                       = "S2DeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S3ToS2DeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 256;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeL2";
    strength                            = 1;
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minL2NormTolerated                  = 0;
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S3ToS2LeftDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S3";
    postLayerName                       = "S2LeftDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S3ToS2LeftDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 256;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeGroup";
    normalizeGroupName                  = "S3ToS2DeconError";
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S3ToS2RightDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S3";
    postLayerName                       = "S2RightDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S3ToS2RightDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 256;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeGroup";
    normalizeGroupName                  = "S3ToS2DeconError";
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S3LeftToS2LeftDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S3Left";
    postLayerName                       = "S2LeftDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S3LeftToS2LeftDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 256;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeL2";
    strength                            = 1;
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minL2NormTolerated                  = 0;
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S3RightToS2RightDeconError = {
groupType = "MomentumConn";
    preLayerName                        = "S3Right";
    postLayerName                       = "S2RightDeconError";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "FileWeight";
    initWeightsFile                     = "/home/gkenyon/KITTI/KITTI_S1_128_S2_256_S3_512_DCA/KITTI_train6/Checkpoints/Checkpoint72000/S3RightToS2RightDeconError_W.pvp";
    useListOfArborFiles                 = false;
    combineWeightFiles                  = false;
    --weightInitType                      = "UniformRandomWeight";
    --initWeightsFile                     = nil;
    --wMinInit                            = -1;
    --wMaxInit                            = 1;
    --sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 10;
    nyp                                 = 10;
    nfp                                 = 256;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeL2";
    strength                            = 1;
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minL2NormTolerated                  = 0;
    dWMax                               = 10;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumMethod                      = "viscosity";
    momentumTau                         = 400;
    momentumDecay                       = 0;
    batchPeriod                         = 1;
};

S2DeconErrorToS3DeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S2DeconError";
    postLayerName                       = "S3DeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S3ToS2DeconError";
};

S2LeftDeconErrorToS3DeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S2LeftDeconError";
    postLayerName                       = "S3DeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S3ToS2LeftDeconError";
};

S2RightDeconErrorToS3DeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S2RightDeconError";
    postLayerName                       = "S3DeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S3ToS2RightDeconError";
};

S2LeftDeconErrorToS3LeftDeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S2LeftDeconError";
    postLayerName                       = "S3LeftDeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S3LeftToS2LeftDeconError";
};

S2RightDeconErrorToS3RightDeconError = {
groupType = "TransposeConn";
    preLayerName                        = "S2RightDeconError";
    postLayerName                       = "S3RightDeconError";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S3RightToS2RightDeconError";
};

S3DeconErrorToS3 = {
groupType = "IdentConn";
    preLayerName                        = "S3DeconError";
    postLayerName                       = "S3";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S3LeftDeconErrorToS3Left = {
groupType = "IdentConn";
    preLayerName                        = "S3LeftDeconError";
    postLayerName                       = "S3Left";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S3RightDeconErrorToS3Right = {
groupType = "IdentConn";
    preLayerName                        = "S3RightDeconError";
    postLayerName                       = "S3Right";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S3ToS2DeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S3";
    postLayerName                       = "S2DeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S3ToS2DeconError";
};

S3ToS2LeftDeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S3";
    postLayerName                       = "S2LeftDeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S3ToS2LeftDeconError";
};

S3ToS2RightDeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S3";
    postLayerName                       = "S2RightDeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S3ToS2RightDeconError";
};

S3LeftToS2LeftDeconS3Left = {
groupType = "CloneConn";
    preLayerName                        = "S3Left";
    postLayerName                       = "S2LeftDeconS3Left";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S3LeftToS2LeftDeconError";
};

S3RightToS2RightDeconS3Right = {
groupType = "CloneConn";
    preLayerName                        = "S3Right";
    postLayerName                       = "S2RightDeconS3Right";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S3RightToS2RightDeconError";
};

S2DeconS3ToS1DeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S2DeconS3";
    postLayerName                       = "S1DeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2ToS1DeconError";
};

S2LeftDeconS3ToS1LeftDeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S2LeftDeconS3";
    postLayerName                       = "S1LeftDeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2LeftToS1LeftDeconError";
};

S2RightDeconS3ToS1RightDeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S2RightDeconS3";
    postLayerName                       = "S1RightDeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2RightToS1RightDeconError";
};

S2LeftDeconS3LeftToS1LeftDeconS3Left = {
groupType = "CloneConn";
    preLayerName                        = "S2LeftDeconS3Left";
    postLayerName                       = "S1LeftDeconS3Left";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2LeftToS1LeftDeconError";
};

S2RightDeconS3RightToS1RightDeconS3Right = {
groupType = "CloneConn";
    preLayerName                        = "S2RightDeconS3Right";
    postLayerName                       = "S1RightDeconS3Right";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S2RightToS1RightDeconError";
};

S1DeconS3ToImageLeftDeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S1DeconS3";
    postLayerName                       = "ImageLeftDeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1ToImageLeftDeconError";
};

S1LeftDeconS3ToImageLeftDeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S1LeftDeconS3";
    postLayerName                       = "ImageLeftDeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1LeftToImageLeftDeconError";
};

S1LeftDeconS3LeftToImageLeftDeconS3Left = {
groupType = "CloneConn";
    preLayerName                        = "S1LeftDeconS3Left";
    postLayerName                       = "ImageLeftDeconS3Left";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1LeftToImageLeftDeconError";
};

S1DeconS3ToImageRightDeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S1DeconS3";
    postLayerName                       = "ImageRightDeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1ToImageRightDeconError";
};

S1RightDeconS3ToImageRightDeconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S1RightDeconS3";
    postLayerName                       = "ImageRightDeconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1RightToImageRightDeconError";
};

S1RightDeconS3RightToImageRightDeconS3Right = {
groupType = "CloneConn";
    preLayerName                        = "S1RightDeconS3Right";
    postLayerName                       = "ImageRightDeconS3Right";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = true;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    gpuGroupIdx                         = -1;
    originalConnName                    = "S1RightToImageRightDeconError";
};

ImageLeftDeconS3ToImageLeftDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageLeftDeconS3";
    postLayerName                       = "ImageLeftDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageLeftDeconS3LeftToImageLeftDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageLeftDeconS3Left";
    postLayerName                       = "ImageLeftDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageRightDeconS3ToImageRightDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageRightDeconS3";
    postLayerName                       = "ImageRightDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

ImageRightDeconS3RightToImageRightDecon = {
groupType = "IdentConn";
    preLayerName                        = "ImageRightDeconS3Right";
    postLayerName                       = "ImageRightDecon";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

GroundTruthPixelsToGroundTruthDownsample = {
groupType = "PoolingConn";
    preLayerName                        = "GroundTruthPixels";
    postLayerName                       = "GroundTruthDownsample";
    channelCode                         = 0;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    initializeFromCheckpointFlag        = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "maxpooling";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 1;
    shrinkPatches                       = false;
    needPostIndexLayer                  = false;
};

GroundTruthToGroundTruthReconS3Error = {
groupType = "IdentConn";
    preLayerName                        = "GroundTruth";
    postLayerName                       = "GroundTruthReconS3Error";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

GroundTruthToGroundTruthReconS2Error = {
groupType = "IdentConn";
    preLayerName                        = "GroundTruth";
    postLayerName                       = "GroundTruthReconS2Error";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

GroundTruthToGroundTruthReconS1Error = {
groupType = "IdentConn";
    preLayerName                        = "GroundTruth";
    postLayerName                       = "GroundTruthReconS1Error";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

GroundTruthToGroundTruthReconS1S2S3Error = {
groupType = "IdentConn";
    preLayerName                        = "GroundTruth";
    postLayerName                       = "GroundTruthReconS1S2S3Error";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

S3ToS3MaxPooled = {
groupType = "PoolingConn";
    preLayerName                        = "S3";
    postLayerName                       = "S3MaxPooled";
    channelCode                         = 0;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    initializeFromCheckpointFlag        = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "maxpooling";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 512;
    shrinkPatches                       = false;
    needPostIndexLayer                  = false;
};

S2ToS2MaxPooled = {
groupType = "PoolingConn";
    preLayerName                        = "S2";
    postLayerName                       = "S2MaxPooled";
    channelCode                         = 0;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    initializeFromCheckpointFlag        = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "maxpooling";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 256;
    shrinkPatches                       = false;
    needPostIndexLayer                  = false;
};

S1ToS1MaxPooled = {
groupType = "PoolingConn";
    preLayerName                        = "S1";
    postLayerName                       = "S1MaxPooled";
    channelCode                         = 0;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    initializeFromCheckpointFlag        = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "maxpooling";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 128;
    shrinkPatches                       = false;
    needPostIndexLayer                  = false;
};

S3MaxPooledToGroundTruthReconS3Error = {
groupType = "HyPerConn";
    preLayerName                        = "S3MaxPooled";
    postLayerName                       = "GroundTruthReconS3Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 1;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

S2MaxPooledToGroundTruthReconS2Error = {
groupType = "HyPerConn";
    preLayerName                        = "S2MaxPooled";
    postLayerName                       = "GroundTruthReconS2Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 1;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

S1MaxPooledToGroundTruthReconS1Error = {
groupType = "HyPerConn";
    preLayerName                        = "S1MaxPooled";
    postLayerName                       = "GroundTruthReconS1Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 1;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

S1MaxPooledToGroundTruthReconS1S2S3Error = {
groupType = "HyPerConn";
    preLayerName                        = "S1MaxPooled";
    postLayerName                       = "GroundTruthReconS1S2S3Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 1;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

S2MaxPooledToGroundTruthReconS1S2S3Error = {
groupType = "HyPerConn";
    preLayerName                        = "S2MaxPooled";
    postLayerName                       = "GroundTruthReconS1S2S3Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 1;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

S3MaxPooledToGroundTruthReconS1S2S3Error = {
groupType = "HyPerConn";
    preLayerName                        = "S3MaxPooled";
    postLayerName                       = "GroundTruthReconS1S2S3Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 1;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

BiasS3ToGroundTruthReconS3Error = {
groupType = "HyPerConn";
    preLayerName                        = "BiasS3";
    postLayerName                       = "GroundTruthReconS3Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 0.01;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

BiasS2ToGroundTruthReconS2Error = {
groupType = "HyPerConn";
    preLayerName                        = "BiasS2";
    postLayerName                       = "GroundTruthReconS2Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 0.01;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

BiasS1ToGroundTruthReconS1Error = {
groupType = "HyPerConn";
    preLayerName                        = "BiasS1";
    postLayerName                       = "GroundTruthReconS1Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 0.01;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

BiasS1S2S3ToGroundTruthReconS1S2S3Error = {
groupType = "HyPerConn";
    preLayerName                        = "BiasS1S2S3";
    postLayerName                       = "GroundTruthReconS1S2S3Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -0;
    wMaxInit                            = 0;
    sparseFraction                      = 0;
    initializeFromCheckpointFlag        = false;
    triggerLayerName                    = "ImageLeft";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = 1;
    nyp                                 = 1;
    nfp                                 = 32;
    shrinkPatches                       = false;
    normalizeMethod                     = "none";
    dWMax                               = 0.01;
    keepKernelsSynchronized             = true;
    useMask                             = false;
};

S3MaxPooledToGroundTruthReconS3 = {
groupType = "CloneConn";
    preLayerName                        = "S3MaxPooled";
    postLayerName                       = "GroundTruthReconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S3MaxPooledToGroundTruthReconS3Error";
};

S2MaxPooledToGroundTruthReconS2 = {
groupType = "CloneConn";
    preLayerName                        = "S2MaxPooled";
    postLayerName                       = "GroundTruthReconS2";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S2MaxPooledToGroundTruthReconS2Error";
};

S1MaxPooledToGroundTruthReconS1 = {
groupType = "CloneConn";
    preLayerName                        = "S1MaxPooled";
    postLayerName                       = "GroundTruthReconS1";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S1MaxPooledToGroundTruthReconS1Error";
};

S1MaxPooledToGroundTruthReconS1S2S3 = {
groupType = "CloneConn";
    preLayerName                        = "S1MaxPooled";
    postLayerName                       = "GroundTruthReconS1S2S3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S1MaxPooledToGroundTruthReconS1S2S3Error";
};

S2MaxPooledToGroundTruthReconS1S2S3 = {
groupType = "CloneConn";
    preLayerName                        = "S2MaxPooled";
    postLayerName                       = "GroundTruthReconS1S2S3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S2MaxPooledToGroundTruthReconS1S2S3Error";
};

S3MaxPooledToGroundTruthReconS1S2S3 = {
groupType = "CloneConn";
    preLayerName                        = "S3MaxPooled";
    postLayerName                       = "GroundTruthReconS1S2S3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S3MaxPooledToGroundTruthReconS1S2S3Error";
};

BiasS3ToGroundTruthReconS3 = {
groupType = "CloneConn";
    preLayerName                        = "BiasS3";
    postLayerName                       = "GroundTruthReconS3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "BiasS3ToGroundTruthReconS3Error";
};

BiasS2ToGroundTruthReconS2 = {
groupType = "CloneConn";
    preLayerName                        = "BiasS2";
    postLayerName                       = "GroundTruthReconS2";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "BiasS2ToGroundTruthReconS2Error";
};

BiasS1ToGroundTruthReconS1 = {
groupType = "CloneConn";
    preLayerName                        = "BiasS1";
    postLayerName                       = "GroundTruthReconS1";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "BiasS1ToGroundTruthReconS1Error";
};

BiasS1S2S3ToGroundTruthReconS1S2S3 = {
groupType = "CloneConn";
    preLayerName                        = "BiasS1S2S3";
    postLayerName                       = "GroundTruthReconS1S2S3";
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "BiasS1S2S3ToGroundTruthReconS1S2S3Error";
};

GroundTruthReconS3ToGroundTruthReconS3Error = {
groupType = "IdentConn";
    preLayerName                        = "GroundTruthReconS3";
    postLayerName                       = "GroundTruthReconS3Error";
    channelCode                         = 1;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

GroundTruthReconS2ToGroundTruthReconS2Error = {
groupType = "IdentConn";
    preLayerName                        = "GroundTruthReconS2";
    postLayerName                       = "GroundTruthReconS2Error";
    channelCode                         = 1;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

GroundTruthReconS1ToGroundTruthReconS1Error = {
groupType = "IdentConn";
    preLayerName                        = "GroundTruthReconS1";
    postLayerName                       = "GroundTruthReconS1Error";
    channelCode                         = 1;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

GroundTruthReconS1S2S3ToGroundTruthReconS1S2S3Error = {
groupType = "IdentConn";
    preLayerName                        = "GroundTruthReconS1S2S3";
    postLayerName                       = "GroundTruthReconS1S2S3Error";
    channelCode                         = 1;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
};

} --End of pvParameters


   pv.addGroup(pvParameters, "EnergyProbe", 
	       {
		  groupType                           = "ColumnEnergyProbe";
		  probeOutputFile                     = "S1EnergyProbe.txt";
	       }
   )
   
   pv.addGroup(pvParameters, "ImageLeftDeconErrorL2NormEnergyProbe",
	       {
		  groupType                           = "L2NormProbe";
		  targetLayer                         = "ImageLeftDeconError";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "ImageLeftDeconErrorL2NormEnergyProbe.txt";
		  triggerLayerName                    = NULL; --"Image";
		  --triggerOffset                       = 1;
		  energyProbe                         = "EnergyProbe";
		  coefficient                         = 0.5;
		  maskLayerName                       = NULL;
		  exponent                            = 2;
	       }
   )

   pv.addGroup(pvParameters, "ImageRightDeconErrorL2NormEnergyProbe",
	       {
		  groupType                           = "L2NormProbe";
		  targetLayer                         = "ImageRightDeconError";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "ImageRightDeconErrorL2NormEnergyProbe.txt";
		  triggerLayerName                    = NULL; --"Image";
		  --triggerOffset                       = 1;
		  energyProbe                         = "EnergyProbe";
		  coefficient                         = 0.5;
		  maskLayerName                       = NULL;
		  exponent                            = 2;
	       }
   )

   pv.addGroup(pvParameters, "S1FirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S1";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S1FirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   pv.addGroup(pvParameters, "S1LeftFirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S1Left";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S1LeftFirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   pv.addGroup(pvParameters, "S1RightFirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S1Right";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S1RightFirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   pv.addGroup(pvParameters, "S2FirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S2";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S2FirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   pv.addGroup(pvParameters, "S2LeftFirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S2Left";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S2LeftFirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   pv.addGroup(pvParameters, "S2RightFirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S2Right";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S2RightFirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   pv.addGroup(pvParameters, "S3FirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S3";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S3FirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   pv.addGroup(pvParameters, "S3LeftFirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S3Left";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S3LeftFirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )

   pv.addGroup(pvParameters, "S3RightFirmThresholdCostFnLCAProbe",
	       {
		  groupType                           = "FirmThresholdCostFnLCAProbe";
		  targetLayer                         = "S3Right";
		  message                             = NULL;
		  textOutputFlag                      = true;
		  probeOutputFile                     = "S3RightFirmThresholdCostFnLCAProbe.txt";
		  triggerLayerName                    = NULL;
		  energyProbe                         = "EnergyProbe";
		  maskLayerName                       = NULL;
	       }
   )



-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParameters)
