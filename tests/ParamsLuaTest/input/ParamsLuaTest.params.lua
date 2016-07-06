package.path = package.path .. ";" .. "/home/pschultz/Workspace/git-repo/OpenPV/src/../parameterWrapper/?.lua"
local pv = require "PVModule"
local pi = 3.1415926535897931
local xsize = 32
local ysize = 32
local infeatures = 1
local outfeatures = 8

-- Base table variable to store
local pvParameters = {
column = {
groupType = "HyPerCol";
    startTime                           = 0;
    dt                                  = 1;
    dtAdaptFlag                         = false;
    dtAdaptController                   = nil;
    stopTime                            = 10;
    progressInterval                    = 10;
    writeProgressToErr                  = false;
    verifyWrites                        = false;
    outputPath                          = "output-lua/";
    printParamsFilename                 = "pv.params";
    randomSeed                          = 1234567890;
    nx                                  = xsize;
    ny                                  = ysize;
    nbatch                              = 1;
    filenamesContainLayerNames          = 0;
    filenamesContainConnectionNames     = 0;
    initializeFromCheckpointDir         = "";
    checkpointWrite                     = false;
    suppressLastOutput                  = false;
    errorOnNotANumber                   = true;
};

Input = {
groupType = "ImagePvp";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = infeatures;
    phase                               = 0;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    writeStep                           = -1;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    inputPath                           = "input/sampleimage.pvp";
    offsetAnchor                        = "tl";
    offsetX                             = 0;
    offsetY                             = 0;
    writeImages                         = 0;
    autoResizeFlag                      = false;
    inverseFlag                         = false;
    normalizeLuminanceFlag              = false;
    jitterFlag                          = 0;
    useImageBCflag                      = false;
    padValue                            = 0;
    pvpFrameIdx                         = 0;
};

Output = {
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = outfeatures;
    phase                               = 1;
    mirrorBCflag                        = true;
    InitVType                           = "ZeroV";
    triggerLayerName                    = nil;
    writeStep                           = 1;
    initialWriteTime                    = 0;
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

InputToOutput = {
groupType = "HyPerConn";
    preLayerName                        = "Input";
    postLayerName                       = "Output";
    channelCode                         = 0;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = false;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "Gauss2DWeight";
    initWeightsFile                     = nil;
    aspect                              = 3;
    sigma                               = 1;
    rMax                                = infinity;
    rMin                                = 0;
    strength                            = 4;
    numOrientationsPost                 = outfeatures;
    deltaThetaMax                       = 2*pi;
    thetaMax                            = 1;
    numFlanks                           = 1;
    flankShift                          = 0;
    rotate                              = 0;
    bowtieFlag                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    nxp                                 = 7;
    nyp                                 = 7;
    nfp                                 = outfeatures;
    shrinkPatches                       = false;
    normalizeMethod                     = "normalizeSum";
    normalizeArborsIndividually         = false;
    normalizeOnInitialize               = true;
    normalizeOnWeightUpdate             = true;
    rMinX                               = 0;
    rMinY                               = 0;
    nonnegativeConstraintFlag           = false;
    normalize_cutoff                    = 0;
    normalizeFromPostPerspective        = false;
    minSumTolerated                     = 0;
    weightSparsity                      = 0;
};

} --End of pvParameters

-- Print out PetaVision approved parameter file to the console
paramsFileString = pv.createParamsFileString(pvParameters)
if (not suppressWrite) then
    io.write(paramsFileString)
end
