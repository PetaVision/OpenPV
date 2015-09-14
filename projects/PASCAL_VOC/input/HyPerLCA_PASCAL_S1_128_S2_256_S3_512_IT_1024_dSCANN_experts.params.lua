-- PetaVision params file for 3 layer dSCANN topped with an IT dictionary of experts: 

--///////////////////////////
--// created by garkenyon, 08/20/15
--//
--//
--// implements a 3 layer multi-strided, multi-scale, multi-temporal deep sparse convolutional neural network (dSCANN) with symmetrical bottom-up and top-down connectivity
--//   designed to minimize an energy function that penalizes reconstruction error of the form:
--//    E = E_DCNN + E_DBN + E_Task + S(S1) + S(S2) + S(S3)
--//      = ||I - W1*S1 - W1*W2*S2 - W1*W2*W3*S3||^2 + ||S1 - V2*S2||^2 + ||S2 - V3*S3||^2 + ||GT - T*S3||^2
--//   the eq. of motion for d(S2)/dt is of the form:
--//      W2^T * W1^T * [I - W1*S1 - W1*W2*S2 - W1*W2*W3*S3] + V2^T * [S1 - V2*S2] - [S2 - V3*S3]
--//   which in terms of class names defined below becomes:
--//      S1DeconErrorToS2 * ImageDeconErrorToS1 * ImageDeconError + S1ReconS2ErrorToS2 * S1ReconS2Error
--//   where
--//      ImageDeconError = Image - ImageDecon = Image - S1ToImageDeconError * [S1 + S2ToS1DeconError * [S2 + S3ToS2DeconError * S3]]  
--//   the top-level S3 reconstructs PASCAL ground truth
--//   stride_S3 = 2*stride_S2 = 4*stride_S1
--//   nxp_S3 = 2*nxp_S2 = 4*nxp_S1 {18X18 -> 36X36 -> 72X72}
--//   S(A) = integral{T(A) - u}dA
--//

-- Load util module in PV trunk: NOTE this may need to change
--package.path = package.path .. ";" .. os.getenv("HOME") .. "/workspace/PetaVision/parameterWrapper/PVModule.lua"
package.path = package.path .. ";" .. "/nh/compneuro/Data/openpv" .. "/pv-core/parameterWrapper/PVModule.lua"
local pv = require "PVModule"

-- Global variable, for debug parsing
-- Needed by printConsole
debugParsing              = true

-- User defined variables

--Image parameters
local numImages           = 7958;
local imageListPath = "/nh/compneuro/Data/PASCAL_VOC/VOC2007/VOC2007_landscape_192X256_list.txt"
local GroundTruthPath = "/nh/compneuro/Data/PASCAL_VOC/VOC2007/VOC2007_landscape_192X256.pvp"
local displayPeriod       = 1200
local startFrame          = 0
local numColors           = 3

-- HyPerCol params
local nxSize              = 256
local nySize              = 192
local experimentName      = "PASCAL_S1_128_S2_256_S3_512_IT_1024_dSCANN_experts"
local runName             = "VOC2007_landscape"
local runVersion          = 1
local outputPathRoot      = "/nh/compneuro/Data/PASCAL_VOC/" .. experimentName .. "/" .. runName
local outputPath          = outputPathRoot .. runVersion
local startTime           = 0
local stopTime            = numImages*displayPeriod
local checkpointID        = stopTime
local initializeFromCheckpointDir = NULL;
local initializeFromCheckpointFlag = false;
if (runVersion > 1) then
   initializeFromCheckpointFlag = true
   initializeFromCheckpointDir  = outputPathRoot .. runVersion-1 .. "/Checkpoints/Checkpoint" .. checkpointID
end
local inf                 = 3.40282e+38

--i/o parameters
local writePeriod         = 1 * displayPeriod;

--HyPerLCA parameters
local numScales             = 4
local strideMin             = 2
local strideMultiplier      = 2
local numFeaturesMin        = 128
local numFeaturesMultiplier = 2
local patchSizeMin          = 18
local patchSizeMultiplier   = 2
local initLCAWeights        = NUL
local tauMin                = 400
local tauMultiplier         = 2
local VThresh               = 0.003125
local VWidth                = 10.0
local learningRate          = 0
local dWMax                 = 10.0
local learningMomentumTau   = 400

--Ground Truth parameters
local numClasses            = 20 --20

local defaultANNLayer = {
    groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = numColors;
    phase                               = 1;
    triggerLayerName                    = NULL;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
    InitVType                           = "ZeroV";
    writeStep                           = writePeriod;
    initialWriteTime                    = writePeriod;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -inf;
    AMin                                = -inf;
    AMax                                = inf;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
}

local defaultTriggeredANNLayer = pv.deepCopy(defaultANNLayer);
defaultTriggeredANNLayer.triggerLayerName = "Image";
defaultTriggeredANNLayer.triggerBehavior  = "updateOnlyOnTrigger";
defaultTriggeredANNLayer.triggerOffset = 0;

local defaultANNErrorLayer = pv.deepCopy(defaultANNLayer);
defaultANNErrorLayer.errScale              = 1;
defaultANNErrorLayer.useMask               = false;
defaultANNErrorLayer.AMin                  = nil;
defaultANNErrorLayer.AMax                  = nil;
defaultANNErrorLayer.AShift                = nil;
defaultANNErrorLayer.VWidth                = nil;

local resetLCALayer                        = true
local defaultHyPerLCALayer = pv.deepCopy(defaultANNLayer);
defaultHyPerLCALayer.AMin                  = 0;
defaultHyPerLCALayer.VThresh               = VThresh;
defaultHyPerLCALayer.VWidth                = VWidth;
if (resetLCALayer) then
   defaultHyPerLCALayer.triggerLayerName      = "Image";
   defaultHyPerLCALayer.triggerBehavior       = "resetStateOnTrigger";
   defaultHyPerLCALayer.triggerResetLayerName = "Constant"; -- .. "S1"
   defaultHyPerLCALayer.triggerOffset         = 0.0;
end
defaultHyPerLCALayer.updateGpu             = false;
defaultHyPerLCALayer.timeConstantTau       = tauMin;
defaultHyPerLCALayer.selfInteract          = true;
defaultHyPerLCALayer.sparseLayer           = true;
defaultHyPerLCALayer.writeSparseValues     = true;
defaultHyPerLCALayer.writeStep             = displayPeriod;
defaultHyPerLCALayer.initialWriteTime      = displayPeriod;
if (initializeFromCheckpointFlag) then
   defaultHyPerLCALayer.InitVType          = "InitVFromFile";
   defaultHyPerLCALayer.Vfilename          =  inputPath .. "/Checkpoints/Checkpoint" .. checkpointID; -- .. "/S1_V.pvp"
else
   if (resetLCALayer) then
      defaultHyPerLCALayer.InitVType          = "ConstantV";
      defaultHyPerLCALayer.valueV             =  VThresh;
   else
      defaultHyPerLCALayer.InitVType          = "UniformRandomV";
      defaultHyPerLCALayer.minV               =  -2*VThresh;
      defaultHyPerLCALayer.maxV               =  2*VThresh;
   end
end


-- Base table variable to store
local pvParameters = {
column = {
    groupType = "HyPerCol"; --String values
    startTime                           = 0;
    dt                                  = 1;
    dtAdaptFlag                         = true;
    dtScaleMax                          = 1;
    dtScaleMin                          = 0.02;
    dtChangeMax                         = 0.01;
    dtChangeMin                         = -0.02;
    dtMinToleratedTimeScale             = 0.0001;
    stopTime                            = stopTime; 
    progressInterval                    = 1000;
    writeProgressToErr                  = true;
    verifyWrites                        = false;
    outputPath                          = outputPath;
    printParamsFilename                 = experimentName .. "_" .. runName .. ".params";
    randomSeed                          = 1234567890;
    nx                                  = nxSize;
    ny                                  = nySize;
    filenamesContainLayerNames          = true;
    filenamesContainConnectionNames     = true;
    initializeFromCheckpointDir         = outputPath .. "/Checkpoints/Checkpoint" .. checkpointID;
    defaultInitializeFromCheckpointFlag = false;
    checkpointWrite                     = true;
    checkpointWriteDir                  = outputPath .. "/Checkpoints";
    checkpointWriteTriggerMode          = "step";
    checkpointWriteStepInterval         = writePeriod;
    deleteOlderCheckpoints              = false;
    suppressNonplasticCheckpoints       = false;
    writeTimescales                     = true;
    errorOnNotANumber                   = false;
};
} --End of pvParameters

local phase = 0
pv.addGroup(pvParameters, "Image", 
{
    groupType = "Movie";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = numColors;
    phase                               = phase;
    mirrorBCflag                        = true;
    initializeFromCheckpointFlag        = false;
    writeStep                           = writePeriod;
    initialWriteTime                    = writePeriod;
    sparseLayer                         = false;
    writeSparseValues                   = false;
    updateGpu                           = false;
    dataType                            = nil;
    offsetAnchor                        = "tl";
    offsetX                             = 0;
    offsetY                             = 0;
    writeImages                         = 0;
    useImageBCflag                      = false;
    autoResizeFlag                      = false;
    inverseFlag                         = false;
    normalizeLuminanceFlag              = true;
    normalizeStdDev                     = true;
    jitterFlag                          = 0;
    padValue                            = 0;
    inputPath                           = imageListPath;
    displayPeriod                       = displayPeriod;
    echoFramePathnameFlag               = true;
    start_frame_index                   = 0;
    batchMethod                         = "bySpecified";
    skip_frame_index                    = 0;
    writeFrameToTimestamp               = true;
    flipOnTimescaleError                = true;
    resetToStartOnLoop                  = false;
 }
)


pv.addGroup(pvParameters, "ImageReconS1",
{
groupType = "ANNLayer";
ANNNormalizedErrorLayer "ImageDeconError" = {
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 1;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerFlag                         = false;
    writeStep                           = 4800;
    initialWriteTime                    = 4800;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = NULL;
    VThresh                             = 0;
    AMin                                = 0;
    AMax                                = 3.40282e+38;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
    errScale                            = 1;
    )}







pv.addGroup(pvParameters, "ImageReconS1ExpertsError", pvParameters["ImageReconS1Error"])

pv.addGroup(pvParameters, "ImageReconS1",
{
groupType = "ANNLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = 3;
    phase                               = 7;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "ZeroV";
    triggerFlag                         = false;
    writeStep                           = writePeriod;
    initialWriteTime                    = writePeriod;
    sparseLayer                         = false;
    updateGpu                           = false;
    dataType                            = nil;
    VThresh                             = -inf;
    AMin                                = -inf;
    AMax                                = inf;
    AShift                              = 0;
    VWidth                              = 0;
    clearGSynInterval                   = 0;
}
)



pv.addGroup(pvParameters, "ImageReconS1Experts", pvParameters["ImageReconS1"])

pv.addGroup(pvParameters, "GroundTruth", pvParameters["Image"],
{
    groupType = "MoviePvp";
    nf                                  = numClasses;
    sparseLayer                         = true;
    writeSparseValues                   = false;
    autoResizeFlag                      = false;
    normalizeLuminanceFlag              = false;
    inputPath                           = GroundTruthPath;
    displayPeriod                       = displayPeriod;
    --readPvpFile                         = true;
}
)
pvParameters["GroundTruth"].readPvpFile    = true;

pv.addGroup(pvParameters, "Background", 
{
groupType = "BackgroundLayer";
    nxScale                             = 1;
    nyScale                             = 1;
    nf                                  = numClasses + 1;
    phase                               = 1;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    triggerFlag                         = true;
    triggerLayerName                    = "GroundTruth";
    triggerOffset                       = 0;
    writeStep                           = -1;
    sparseLayer                         = true;
    writeSparseValues                   = false;
    updateGpu                           = false;
    dataType                            = nil;
    originalLayerName                   = "GroundTruth";
    repFeatureNum                       = 1;
}
)

phase = phase + 1
pv.addGroup(pvParameters, "ImageDecon", defaultANNLayer, 
{
    phase                               = phase;
}
) 
    
pv.addGroup(pvParameters, "ImageRecon", defaultTriggeredANNLayer, 
{
    phase                               = phase;
}
) 
    



--connections 
pv.addGroup(pvParameters, "ImageToImageReconS1Error",
{
groupType = "HyPerConn";
    preLayerName                        = "Image";
    postLayerName                       = "ImageReconS1Error";
    channelCode                         = 0;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = false;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "OneToOneWeights";
    initWeightsFile                     = nil;
    weightInit                          = 0.032075; --((1/patchSizeMin)*(1/patchSizeMin)*(1/3))^(1/2); --
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
}
)

pv.addGroup(pvParameters, "ImageToImageReconS1ExpertsError", pvParameters["ImageToImageReconS1Error"],{
    preLayerName                        = "Image";
    postLayerName                       = "ImageReconS1ExpertsError";
}
)

pv.addGroup(pvParameters, "ImageReconS1ToImageReconS1Error",
{
groupType = "IdentConn";
    preLayerName                        = "ImageReconS1";
    postLayerName                       = "ImageReconS1Error";
    channelCode                         = 1;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
    writeStep                           = -1;
}
)

pv.addGroup(pvParameters, "ImageReconS1ExpertsToImageReconS1ExpertsError", pvParameters["ImageReconS1ToImageReconS1Error"],{
    preLayerName                        = "ImageReconS1Experts";
    postLayerName                       = "ImageReconS1ExpertsError";
}
)


--begin loop over scales

local stride = strideMin
local numExperts = numExpertsMax
local patchSize = patchSizeMin
local tau = tauMin

for i_scale = 1, numScales do

pv.addGroup(pvParameters, "S1Stride" .. stride,
{
groupType = "HyPerLCALayer";
    nxScale                             = 1.0/stride;
    nyScale                             = 1.0/stride;
    nf                                  = numExperts*(numClasses+1);
    phase                               = 2;
    mirrorBCflag                        = false;
    valueBC                             = 0;
    initializeFromCheckpointFlag        = false;
    InitVType                           = "UniformRandomV";
    minV                                = -1;
    maxV                                = 0.05;
    triggerFlag                         = false;
    writeStep                           = displayPeriod;
    initialWriteTime                    = displayPeriod;
    sparseLayer                         = true;
    writeSparseValues                   = true;
    updateGpu                           = true;
    dataType                            = nil;
    VThresh                             = 0.025;
    AMin                                = 0;
    AMax                                = inf;
    AShift                              = 0;
    VWidth                              = 10;
    clearGSynInterval                   = 0;
    timeConstantTau                     = tau;
    numWindowX                          = 1;
    numWindowY                          = 1;
    selfInteract                        = true;
}
)

pv.addGroup(pvParameters,
	"ImageReconS1Stride" .. stride, pvParameters["ImageReconS1"],
{
    phase                               = 6;
}
)

pv.addGroup(pvParameters, "S1Stride" .. stride .. "ToImageReconS1Error",
{
groupType = "MomentumConn";
    preLayerName                        = "S1Stride" .. stride;
    postLayerName                       = "ImageReconS1Error";
    channelCode                         = -1;
    delay                               = {0.000000};
    numAxonalArbors                     = 1;
    plasticityFlag                      = true;
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    sharedWeights                       = true;
    weightInitType                      = "UniformRandomWeight";
    initWeightsFile                     = nil;
    wMinInit                            = -1;
    wMaxInit                            = 1;
    sparseFraction                      = 0.9;
    initializeFromCheckpointFlag        = false;
    triggerFlag                         = true;
    triggerLayerName                    = "Image";
    triggerOffset                       = 1;
    updateGSynFromPostPerspective       = true;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    combine_dW_with_W_flag              = false;
    nxp                                 = patchSize;
    nyp                                 = patchSize;
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
    dWMax                               = 1.0;
    keepKernelsSynchronized             = true;
    useMask                             = false;
    momentumTau                         = 200;
    momentumMethod                      = "viscosity";
    momentumDecay                       = 0;
}
)

pv.addGroup(pvParameters, "ImageReconS1ErrorToS1Stride" .. stride,
{
groupType = "TransposeConn";
    preLayerName                        = "ImageReconS1Error";
    postLayerName                       = "S1Stride" .. stride;
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
    originalConnName                    = "S1Stride" .. stride .. "ToImageReconS1Error";
}
)

pv.addGroup(pvParameters, "S1Stride" .. stride .. "ToImageReconS1Stride" .. stride,
{
groupType = "CloneConn";
    preLayerName                        = "S1Stride" .. stride;
    postLayerName                       = "ImageReconS1Stride" .. stride;
    channelCode                         = 0;
    delay                               = {0.000000};
    convertRateToSpikeCount             = false;
    receiveGpu                          = false;
    updateGSynFromPostPerspective       = false;
    pvpatchAccumulateType               = "convolve";
    writeStep                           = -1;
    writeCompressedCheckpoints          = false;
    selfFlag                            = false;
    originalConnName                    = "S1Stride" .. stride .. "ToImageReconS1Error";
}
)

pv.addGroup(pvParameters, "ImageReconS1Stride" .. stride .. "ToImageReconS1",
{
groupType = "IdentConn";
    preLayerName                        = "ImageReconS1Stride" .. stride;
    postLayerName                       = "ImageReconS1";
    channelCode                         = 0;
    delay                               = {0.000000};
    initWeightsFile                     = nil;
    writeStep                           = -1;
}
)

pv.addGroup(pvParameters, "ImageReconS1ExpertsStride" .. stride .. "ToImageReconS1Experts", pvParameters["ImageReconS1Stride" .. stride .. "ToImageReconS1"],
{
    preLayerName                        = "ImageReconS1ExpertsStride" .. stride;
    postLayerName                       = "ImageReconS1Experts";
}
)


pv.addGroup(pvParameters,
	"ImageReconS1ExpertsStride" .. stride, pvParameters["ImageReconS1Stride" .. stride],
{
    phase                               = 6;
}
)


for i_expert = 0 , numClasses do

  pv.addGroup(pvParameters,
  	"ImageReconS1Expert" .. i_expert .. "Stride" .. stride, pvParameters["ImageReconS1ExpertsStride" .. stride],
{
    phase                               = 5;
}
)


   pv.addGroup(pvParameters,
   "S1Expert" .. i_expert .. "Stride" .. stride, pvParameters["S1Stride" .. stride],
{
    nf                                 = numExperts;
}
)

pv.addGroup(pvParameters, "S1Expert" .. i_expert .. "Stride" .. stride .. "ToImageReconS1ExpertsError", pvParameters["S1Stride" .. stride .. "ToImageReconS1Error"],
{
    preLayerName                        = "S1Expert" .. i_expert .. "Stride" .. stride;
    postLayerName                       = "ImageReconS1ExpertsError";
    useMask                             = true;
}
)
pvParameters["S1Expert" .. i_expert .. "Stride" .. stride .. "ToImageReconS1ExpertsError"].maskLayerName = "Background";
pvParameters["S1Expert" .. i_expert .. "Stride" .. stride .. "ToImageReconS1ExpertsError"].maskFeatureIdx = i_expert;

pv.addGroup(pvParameters, "ImageReconS1ExpertsErrorToS1Expert" .. i_expert .. "Stride" .. stride, pvParameters["ImageReconS1ErrorToS1Stride" .. stride],
{
    preLayerName                        = "ImageReconS1ExpertsError";
    postLayerName                       = "S1Expert" .. i_expert .. "Stride" .. stride;
    originalConnName                    = "S1Expert" .. i_expert .. "Stride" .. stride .. "ToImageReconS1ExpertsError";
}
)

pv.addGroup(pvParameters, "S1Expert" .. i_expert .. "Stride" .. stride .. "ToImageReconS1Expert" .. i_expert .. "Stride" .. stride, pvParameters["S1Stride" .. stride .. "ToImageReconS1Stride" .. stride],
{
    preLayerName                        = "S1Expert" .. i_expert .. "Stride" .. stride;
    postLayerName                       = "ImageReconS1Expert" .. i_expert .. "Stride" .. stride;
    originalConnName                    = "S1Expert" .. i_expert .. "Stride" .. stride .. "ToImageReconS1ExpertsError";
}
)

pv.addGroup(pvParameters,
	"ImageReconS1Expert" .. i_expert .. "Stride" .. stride .. "ToImageReconS1ExpertsStride" .. stride, pvParameters["ImageReconS1ExpertsStride" .. stride .. "ToImageReconS1Experts"],
{
   preLayerName                        = "ImageReconS1Expert" .. i_expert .. "Stride" .. stride;
   postLayerName                       = "ImageReconS1ExpertsStride" .. stride;
}
)


end -- loop over experts

stride            = stride * strideMultiplier
numExperts        = numExperts * numExpertsMultiplier
patchSize         = patchSize * patchSizeMultiplier
tau               = tau * tauMultiplier

end -- end loop over scales




-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParameters)
