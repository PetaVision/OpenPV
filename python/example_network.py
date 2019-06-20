from ParameterWrapper import infinity, INFINITY, nil, NULL, null, true, false 
import ParameterWrapper as params

batch_size     = 1
batch_width    = 1
threads        = 4
rows           = 1
cols           = 1

num_samples    = 50000
epochs         = 1 
display_period = 500 
stop_time      = num_samples / batch_size * display_period * epochs
cp_interval    = 5

folder_name    = 'cifar'

input_path     = '/shared/cifar-10-batches-mat/mixed_cifar.txt'
input_width    = 32
input_height   = 32
input_features = 3
patch          = 8
stride         = 2
dictionary     = 128 
thresh         = 0.25 
learning_rate  = 0.1

weights_folder = None; #folder_name + '/weights/';

pvParams = {
    'column' : {
        'groupType'                         : 'HyPerCol',
        'dt'                                : 1,
        'stopTime'                          : stop_time,
        'progressInterval'                  : 1,
        'writeProgressToErr'                : True,
        'verifyWrites'                      : False,
        'outputPath'                        : folder_name,
        'printParamsFilename'               : folder_name + '.params',
        'randomSeed'                        : 1234567891,
        'nx'                                : input_width,
        'ny'                                : input_height,
        'nbatch'                            : batch_size,
        'checkpointWrite'                   : True,
        'checkpointWriteDir'                : folder_name + '/Checkpoints',
        'checkpointWriteTriggerMode'        : 'step',
        'checkpointWriteStepInterval'       : display_period * cp_interval,
        'checkpointIndexWidth'              : -1,
        'deleteOlderCheckpoints'            : True,
        'suppressNonplasticCheckpoints'     : False,
        'initializeFromCheckpointDir'       : '',
        'errorOnNotANumber'                 : False
    }
}

params.addGroup(pvParams, 'Test',  {
            'groupType'                     : 'LeakyIntegrator',
            'nxScale'                       : 1.0,
            'nyScale'                       : 1.0,
            'nf'                            : input_features,
            'phase'                         : 1,
            'writeStep'                     : -1,
            'initialWriteTime'              : -1,
            'integrationTime'               : infinity,
            'InitVType'                     : 'ZeroV'
        }
   )


params.addGroup(pvParams, 'Image',  {
            'groupType'                     : 'ImageLayer',
            'nxScale'                       : 1.0,
            'nyScale'                       : 1.0,
            'nf'                            : input_features,
            'phase'                         : 1,
            'writeStep'                     : -1,
            'initialWriteTime'              : -1,
            'offsetAnchor'                  : 'cc',
            'inverseFlag'                   : False,
            'normalizeLuminanceFlag'        : True,
            'normalizeStdDev'               : True,
            'autoResizeFlag'                : False,
            'batchMethod'                   : 'random',
            'writeFrameToTimestamp'         : True,
            'resetToStartOnLoop'            : False,
            'displayPeriod'                 : display_period,
            'inputPath'                     : input_path
        }
   )

params.addGroup(pvParams, 'ImageError', {
            'groupType'                     : 'HyPerLayer',
            'nxScale'                       : 1,
            'nyScale'                       : 1,
            'nf'                            : input_features,
            'phase'                         : 2,
            'writeStep'                     : -1,
            'initialWriteTime'              : -1,
            'InitVType'                     : 'ZeroV'
        }
   )

params.addGroup(pvParams, 'S1', {
            'groupType'                     : 'HyPerLCALayer',
            'nxScale'                       : 1 / stride,
            'nyScale'                       : 1 / stride,
            'nf'                            : dictionary,
            'phase'                         : 3,
            'InitVType'                     : 'ConstantV',
            'valueV'                        : thresh * 0.9,
            'triggerLayerName'              : NULL,
            'sparseLayer'                   : True,
            'writeSparseValues'             : True,
            'updateGpu'                     : True,
            'VThresh'                       : thresh,
            'AMin'                          : 0,
            'AMax'                          : infinity,
            'AShift'                        : thresh,
            'VWidth'                        : 0,
            'timeConstantTau'               : 100,
            'selfInteract'                  : True,
            'adaptiveTimeScaleProbe'        : 'AdaptProbe',
            'writeStep'                     : -1,
            'initialWriteTime'              : -1
        }
   )

params.addGroup(pvParams, 'ImageRecon',  {
            'groupType'                     : 'HyPerLayer',
            'nxScale'                       : 1,
            'nyScale'                       : 1,
            'nf'                            : input_features,
            'phase'                         : 4,
            'InitVType'                     : 'ZeroV',
            'writeStep'                     : -1,
            'initialWriteTime'              : -1
        }
   )

params.addGroup(pvParams, 'ImageToImageError', {
            'groupType'                     : 'IdentConn',
            'preLayerName'                  : 'Image',
            'postLayerName'                 : 'ImageError',
            'channelCode'                   : 0
        }
   )

params.addGroup(pvParams, 'ImageErrorToS1', {
            'groupType'                     : 'TransposeConn',
            'preLayerName'                  : 'ImageError',
            'postLayerName'                 : 'S1',
            'channelCode'                   : 0,
            'receiveGpu'                    : True,
            'updateGSynFromPostPerspective' : True,
            'pvpatchAccumulateType'         : 'convolve',
            'writeStep'                     : -1,
            'originalConnName'              : 'S1ToImageError'
        }
   )

params.addGroup(pvParams, 'S1ToImageError', {
            'groupType'                     : 'MomentumConn',
            'preLayerName'                  : 'S1',
            'postLayerName'                 : 'ImageError',
            'channelCode'                   : -1,
            'plasticityFlag'                : True,
            'sharedWeights'                 : True,
            'weightInitType'                : 'UniformRandomWeight',
            'wMinInit'                      : -1,
            'wMaxInit'                      : 1,
            'minNNZ'                        : 1,
            'sparseFraction'                : 0.99,
            'triggerLayerName'              : 'Image',
            'pvpatchAccumulateType'         : 'convolve',
            'nxp'                           : patch,
            'nyp'                           : patch,
            'normalizeMethod'               : 'normalizeL2',
            'strength'                      : 1,
            'normalizeOnInitialize'         : True,
            'normalizeOnWeightUpdate'       : True,
            'minL2NormTolerated'            : 0,
            'dWMax'                         : learning_rate,
            'timeConstantTau'               : 500,
            'momentumMethod'                : 'viscosity',
            'momentumDecay'                 : 0,
            'initialWriteTime'              : -1,
            'writeStep'                     : -1
        }
    )

if weights_folder != None:
    pvParams['S1ToImageError']['weightInitType'] = 'FileWeight'
    pvParams['S1ToImageError']['initWeightsFile'] = weights_folder + 'S1ToImageError_W.pvp'

params.addGroup(pvParams, 'S1ToImageRecon', {
            'groupType'                     : 'CloneConn',
            'preLayerName'                  : 'S1',
            'postLayerName'                 : 'ImageRecon',
            'channelCode'                   : 0,
            'pvpatchAccumulateType'         : 'convolve',
            'originalConnName'              : 'S1ToImageError'
        }
   )

params.addGroup(pvParams, "ImageReconToImageError", {
            'groupType'                     : 'IdentConn',
            'preLayerName'                  : 'ImageRecon',
            'postLayerName'                 : 'ImageError',
            'channelCode'                   : 1
        }
   )

params.addGroup(pvParams, 'AdaptProbe', {
            'groupType'                     : 'KneeTimeScaleProbe',
            'targetName'                    : 'EnergyProbe',
            'textOutputFlag'                : True,
            'probeOutputFile'               : 'AdaptiveTimeScales.txt',
            'triggerLayerName'              : 'Image',
            'triggerOffset'                 : 0,
            'baseMax'                       : 0.011,
            'baseMin'                       : 0.01,
            'tauFactor'                     : 0.025,
            'growthFactor'                  : 0.025,
            'writeTimeScales'               : True,
            'kneeThresh'                    : 10,
            'kneeSlope'                     : 0.025
        }
   )

params.addGroup(pvParams, 'EnergyProbe', {
            'groupType'                     : 'ColumnEnergyProbe',
            'message'                       : None,
            'textOutputFlag'                : True,
            'probeOutputFile'               : 'EnergyProbe.txt',
            'triggerLayerName'              : None,
            'energyProbe'                   : None
        }
   )

params.addGroup(pvParams, 'ImageErrorL2Probe', {
            'groupType'                     : 'L2NormProbe',
            'targetLayer'                   : 'ImageError',
            'message'                       : None,
            'textOutputFlag'                : True,
            'probeOutputFile'               : 'ImageErrorL2.txt',
            'energyProbe'                   : 'EnergyProbe',
            'coefficient'                   : 0.5,
            'maskLayerName'                 : None,
            'exponent'                      : 2
        }
   )

params.addGroup(pvParams, 'S1L1Probe', {
            'groupType'                     : 'L1NormProbe',
            'targetLayer'                   : 'S1',
            'message'                       : None,
            'textOutputFlag'                : True,
            'probeOutputFile'               : 'S1L1Probe.txt',
            'energyProbe'                   : 'EnergyProbe',
            'coefficient'                   : pvParams['S1']['VThresh'],
            'maskLayerName'                 : None
        }
   )

#print(params.createParamsFileString(pvParams, False))
