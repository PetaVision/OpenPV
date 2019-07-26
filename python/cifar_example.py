import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
sys.path.append("/PATH_TO_OPENPV/python")
import OpenPV as pv 

epochs         = 5
num_images     = 50000 * epochs
display_period = 1000
batches        = 32 
stop_time      = display_period * num_images / batches
threshold      = 0.125
dictionary     = 256 
learning_rate  = 1.0

folder_name   = 'cifar'

args = {   'NumThreads' : 2,
           'NumRows'    : 1,
           'NumColumns' : 1,
           'BatchWidth' : 16,
           'LogFile'    : folder_name + '.log'
}

params = {
    'column' : {
        'groupType'                         : 'HyPerCol',
        'dt'                                : 1,
        'stopTime'                          : stop_time,
        'progressInterval'                  : display_period / 4, 
        'writeProgressToErr'                : True,
        'verifyWrites'                      : False,
        'outputPath'                        : folder_name,
        'printParamsFilename'               : folder_name + '.params',
        'randomSeed'                        : 1234567892,
        'nx'                                : 32,
        'ny'                                : 32,
        'nbatch'                            : batches,
        'checkpointWrite'                   : True,
        'checkpointWriteDir'                : folder_name + '/Checkpoints',
        'checkpointWriteTriggerMode'        : 'step',
        'checkpointWriteStepInterval'       : display_period * 10,
        'checkpointIndexWidth'              : -1,
        'deleteOlderCheckpoints'            : True,
        'suppressNonplasticCheckpoints'     : True,
        'initializeFromCheckpointDir'       : '',
        'errorOnNotANumber'                 : False
    }
}

pv.addGroup(params, 'Input',  {
            'groupType'                     : 'ImageLayer',
            'nxScale'                       : 1.0,
            'nyScale'                       : 1.0,
            'nf'                            : 3,
            'phase'                         : 0,
            'writeStep'                     : -1,
            'initialWriteTime'              : -1,
            'InitVType'                     : 'ZeroV',
            'offsetAnchor'                  : 'cc',
            'inverseFlag'                   : False,
            'normalizeLuminanceFlag'        : True,
            'normalizeStdDev'               : True,
            'autoResizeFlag'                : False,
            'batchMethod'                   : 'random',
            'writeFrameToTimestamp'         : True,
            'resetToStartOnLoop'            : False,
            'displayPeriod'                 : display_period,
            'inputPath'                     : '/shared/cifar-10-batches-mat/mixed_cifar.txt'
        }
   )

pv.addLCASubnet(params,
                 lcaLayerName    = 'S1',
                 inputLayerName  = 'Input',
                 inputValueScale = 1.0,
                 stride          = 2,
                 lcaParams       = {
                    'nf'                      : dictionary,
                    'timeConstantTau'         : 250.0,
                    'AMin'                    : 0.0,
                    'AMax'                    : float('inf'),
                    'AShift'                  : threshold,
                    'VThresh'                 : threshold,
                    'VWidth'                  : 0.0,
                    'InitVType'               : 'ZeroV',
                    'adaptiveTimeScaleProbe'  : 'AdaptProbe'
                 },
                 connParams      = {
                    'groupType'               : 'MomentumConn',
                    'nxp'                     : 8,
                    'nyp'                     : 8,
                    'weightInitType'          : 'UniformRandomWeight',
                    'wMinInit'                : -1.0,
                    'wMaxInit'                : 1.0,
                    'minNNZ'                  : 1,
                    'sparseFraction'          : 0.975,
                    'normalizeMethod'         : 'normalizeL2',
                    'strength'                : 1,
                    'normalizeOnInitialize'   : True,
                    'normalizeOnWeightUpdate' : True,
                    'minL2NormTolerated'      : 0,
                    'plasticityFlag'          : True,
                    'dWMax'                   : learning_rate,
                    'triggerLayerName'        : 'Input',
                    'triggerOffset'           : 1,
                    'timeConstantTau'         : 50,
                    'momentumMethod'          : 'viscosity',
                    'momentumDecay'           : 0
                 })

pv.addGroup(params, 'AdaptProbe', {
            'groupType'                     : 'AdaptiveTimeScaleProbe',
            'targetName'                    : 'EnergyProbe',
            'textOutputFlag'                : False,
            'probeOutputFile'               : 'AdaptiveTimeScales.txt',
            'triggerLayerName'              : 'Input',
            'triggerOffset'                 : 0,
            'baseMax'                       : 0.011,
            'baseMin'                       : 0.01,
            'tauFactor'                     : 0.01,
            'growthFactor'                  : 0.01
        }
   )


# Analysis process callbacks

def recon_analysis(**kwargs):
        im1 = kwargs['Input']
        im2 = kwargs['InputErrorS1']
        im3 = kwargs['InputReconS1']
        im_row = np.concatenate((im1, im2, im3), axis=1)
        full = np.concatenate(np.split(im_row, np.shape(im_row)[0], 0), axis=2)
        if np.min(full) != np.max(full):
            full = (full - np.min(im1)) / (np.max(im1) - np.min(im1))
        plt.ion()
        fig = plt.figure(1)
        fig.canvas.set_window_title('timestep ' + str(kwargs['simtime']))
        plt.show()
        plt.clf()
        plt.imshow(np.clip(np.squeeze(full), 0.0, 1.0))
        plt.draw()
        plt.pause(0.001)
        fig.savefig('recon_analysis.png')

def weight_analysis(**kwargs):
      w = kwargs['S1ToInputErrorS1']
      w = np.split(w, np.shape(w)[0], 0)
      sq = math.sqrt(len(w))
      width  = math.ceil(np.shape(w[0])[2] * sq)
      height = math.ceil(np.shape(w[0])[1] * sq)
      ratio = height / width
      per_row = math.ceil(sq * ratio)
      rows = []
      for i in range(math.ceil(len(w)/per_row)):
         rows.append(np.concatenate(w[i*per_row:min(len(w), (i+1)*per_row)], axis=2))
      w_full = np.concatenate(rows, axis=1)
      if np.min(w_full) != np.max(w_full):
          w_full = (w_full - np.min(w_full)) / (np.max(w_full) - np.min(w_full))
      plt.ion()
      fig = plt.figure(2)
      fig.canvas.set_window_title('timestep ' + str(kwargs['simtime']))
      plt.show()
      plt.clf()
      plt.imshow(np.squeeze(w_full))
      plt.draw()
      plt.pause(0.001)
      fig.savefig('weight_analysis.png')


#Create our Run object

run = pv.Runner(
      args=args,
      params=pv.createParamsFileString(params))

# Set up our analysis processes

run.analyze(recon_analysis, display_period)   \
      .watch('Input',        pv.PVType.LAYER_A) \
      .watch('InputErrorS1', pv.PVType.LAYER_A) \
      .watch('InputReconS1', pv.PVType.LAYER_A)

run.analyze(weight_analysis, display_period)  \
      .watch('S1ToInputErrorS1', pv.PVType.CONNECTION)

run.run()

input("Finished, press any key to continue...")
