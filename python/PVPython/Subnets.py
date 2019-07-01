import PVPython as pv

def addLCASubnet(params,
                 lcaLayerName,
                 inputLayerName,
                 inputValueScale,
                 stride,
                 lcaParams,
                 connParams):

    inputLayer     = params[inputLayerName]
    errorLayerName = inputLayerName + 'Error' + lcaLayerName
    reconLayerName = inputLayerName + 'Recon' + lcaLayerName

    inputToError = {
        'groupType'     : 'IdentConn',
        'preLayerName'  : inputLayerName,
        'postLayerName' : errorLayerName,
        'channelCode'   : 0,
        'writeStep'     : -1
    }
    if inputValueScale != 1.0:
        inputToError['groupType'] = 'RescaleConn'
        inputToError['scale']     = inputValueScale
    pv.addGroup(params, inputLayerName + 'To' + errorLayerName, inputToError)

    errorLayer = {
        'groupType'        : 'HyPerLayer',
        'nxScale'          : inputLayer['nxScale'],
        'nyScale'          : inputLayer['nyScale'],
        'nf'               : inputLayer['nf'],
        'phase'            : inputLayer['phase'] + 1,
        'InitVType'        : 'ZeroV',
        'writeStep'        : -1,
        'initialWriteTime' : -1
    }
    pv.addGroup(params, errorLayerName, errorLayer)

    reconLayer = {
        'groupType'        : 'HyPerLayer',
        'nxScale'          : inputLayer['nxScale'],
        'nyScale'          : inputLayer['nyScale'],
        'nf'               : inputLayer['nf'],
        'phase'            : inputLayer['phase'] + 3,
        'InitVType'        : 'ZeroV',
        'writeStep'        : -1,
        'initialWriteTime' : -1
    }
    pv.addGroup(params, reconLayerName, reconLayer)

    lcaLayer = {
        'groupType'        : 'HyPerLCALayer',
        'nxScale'          : inputLayer['nxScale'] / stride,
        'nyScale'          : inputLayer['nyScale'] / stride,
        'phase'            : inputLayer['phase'] + 2,
        'sparseLayer'      : True,
        'updateGpu'        : True,
        'writeSparseValues': True,
        'selfInteract'     : True,
        'writeStep'        : -1,
        'initialWriteTime' : -1
    }
    for k, v in lcaParams.items():
        lcaLayer[k] = v
    pv.addGroup(params, lcaLayerName, lcaLayer)

    lcaToError = {
        'groupType'        : 'MomentumConn',
        'preLayerName'     : lcaLayerName,
        'postLayerName'    : errorLayerName,
        'channelCode'      : -1,
        'writeStep'        : -1
    }
    for k, v in connParams.items():
        lcaToError[k] = v
    pv.addGroup(params, lcaLayerName + 'To' + errorLayerName, lcaToError)

    errorToLca = {
        'groupType'        : 'TransposeConn',
        'preLayerName'     : errorLayerName,
        'postLayerName'    : lcaLayerName,
        'originalConnName' : lcaLayerName + 'To' + errorLayerName,
        'channelCode'      : 0,
        'writeStep'        : -1,
        'receiveGpu'       : True,
        'updateGSynFromPostPerspective' : True
    }
    pv.addGroup(params, errorLayerName + 'To' + lcaLayerName, errorToLca)

    lcaToRecon = {
        'groupType'        : 'CloneConn',
        'preLayerName'     : lcaLayerName,
        'postLayerName'    : reconLayerName,
        'originalConnName' : lcaLayerName + 'To' + errorLayerName,
        'channelCode'      : 0,
        'writeStep'        : -1,
    }
    pv.addGroup(params, lcaLayerName + 'To' + reconLayerName, lcaToRecon)
    
    reconToError = {
        'groupType'     : 'IdentConn',
        'preLayerName'  : reconLayerName,
        'postLayerName' : errorLayerName,
        'channelCode'   : 1,
        'writeStep'     : -1
    }
    pv.addGroup(params, reconLayerName + 'To' + errorLayerName, reconToError)

