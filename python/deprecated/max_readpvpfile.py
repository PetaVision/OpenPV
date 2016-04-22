from .readpvpheader import readpvpheader,extendedHeaderPattern
from .pv_object import PV_Object
import numpy as np
import os

class DataFrame(tuple):
    def __new__ (cls, time, values):
        return super(DataFrame, cls).__new__(cls, tuple([time,values]))
    def __init__(self, time, values):
        self.time = time
        self.values = values

def readpvpfile(filename,
                progressPeriod=0,
                lastFrame=float('inf'),
                startFrame=0,
                skipFrames=1):
    import os
    assert startFrame >= 0
    assert lastFrame >= 1
    assert progressPeriod >= 0
    assert skipFrames >= 1
    assert startFrame < lastFrame
    dataTypeSwitch = {1: np.uint8,
                      2: np.int32,
                      3: np.float32,
                      4: np.int32}
    with open(filename,'rb') as stream:
        header = readpvpheader(stream)
        dataType = dataTypeSwitch[header['datatype']]
        lastFrame = min(lastFrame, header['nbands'])

# Deprecated filetype not implemented in python (requires oct2py):

        # PVP FILE (deprecated)
        if header['filetype'] == 1:
            from oct2py import octave
            import re
            octave.addpath(re.match('(.*)(python)',__file__).group(0) + '/mlab/util')
            raw_data = octave.readpvpfile(filename)
            return raw_data

# Supported older filetypes, not fully tested in Python 2

        # SPIKING ACTIVITY FILE
        if header['filetype'] == 2:
            lastFrame = min(lastFrame, header['nbands'])
            data = []
            for frame in range(lastFrame):
                if frame < startFrame or (frame % skipFrames):
                    numActive = np.fromfile(stream,np.uint32,3)[-1]
                    stream.seek(np.dtype(np.uint32).itemsize * numActive,
                                os.SEEK_CUR)
                    continue
                else:
                    time = np.fromfile(stream,np.float64,1)[0]
                    numActive = np.fromfile(stream,np.uint32,1)
                    currentData = np.fromfile(stream,np.uint32,numActive)
                    data.append(DataFrame(time, currentData))
                    if progressPeriod:
                        if not frame % progressPeriod and frame:
                            print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))
            return PV_Object(data,header)

        # NON-KERNEL WEIGHT FILE
        elif header['filetype'] == 3:
            fileSize = os.path.getsize(filename)
            frameSize = header['recordsize'] * header['nbands'] + header['headersize']
            lastFrame = min(lastFrame,fileSize/frameSize)
            shape = (header['nxp'], header['nyp'], header['nfp'])
            patchPattern = np.dtype([('nx', np.uint16),
                                     ('ny', np.uint16),
                                     ('offset', np.uint32),
                                     ('values', dataType, shape)])
            data = []
            stream.seek(0)
            for frame in range(startFrame, lastFrame):
                if not frame % skipFrames:
                    stream.seek(frame*frameSize)
                    time = np.fromfile(stream,extendedHeaderPattern,1)['time'][0]
                    data.append(DataFrame(time,[]))
                    for arbor in range(header['nbands']):
                        currentData = np.fromfile(stream,
                                                  patchPattern,
                                                  header['numpatches'])['values']
                        data[frame].values.append(np.squeeze(np.transpose(currentData,
                                                                          [2,1,3,0])))
                        if progressPeriod:
                            if not frame % progressPeriod and frame:
                                print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))
                    return PV_Object(data,header)

# Supported newer filetypes, fully implemented in python and tested as of 2/2/2016

        # DENSE, NON-SPIKING ACTIVITY FILE
        elif header['filetype'] == 4:
            lastFrame = min(lastFrame, header['nbands'])
            shape = (header['ny'], header['nx'], header['nf'])
            pattern = np.dtype([('time', np.float64),
                                ('values', dataType, shape)])
            frameSize = pattern.itemsize
            data = []
            for frame in range(startFrame, lastFrame):
                if not frame % skipFrames:
                    stream.seek(header['headersize'] + frame*frameSize)
                    currentData = np.fromfile(stream, pattern, 1)
                    data.append(DataFrame(currentData['time'][0],
                                          currentData['values'][0]))
                    if progressPeriod:
                        if not frame % progressPeriod and frame:
                            print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))
            return PV_Object(data,header)

        # KERNEL WEIGHT FILE
        elif header['filetype'] == 5:
            fileSize = os.path.getsize(filename)
            frameSize = header['recordsize'] * header['nbands'] + header['headersize']
            lastFrame = min(lastFrame,fileSize/frameSize)
            shape = (header['nxp'], header['nyp'], header['nfp'])
            patchPattern = np.dtype([('nx', np.uint16),
                                     ('ny', np.uint16),
                                     ('offset', np.uint32),
                                     ('values', dataType, shape)])
            data = []
            stream.seek(0)
            for frame in range(startFrame, lastFrame):
                if not frame % skipFrames:
                    stream.seek(frame*frameSize)
                    time = np.fromfile(stream,extendedHeaderPattern,1)['time'][0]
                    data.append(DataFrame(time,[]))
                    for arbor in range(header['nbands']):
                        currentData = np.fromfile(stream,
                                                  patchPattern,
                                                  header['numpatches'])['values']
                        data[frame].values.append(np.squeeze(np.transpose(currentData,
                                                                          [2,1,3,0])))
                        if progressPeriod:
                            if not frame % progressPeriod and frame:
                                print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))
                    return PV_Object(data,header)

        # SPARSE ACTIVITY FILE
        elif header['filetype'] == 6:
            lastFrame = min(lastFrame, header['nbands'])
            entryPattern = np.dtype([('index', np.int32),
                                     ('activation', np.float32)])
            data = []
            for frame in range(lastFrame):
                if frame < startFrame or (frame % skipFrames):
                    numActive = np.fromfile(stream,np.uint32,3)[-1]
                    stream.seek(entryPattern.itemsize * numActive,
                                os.SEEK_CUR)
                    continue
                else:
                    time = np.fromfile(stream,np.float64,1)[0]
                    numActive = np.fromfile(stream,np.uint32,1)
                    currentData = np.fromfile(stream,entryPattern,numActive)
                    data.append(DataFrame(time,zip(currentData['index'],
                                                   currentData['activation'])))
                    if progressPeriod:
                        if not frame % progressPeriod and frame:
                            print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))
            return PV_Object(data,header)
