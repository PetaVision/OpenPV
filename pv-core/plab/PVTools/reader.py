import numpy as np

class DataFrame:
    def __init__(self,time,values):
        self.time = time
        self.values = values

headerPattern = [('headersize', np.int32),
                 ('numparams', np.int32),
                 ('filetype', np.int32),
                 ('nx', np.int32),
                 ('ny', np.int32),
                 ('nf', np.int32),
                 ('numrecords', np.int32),
                 ('recordsize', np.int32),
                 ('datasize', np.int32),
                 ('datatype', np.int32),
                 ('nxprocs', np.int32),
                 ('nyprocs', np.int32),
                 ('nxGlobal', np.int32),
                 ('nyGlobal', np.int32),
                 ('kx0', np.int32),
                 ('ky0', np.int32),
                 ('nbatch', np.int32),
                 ('nbands', np.int32),
                 ('time', np.float64)]

headerPattern_ext = headerPattern + [('nxp', np.int32),
                                     ('nyp', np.int32),
                                     ('nfp', np.int32),
                                     ('wMin', np.float32),
                                     ('wMax', np.float32),
                                     ('numpatches', np.int32)]

headerPattern = np.dtype(headerPattern)
headerPattern_ext = np.dtype(headerPattern_ext)

def readpvpheader(fileStream):
    header = np.fromfile(fileStream,headerPattern,1)
    if header['numparams'] == 26:
        fileStream.seek(0)
        header = np.fromfile(fileStream,headerPattern_ext,1)
    return zip(header.dtype.names,header[0])

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
        header = dict(readpvpheader(stream))
        dataType = dataTypeSwitch[header['datatype']]
        lastFrame = min(lastFrame, header['nbands'])

# Older filetypes, not fully implemented in python (require oct2py):

        # PVP FILE (deprecated)
        if header['filetype'] == 1:
            from oct2py import octave
            import re
            octave.addpath(re.match('(.*)(plab)',__file__).group(0) + '/mlab/util')
            raw_data = octave.readpvpfile(filename)
            return raw_data

        # SPIKING ACTIVITY FILE
        if header['filetype'] == 2:
            from oct2py import octave
            import re
            octave.addpath(re.match('(.*)(plab)',__file__).group(0) + '/mlab/util')
            raw_data = octave.readpvpfile(filename)
            return raw_data

        # NON-KERNEL WEIGHT FILE
        elif header['filetype'] == 3:
            from oct2py import octave
            import re
            octave.addpath(re.match('(.*)(plab)',__file__).group(0) + '/mlab/util')
            raw_data = octave.readpvpfile(filename)
            return raw_data

# Newer filetypes, fully implemented in python

        # NON-SPIKING ACTIVITY FILE
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
            return (data,header)

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
                    time = np.fromfile(stream,headerPattern_ext,1)['time'][0]
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
                    return (data,header)
                
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
                    time = np.fromfile(stream,np.float64,1)
                    numActive = np.fromfile(stream,np.uint32,1)
                    currentData = np.fromfile(stream,entryPattern,numActive)
                    data.append(DataFrame(time,np.array([currentData['index'],
                                                         currentData['activation']]).T))
                    if progressPeriod:
                        if not frame % progressPeriod and frame:
                            print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))
            return (data,header)
