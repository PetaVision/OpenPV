from .readpvpheader import readpvpheader,extendedHeaderPattern
import scipy.sparse as sp
import numpy as np
import os
import pdb

"""
readpvpfile reads pvp files from the output of OpenPV
filename: Filename of the pvp file.
progressPeriod: How often to print out progress. 0 means do not print progress (default).
lastFrame: The last frame to read in the pvp file
startFrame: The first frame to read in the pvp file
skipFrames: Skips frames between start and last frame

Returns a dictionary with 3 keys: values, time, header
Values can be:
   4D dense numpy array of size [numFrames, ny, nx, nf] for reading dense activity files
   6D dense numpy array of size [numFrames, numArbors, numKernels, ny, nx, nf] for reading weights
   2D coo_sparse array of size [numFrames, ny*nx*nf] for reading sparse activity files
Time is a 1D numpy array of size [numFrame] that correspond to the simulation time of the frame
Header is a dictionary of key value pairs corresponding to the header of the pvp file.

For a detailed explanation of pvp files as well as what is contained in the header, please see the wiki on the github page
https://github.com/PetaVision/OpenPV/wiki/PetaVision-Output-(PVP)-file-specifications
"""


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

        #Initialize data structure
        data = {}
        data["header"] = header

# Deprecated filetype not implemented in python (requires oct2py):

        # PVP FILE (deprecated)
        if header['filetype'] == 1:
            from oct2py import octave
            import re
            octave.addpath(re.match('(.*)(python)',__file__).group(0) + '/mlab/util')
            raw_data = octave.readpvpfile(filename)
            data["values"] = raw_data
            return data
# Supported older filetypes, not fully tested in Python 2

        # SPIKING ACTIVITY FILE
        #TODO filetype 6 has fixed the frame indexing. It's probably broken here.
        if header['filetype'] == 2:
            lastFrame = min(lastFrame, header['nbands'])

            framesList = []
            idxList = []
            timeList = []

            frameNum = 0
            for frame in range(lastFrame):
                if frame < startFrame or (frame % skipFrames):
                    numActive = np.fromfile(stream,np.uint32,3)[-1]
                    stream.seek(np.dtype(np.uint32).itemsize * numActive,
                                os.SEEK_CUR)
                    continue
                else:
                    time = np.fromfile(stream,np.float64,1)[0]
                    timeList.append(time)

                    numActive = np.fromfile(stream,np.uint32,1)
                    dataIdx = np.fromfile(stream,np.uint32,numActive)

                    idxList.extend(dataIdx)

                    framesList.extend(np.ones((len(dataIdx)))*frameNum)
                    frameNum += 1

                    if progressPeriod:
                        if not frame % progressPeriod and frame:
                            print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))

            #Make coosparsematrix
            data["time"] = np.array(timeList)
            #Values for sparse matrix are all 1's
            data["values"] = sp.coo_matrix((np.ones((len(framesList))), (framesList, idxList)), shape=(frameNum, header["nx"]*header["ny"]*header["nf"]))

            return data

        # NON-KERNEL WEIGHT FILE
        #TODO
        elif header['filetype'] == 3:
        #    fileSize = os.path.getsize(filename)
        #    frameSize = header['recordsize'] * header['nbands'] + header['headersize']
        #    lastFrame = min(lastFrame,fileSize/frameSize)
        #    shape = (header['nxp'], header['nyp'], header['nfp'])
        #    patchPattern = np.dtype([('nx', np.uint16),
        #                             ('ny', np.uint16),
        #                             ('offset', np.uint32),
        #                             ('values', dataType, shape)])
        #    stream.seek(0)

        #    #Initialize values and time arrays
        #    frameList = range(startFrame, lastFrame, skipFrames)
        #    data["values"] = np.zeros((len(frameList), header['nbands'], header['numpatches']
        #    for frame in frameList:
        #         stream.seek(frame*frameSize)
        #         time = np.fromfile(stream,extendedHeaderPattern,1)['time'][0]


        #         data.append(DataFrame(time,[]))
        #         for arbor in range(header['nbands']):
        #             currentData = np.fromfile(stream,
        #                                       patchPattern,
        #                                       header['numpatches'])['values']
        #             data[frame].values.append(np.squeeze(np.transpose(currentData,
        #                                                               [2,1,3,0])))
        #             if progressPeriod:
        #                 if not frame % progressPeriod and frame:
        #                     print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))
        #    return PV_Object(data,header)
           assert(0)

        # DENSE, NON-SPIKING ACTIVITY FILE
        #Tested as of 2/15/16
        elif header['filetype'] == 4:
            lastFrame = min(lastFrame, header['nbands'])
            shape = (header['ny'], header['nx'], header['nf'])
            pattern = np.dtype([('time', np.float64),
                                ('values', dataType, shape)])
            frameSize = pattern.itemsize


            frameRange = range(startFrame, lastFrame, skipFrames)
            data["values"] = np.zeros((len(frameRange), header['ny'], header['nx'], header['nf']))
            data["time"] = np.zeros((len(frameRange)))

            for (i, frame) in enumerate(frameRange):
                 stream.seek(header['headersize'] + frame*frameSize)
                 currentData = np.fromfile(stream, pattern, 1)
                 time = currentData['time'][0]
                 values = currentData['values'][0]
                 data["time"][i] = time
                 data["values"][i, :, :, :] = values

                 if progressPeriod:
                     if not i % progressPeriod and frame:
                         print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))

            return data

        # KERNEL WEIGHT FILE #TODO test
        elif header['filetype'] == 5:
            fileSize = os.path.getsize(filename)
            frameSize = header['recordsize'] * header['nbands'] + header['headersize']
            lastFrame = min(lastFrame,fileSize/frameSize)
            shape = (header['nxp'], header['nyp'], header['nfp'])
            patchPattern = np.dtype([('nx', np.uint16),
                                     ('ny', np.uint16),
                                     ('offset', np.uint32),
                                     ('values', dataType, shape)])
            stream.seek(0)

            frameRange = range(startFrame, lastFrame, skipFrames)
            data["values"] = np.zeros((len(frameRange), header['nbands'], header['numpatches'], header['nyp'], header['nxp'], header['nfp']))
            data["time"] = np.zeros((len(frameRange)));

            for (i, frame) in enumerate(frameRange):
                 stream.seek(frame*frameSize)
                 time = np.fromfile(stream,extendedHeaderPattern,1)['time'][0]
                 data["time"][i] = time
                 for arbor in range(header['nbands']):
                     currentData = np.fromfile(stream,
                                               patchPattern,
                                               header['numpatches'])['values']

                     data["values"][i, arbor, :, :, :, :] = currentData

                 if progressPeriod:
                     if not i % progressPeriod and frame:
                         print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))
            return data

        # SPARSE ACTIVITY FILE
        #TODO
        elif header['filetype'] == 6:
            lastFrame = min(lastFrame, header['nbands'])
            entryPattern = np.dtype([('index', np.int32),
                                     ('activation', np.float32)])
            valuesList = []
            framesList = []
            idxList = []
            timeList = []

            frameNum = 0
            for frame in range(lastFrame):
                if frame in range(startFrame, lastFrame, skipFrames):
                    time = np.fromfile(stream,np.float64,1)[0]
                    timeList.append(time)
                    numActive = np.fromfile(stream,np.uint32,1)
                    currentData = np.fromfile(stream,entryPattern,numActive)
                    dataIdx = currentData['index']
                    dataValues = currentData['activation']
                    idxList.extend(dataIdx)
                    valuesList.extend(dataValues)
                    framesList.extend(np.ones((len(dataIdx)))*frameNum)
                    frameNum += 1
                else:
                    numActive = np.fromfile(stream,np.uint32,3)[-1]
                    stream.seek(entryPattern.itemsize * numActive, os.SEEK_CUR)
                    continue

                if progressPeriod:
                    if frame in range(startFrame, lastFrame, progressPeriod):
                        print("File "+filename+": frame "+str(frame)+" of "+str(lastFrame))

            #Make coosparsematrix
            data["time"] = np.array(timeList)
            data["values"] = sp.coo_matrix((valuesList, (framesList, idxList)), shape=(frameNum, header["nx"]*header["ny"]*header["nf"]))

            return data
