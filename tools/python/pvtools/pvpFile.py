import pdb
import numpy as np
import scipy.sparse as sp
from pvtools.readpvpheader import readpvpheader,headerPattern,extendedHeaderPattern
import os

"""
An object for reading and writing pvp files. We open with a mode of 'r' (read), 'w' (overwrite), and 'a' (append)

read([start=0, stop=-1, skip=1, progress=0])
Returns a dictionary with 3 keys: values, time, header
Values can be:
   4D dense numpy array of size [numFrames, ny, nx, nf] for reading dense activity files
   6D dense numpy array of size [numFrames, numArbors, numKernels, ny, nx, nf] for reading weights
   2D csr_sparse array of size [numFrames, ny*nx*nf] for reading sparse activity files
Time is a 1D numpy array of size [numFrame] that correspond to the simulation time of the frame
Header is a dictionary of key value pairs corresponding to the header of the pvp file.
Here, numFrames dimension is generated with range(start, stop, skip)

write(data, [shape=None, useExistingHeader=False])
Writes to the open file
data must a dictionary with at least 2 keys: values, time
If useExistingHeader is set to true, data must also have header
Values and Time are the same as what read returns. If data is a csr sparse file, shape must be provided

For a detailed explanation of pvp files as well as what is contained in the header, please see the wiki on the github page
https://github.com/PetaVision/OpenPV/wiki/PetaVision-Output-(PVP)-file-specifications
"""

#A class for pvp file streaming for reading and writing
class pvpOpen(object):
    #Constructor for opening file
    def __init__(self, filename, mode):
        self.mode = mode
        self.filename = filename
        #Check mode argument
        if(mode == 'r'):
            #This should throw an error if the file doesn't exist
            self.pvpFile = open(filename, 'rb')
        #We open the file for read+write for reading the current number of frames
        elif(mode == 'w'):
            self.pvpFile = open(filename, 'wb+')
        elif(mode == 'a'):
            #File must exist in append mode
            if(not os.path.isfile(filename)):
                raise Exception("File " + filename + " does not exist")
            #We do r+ instead of a+ because we do intervening fseeks
            self.pvpFile = open(filename, 'rb+')
        else:
            raise Exception("Mode " + mode + " not recognized")

        #Read header if necessary
        if(mode == 'r' or mode == 'a'):
            self.header = readpvpheader(self.pvpFile)
            #Calculate num frames
            if(self.header['filetype'] == 5):
                filesize = os.path.getsize(filename)
                patchsizeoverall = self.header['nxp'] * self.header['nyp'] * self.header['nfp']
                recordsize = self.header['numpatches'] * (8+4*patchsizeoverall)
                framesize = recordsize * self.header['nbands'] + self.header['headersize']
                self.numFrames = filesize//framesize
            else:
                self.numFrames = self.header['nbands']
            #Check filetypes
            if(self.header['filetype'] == 1 or self.header['filetype'] == 3):
                raise Exception("File type " + str(self.header['filetype']) + " not implemented")
        else:
            self.header = None

        #If we are reading sparse pvp files, we build a frame lookup for where each frame starts
        if(mode == 'r' and (self.header['filetype'] == 2 or self.header['filetype'] == 6)):
            #File pointer should be past header at this point
            self.framePos = self.buildFrameLookup()

    def close(self):
        self.pvpFile.close()

    #Function to build a lookup for frames
    def buildFrameLookup(self):
        #Save file position
        savePos = self.pvpFile.tell()
        #Sparse activity file
        if(self.header['filetype'] == 6):
            entryPattern = np.dtype([('index', np.int32),
                                     ('activation', np.float32)])
        #Sprase spiking file
        else:
            entryPattern = np.dtype(np.uint32)

        #Allocate numpy array of type int64 to keep track of file positions
        outFramePos = np.zeros((self.numFrames)).astype(np.int64)

        print("Building frame lookup for sparse pvp file")
        for frame in range(self.numFrames):
            #if(frame % 100 == 0):
            #    print "Frame " + str(frame) + " out of " + str(self.numFrames)
            #Storing beginning of each frame
            outFramePos[frame] = self.pvpFile.tell()
            numActive = np.fromfile(self.pvpFile, np.uint32,3)[-1]
            self.pvpFile.seek(entryPattern.itemsize * numActive, os.SEEK_CUR)
        print("Done")

        #Restore file position
        self.pvpFile.seek(savePos, os.SEEK_SET)
        return outFramePos

    def read(self, start = 0, stop = -1, skip = 1, progress=0):
        if(self.mode != 'r'):
            raise Exception("File not opened for reading")

        if(stop == -1):
            stop = self.numFrames

        assert start >= 0
        assert stop >= 1
        assert progress >= 0
        assert skip >= 1
        assert start < stop

        dataTypeSwitch = {1: np.uint8,
                          2: np.int32,
                          3: np.float32,
                          4: np.int32}

        dataType = dataTypeSwitch[self.header['datatype']]

        frameRange = range(start, stop, skip)
        data = {}
        data["header"] = self.header
        #Spking activity file
        if self.header['filetype'] == 2:
            framesList = []
            idxList = []
            timeList = []

            for (frameNum, frame) in enumerate(frameRange):
                self.pvpFile.seek(self.framePos[frame], os.SEEK_SET)
                time = np.fromfile(self.pvpFile,np.float64,1)[0]
                timeList.append(time)

                numActive = np.fromfile(self.pvpFile,np.uint32,1)
                dataIdx = np.fromfile(self.pvpFile,np.uint32,numActive)

                idxList.extend(dataIdx)

                framesList.extend(np.ones((len(dataIdx)))*frameNum)

                if progress:
                    if frame % progress == 0:
                        print("File "+self.filename+": frame "+str(frame)+" of "+str(frameRange[-1]))

            #Make csrsparsematrix
            data["time"] = np.array(timeList)
            #Values for sparse matrix are all 1's
            data["values"] = sp.csr_matrix((np.ones((len(framesList))), (framesList, idxList)), shape=(len(frameRange), self.header["nx"]*self.header["ny"]*self.header["nf"]))

        # DENSE, NON-SPIKING ACTIVITY FILE
        elif self.header['filetype'] == 4:
            shape = (self.header['ny'], self.header['nx'], self.header['nf'])
            pattern = np.dtype([('time', np.float64),
                                ('values', dataType, shape)])
            frameSize = pattern.itemsize

            data["values"] = np.zeros((len(frameRange), self.header['ny'], self.header['nx'], self.header['nf']))
            data["time"] = np.zeros((len(frameRange)))

            for (frameNum, frame) in enumerate(frameRange):
                 self.pvpFile.seek(self.header['headersize'] + frame*frameSize, os.SEEK_SET)
                 currentData = np.fromfile(self.pvpFile, pattern, 1)
                 time = currentData['time'][0]
                 values = currentData['values'][0]
                 data["time"][frameNum] = time
                 data["values"][frameNum, :, :, :] = values

                 if progress:
                     if frameNum % progress == 0:
                         print("File "+self.filename+": frame "+str(frame)+" of "+str(frameRange[-1]))


        # KERNEL WEIGHT FILE
        elif self.header['filetype'] == 5:
            shape = (self.header['nyp'], self.header['nxp'], self.header['nfp'])
            patchsizeoverall = self.header['nxp'] * self.header['nyp'] * self.header['nfp']
            recordsize = self.header['numpatches'] * (8+4*patchsizeoverall)
            frameSize = recordsize * self.header['nbands'] + self.header['headersize']
            patchPattern = np.dtype([('nx', np.uint16),
                                     ('ny', np.uint16),
                                     ('offset', np.uint32),
                                     ('values', dataType, shape)])

            data["values"] = np.zeros((len(frameRange), self.header['nbands'], self.header['numpatches'], self.header['nyp'], self.header['nxp'], self.header['nfp']))
            data["time"] = np.zeros((len(frameRange)));

            for (frameNum, frame) in enumerate(frameRange):
                 self.pvpFile.seek(frame*frameSize, os.SEEK_SET)
                 time = np.fromfile(self.pvpFile,extendedHeaderPattern,1)['time'][0]
                 data["time"][frameNum] = time
                 for arbor in range(self.header['nbands']):
                     currentData = np.fromfile(self.pvpFile,
                                               patchPattern,
                                               self.header['numpatches'])['values']

                     data["values"][frameNum, arbor, :, :, :, :] = currentData

                 if progress:
                     if frameNum % progress == 0:
                         print("File "+self.filename+": frame "+str(frame)+" of "+str(frameRange[-1]))

        # SPARSE ACTIVITY FILE
        elif self.header['filetype'] == 6:
            entryPattern = np.dtype([('index', np.int32),
                                     ('activation', np.float32)])
            valuesList = []
            framesList = []
            idxList = []
            timeList = []

            for (frameNum, frame) in enumerate(frameRange):
                self.pvpFile.seek(self.framePos[frame], os.SEEK_SET)
                time = np.fromfile(self.pvpFile,np.float64,1)[0]
                timeList.append(time)
                numActive = np.fromfile(self.pvpFile,np.uint32,1).item()
                currentData = np.fromfile(self.pvpFile,entryPattern,numActive)
                dataIdx = currentData['index']
                dataValues = currentData['activation']
                idxList.extend(dataIdx)
                valuesList.extend(dataValues)
                framesList.extend(np.ones((len(dataIdx)))*frameNum)

                if progress:
                    if frameNum % progress == 0:
                        print("File "+self.filename+": frame "+str(frame)+" of "+str(frameRange[-1]))

            #Make csrsparsematrix
            data["time"] = np.array(timeList)
            data["values"] = sp.csr_matrix((valuesList, (framesList, idxList)), shape=(len(frameRange), self.header["nx"]*self.header["ny"]*self.header["nf"]))

        return data

    def checkData(self, data):
        #Check if dictionary
        if not isinstance(data, dict):
            raise ValueError("Input data structure must be a dictionary with the keys \"values\" and \"time\"")

        #Check for fields values and time
        if not 'values' in data.keys():
            raise ValueError("Input data structure missing \"values\" key");
        if not 'time' in data.keys():
            raise ValueError("Input data structure missing \"time\" key");

        values = data["values"]
        time = data["time"]

        #Make sure the 2 arrays are numpy arrays or sparse matrices
        if not sp.issparse(values) and not type(values).__module__ == np.__name__:
            raise ValueError("Values field must be either a sparse matrix or a numpy array")

        #If time is a list, convert to numpy array
        if type(time) == list:
            data["time"] = np.array(data["time"])
            time = data["time"]

        if not type(time).__module__ == np.__name__:
            raise ValueError("Time field must be either a numpy array or a list")

        #Check dimensions of values and time
        if sp.issparse(values):
            if not values.ndim == 2:
                raise ValueError("Sparse values must have 2 dimensions")
        else:
            if not values.ndim == 4 and not values.ndim == 6:
                raise ValueError("Dense values must have either 4 or 6 dimensions")

        #Check that sizes of values and time matches
        valuesShape = values.shape
        timeShape = time.shape
        if not valuesShape[0] == timeShape[0]:
            raise ValueError("Values must have the same number of frames as time (" + str(valuesShape[0]) + " vs " + str(timeShape[0]) + ")")

        #Values should be single floats, time should be double floats
        data["values"] = data["values"].astype(np.float32)
        data["time"] = data["time"].astype(np.float64)

        #Dense values must be c-contiguous
        if(not sp.issparse(data["values"]) and not data["values"].flags["C_CONTIGUOUS"]):
           data["values"] = data["values"].copy(order='C')

    def generateHeader(self, data, inShape):
        #data["values"] can be one of 3 shapes: dense 4d mat for activity, dense 6d mat for weights
        #scipy csr_sparse matrix for sparse activity

        header = {}

        values = data["values"]

        #If sparse matrix, write as sparse format
        if(sp.issparse(values)):
            if(inShape == None):
                raise ValueError("Sparse values must have shape input when generating header")
            if len(inShape) != 3:
                raise ValueError("Shape parameter must be a 3 tuple of (ny, nx, nf)")
            (ny, nx, nf) = inShape
            (numFrames, numFeat) = values.shape
            if(not numFeat == ny*nx*nf):
                raise ValueError("Shape provided does not match the data shape (" + str(ny) + "*" + str(nx) + "*" + str(nf) + " vs " + str(numFeat) + ")")

            header["headersize"] = np.uint32(80)
            header["numparams"]  = np.uint32(20)
            header["filetype"]   = np.uint32(6)
            header["nx"]         = np.uint32(nx)
            header["ny"]         = np.uint32(ny)
            header["nf"]         = np.uint32(nf)
            header["numrecords"] = np.uint32(1)
            header["recordsize"] = np.uint32(0) #Not used in sparse activity
            header["datasize"]   = np.uint32(8) #Int/float are 4 bytes each
            header["datatype"]   = np.uint32(4) #Type is location-value pair
            header["nxprocs"]    = np.uint32(1) #No longer used
            header["nyprocs"]    = np.uint32(1)
            header["nxExtended"] = np.uint32(nx)
            header["nyExtended"] = np.uint32(ny)
            header["kx0"]        = np.uint32(0)
            header["ky0"]        = np.uint32(0)
            header["nbatch"]     = np.uint32(1)
            header["nbands"]     = np.uint32(numFrames)
            header["time"]       = np.float64(data["time"][0])

        #If 4d dense matrix, write as dense format
        elif(values.ndim == 4):
            (numFrames, ny, nx, nf) = values.shape
            header["headersize"] = np.uint32(80)
            header["numparams"]  = np.uint32(20)
            header["filetype"]   = np.uint32(4)
            header["nx"]         = np.uint32(nx)
            header["ny"]         = np.uint32(ny)
            header["nf"]         = np.uint32(nf)
            header["numrecords"] = np.uint32(1)
            header["recordsize"] = np.uint32(nx*ny*nf) #Not used in sparse activity
            header["datasize"]   = np.uint32(4) #floats are 4 bytes
            header["datatype"]   = np.uint32(3) #Type is float
            header["nxprocs"]    = np.uint32(1) #No longer used
            header["nyprocs"]    = np.uint32(1)
            header["nxExtended"] = np.uint32(nx)
            header["nyExtended"] = np.uint32(ny)
            header["kx0"]        = np.uint32(0)
            header["ky0"]        = np.uint32(0)
            header["nbatch"]     = np.uint32(1)
            header["nbands"]     = np.uint32(numFrames)
            header["time"]       = np.float64(data["time"][0])

        #If 6d dense matrix, write as weights format
        elif(values.ndim == 6):
            (numFrames, numArbors, numKernels, nyp, nxp, nfp) = values.shape
            header["headersize"] = np.uint32(104)
            header["numparams"]  = np.uint32(26)
            header["filetype"]   = np.uint32(5)
            header["nx"]         = np.uint32(1) #size not used by weights
            header["ny"]         = np.uint32(1)
            header["nf"]         = np.uint32(numKernels) #Pre nf
            header["numrecords"] = np.uint32(numArbors)
            #Each data for arbor is preceded by nxp(2 bytes), ny (2 bytes) and offset (4 bytes)
            header["recordsize"] = np.uint32(0) #weight files do not use recordsize
            header["datasize"]   = np.uint32(4) #floats are 4 bytes
            header["datatype"]   = np.uint32(3) #float type
            header["nxprocs"]    = np.uint32(1)
            header["nyprocs"]    = np.uint32(1)
            header["nxExtended"] = np.uint32(1)
            header["nyExtended"] = np.uint32(1)
            header["kx0"]        = np.uint32(0)
            header["ky0"]        = np.uint32(0)
            header["nbatch"]     = np.uint32(1)
            header["nbands"]     = np.uint32(numArbors) #For weights, numArbors is stored in nbands, no field for numFrames
            #This field will be updated on write
            header["time"]       = np.float64(data["time"][0])
            #Weights have extended header
            header["nxp"]        = np.uint32(nxp)
            header["nyp"]        = np.uint32(nyp)
            header["nfp"]        = np.uint32(nfp)
            header["wMax"]       = np.uint32(1) #This field will be updated on write
            header["wMin"]       = np.uint32(1) #This field will be updated on write
            header["numpatches"] = np.uint32(numKernels)
        return header

    def checkHeaders(self, header1, header2):
        if(header1["filetype"] != header2["filetype"]):
            raise Exception("Header filetypes do not match")
        if(header1["filetype"] != 5):
            if(header1["nx"] != header2["nx"] or
               header1["ny"] != header2["ny"] or
               header1["nf"] != header2["nf"]):
                raise Exception("Header shapes do not match")
        else:
            if(header1["nxp"] != header2["nxp"] or
               header1["nyp"] != header2["nyp"] or
               header1["nfp"] != header2["nfp"] or
               header1["numpatches"] != header2["numpatches"] or
               header1["nbands"] != header2["nbands"] #nbands here is numarbors
               ):
                raise Exception("Header shapes do not match")

        #Sanity checks
        assert(header1["numparams"] == header2["numparams"])
        assert(header1["headersize"] == header2["headersize"])
        assert(header1["numrecords"] == header2["numrecords"])
        assert(header1["datatype"] == header2["datatype"])

    def writeHeader(self):
        # To get ordered list of header params
        if self.header['numparams'] == 26:
            hPattern = extendedHeaderPattern
        else:
            hPattern = headerPattern

        #Weight patches write header everytime, so ignore if filetype 5
        if(self.header["filetype"] != 5):
            #Write out header
            for headerEntry in hPattern:
                self.pvpFile.write(headerEntry[1](self.header[headerEntry[0]]))

    def updateHeader(self, data):
        #Only update if filetype is not 5
        if(self.header["filetype"] == 5):
            return
        numDataFrames = data["values"].shape[0]
        #Save current file pos
        savePos = self.pvpFile.tell()
        #Seek to nbands position
        self.pvpFile.seek(68, os.SEEK_SET)
        #Read current number of frames
        numFileFrames = np.fromfile(self.pvpFile, np.int32, 1)[0]
        #Seek back to nbands position and update num frames
        self.pvpFile.seek(68, os.SEEK_SET)
        self.pvpFile.write(np.int32(numDataFrames+numFileFrames))
        #Restore file position
        self.pvpFile.seek(savePos, os.SEEK_SET)

    def write(self, data, shape=None, useExistingHeader=False):
        if(self.mode != 'w' and self.mode != 'a'):
            raise Exception("File not opened for writing")

        #Seek to end of file
        self.pvpFile.seek(0, os.SEEK_END)

        #Check data structure
        self.checkData(data)

        if not 'header' in data.keys():
            if useExistingHeader:
                raise ValueError("Must specify a \"header\" field if using existing header")

        #Generate header if it doesn't exist
        if(not self.header):
            if(useExistingHeader):
                self.header = data["header"]
            else:
                self.header=self.generateHeader(data, shape)
            #Write out full header
            self.writeHeader()

        #Otherwise, check header fields
        else:
            if(useExistingHeader):
                self.checkHeaders(self.header, data["header"])
            else:
                self.checkHeaders(self.header, self.generateHeader(data, shape))
            #Change nbands for number of frames
            self.updateHeader(data)

        if self.header['numparams'] == 26:
            hPattern = extendedHeaderPattern
        else:
            hPattern = headerPattern

        #Write out files based on data
        if self.header['filetype'] == 2:
            raise Exception('Filetype 2 not yet supported for write pvp')

        elif self.header['filetype'] == 4:
            (numFrames, ny, nx, nf) = data["values"].shape
            for dataFrame in range(numFrames):
                self.pvpFile.write(data["time"][dataFrame])
                self.pvpFile.write(data["values"][dataFrame, :, :, :])

        elif self.header['filetype'] == 5:
            (numFrames, numArbors, numKernels, nyp, nxp, nfp) = data["values"].shape
            # Type 5's have a header in each frame
            #Make a copy of header dictionary to avoid changing
            #the header field
            tmpHeader = self.header.copy()
            for dataFrame in range(numFrames):
                #Set header fields that change from frame to frame
                tmpHeader["time"] = np.float64(data["time"][dataFrame])
                ##wMax and wMin are int32's, whereas the max and min might not be an int
                #tmpHeader["wMax"] = np.uint32(np.max(data["values"][dataFrame, :, :, :, :, :]))
                #tmpHeader["wMin"] = np.uint32(np.min(data["values"][dataFrame, :, :, :, :, :]))
                #We write headers here because we need a header per frame
                for headerEntry in hPattern:
                    self.pvpFile.write(headerEntry[1](tmpHeader[headerEntry[0]]))
                #Within each patch, we write out each nxp, nyp, and offset
                for dataArbor in range(numArbors):
                    for dataKernel in range(numKernels):
                        self.pvpFile.write(np.uint16(nxp))
                        self.pvpFile.write(np.uint16(nyp))
                        self.pvpFile.write(np.uint32(0)) #Offset is always 0 for kernels
                        self.pvpFile.write(data["values"][dataFrame, dataArbor, dataKernel, :, :, :])
        #Sparse values
        elif self.header['filetype'] == 6:
            (numFrames, numData) = data["values"].shape
            for dataFrame in range(numFrames):
                frameVals = data["values"].getrow(dataFrame)
                count = frameVals.nnz
                index = frameVals.indices
                value = frameVals.data
                #Write time first, followed by count, followed by values
                self.pvpFile.write(data["time"][dataFrame])
                self.pvpFile.write(np.uint32(count))
                npOut = np.zeros((count, 2)).astype(np.uint32)
                npOut[:, 0] = np.uint32(index)
                npOut[:, 1] = np.float32(value).view(np.uint32)
                self.pvpFile.write(npOut.flatten())


if __name__ == "__main__":
    #wFile = pvpFile("test/test.pvp", 'w')
    #rFile = pvpFile("test/test.pvp", 'r')
    #aFile = pvpFile("test/test.pvp", 'a')
    filename = "/home/slundquist/mountData/datasets/cifar/pvp/tmp.pvp"
    f = pvpOpen(filename, 'a')

    data = {}
    values = np.zeros((2, 32, 32, 3))
    values[1, 3, 3, 0] = .2312
    values[0, 5, 5, 0] = .5342
    data["values"] = values
    data["time"] = range(2)
    f.write(data)
    f.close()

    rf = pvpOpen(filename, 'r')
    outdata = rf.read(progress=100)
    pdb.set_trace()

