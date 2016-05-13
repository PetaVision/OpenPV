import numpy as np
import scipy.sparse as sp
import pdb
from readpvpheader import headerPattern, extendedHeaderPattern

def checkData(data):
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


def generateHeader(data, inShape):
    #data["values"] can be one of 3 shapes: dense 4d mat for activity, dense 6d mat for weights
    #scipy coo_sparse matrix for sparse activity

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
        header["nxGlobal"]   = np.uint32(nx)
        header["nyGlobal"]   = np.uint32(ny)
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
        header["nxGlobal"]   = np.uint32(nx)
        header["nyGlobal"]   = np.uint32(ny)
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
        header["recordsize"] = np.uint32(numKernels * (8+4*nxp*nyp*nfp))
        header["datasize"]   = np.uint32(4) #floats are 4 bytes
        header["datatype"]   = np.uint32(3) #float type
        header["nxprocs"]    = np.uint32(1)
        header["nyprocs"]    = np.uint32(1)
        header["nxGlobal"]   = np.uint32(1)
        header["nyGlobal"]   = np.uint32(1)
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

def writepvpfile(filename, data, shape=None, useExistingHeader=False):

    #Check data structure
    checkData(data)


    if not 'header' in data.keys():
        if useExistingHeader:
            raise ValueError("Must specify a \"header\" field if using existing header")

    #Data can either have a header field or not
    #Generate header if no header field
    if not useExistingHeader:
        #If it doesn't exist, generate header
        data["header"] = generateHeader(data, shape)

    # To get ordered list of header params
    if data["header"]['numparams'] == 26:
        hPattern = extendedHeaderPattern
    else:
        hPattern = headerPattern

    with open(filename, 'wb') as stream:
        if data["header"]['filetype'] == 1:
            print('Filetype 1 not yet supported for write pvp')
        elif data["header"]['filetype'] == 2:
            print('Filetype 2 not yet supported for write pvp')
        elif data["header"]['filetype'] == 3:
            print('Filetype 3 not yet supported for write pvp')

        elif data["header"]['filetype'] == 4:
            (numFrames, ny, nx, nf) = data["values"].shape
            #Write out header
            for headerEntry in hPattern:
                stream.write(headerEntry[1](data["header"][headerEntry[0]]))
            for dataFrame in range(numFrames):
                stream.write(data["time"][dataFrame])
                stream.write(data["values"][dataFrame, :, :, :])

        elif data["header"]['filetype'] == 5:
            (numFrames, numArbors, numKernels, nyp, nxp, nfp) = data["values"].shape
            # Type 5's have a header in each frame
            #Make a copy of header dictionary to avoid changing
            #the header field
            tmpHeader = data["header"].copy()
            for dataFrame in range(numFrames):
                #Set header fields that change from frame to frame
                tmpHeader["time"] = np.float64(data["time"][dataFrame])
                ##wMax and wMin are int32's, whereas the max and min might not be an int
                #tmpHeader["wMax"] = np.uint32(np.max(data["values"][dataFrame, :, :, :, :, :]))
                #tmpHeader["wMin"] = np.uint32(np.min(data["values"][dataFrame, :, :, :, :, :]))
                for headerEntry in hPattern:
                    stream.write(headerEntry[1](tmpHeader[headerEntry[0]]))
                #Within each patch, we write out each nxp, nyp, and offset
                for dataArbor in range(numArbors):
                    for dataKernel in range(numKernels):
                        stream.write(np.uint16(nxp))
                        stream.write(np.uint16(nyp))
                        stream.write(np.uint32(0)) #Offset is always 0 for kernels
                        stream.write(data["values"][dataFrame, dataArbor, dataKernel, :, :, :])

        #Sparse values
        elif data["header"]['filetype'] == 6:
            (numFrames, numData) = data["values"].shape
            # Copied from filetype 4
            for headerEntry in hPattern:
                stream.write(headerEntry[1](data["header"][headerEntry[0]]))

            for dataFrame in range(numFrames):
                frameVals = data["values"].getrow(dataFrame)
                count = frameVals.nnz
                index = frameVals.indices
                value = frameVals.data
                #Write time first, followed by count, followed by values
                stream.write(data["time"][dataFrame])
                stream.write(np.uint32(count))
                for i in range(count):
                    stream.write(np.uint32(index[i]))
                    stream.write(np.float32(value[i]))

if __name__ == "__main__":
    data = {}
    values = np.ones((2, 10))
    data["values"] = sp.coo_matrix(values)
    data["time"] = range(2)
    writepvpfile("test.pvp", data, shape=(2, 5, 1))


