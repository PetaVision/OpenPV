#########################################
##  pvAnalysis.py
##  Written by Dylan Paiton, Sheng Lundquist
##  Nov 17, 2014
##  
## Mimics readpvpfile.m analysis - this scrpt
## will hold a suite of tools for analyzing 
## PetaVision output files.
##
#########################################

import scipy.sparse as sparse
import numpy as np
import struct, os, sys
import pdb

def read_header_file(fileStream, pos=None):
    fileStream.seek(0)
    params = struct.unpack("iiiiiiiiiiiiiiiiiid", fileStream.read(80))
    header = {}
    header["headersize"] = params[0]
    header["numparams"]  = params[1]
    header["filetype"]   = params[2] # 3 is PVP_WGT_FILE_TYPE
                                     # 4 is PVP_NONSPIKING_ACT_FILE_TYPE
    header["nx"]         = params[3]
    header["ny"]         = params[4]
    header["nf"]         = params[5]
    header["numrecords"] = params[6]
    header["recordsize"] = params[7]
    header["datasize"]   = params[8] 
    header["datatype"]   = params[9]  # 3 is float,float,float,...   (dense file)
                                      # 4 is int,float,int,float,... (sparse file)
    header["nxprocs"]    = params[10]
    header["nyprocs"]    = params[11]
    header["nxGlobal"]   = params[12]
    header["nyGlobal"]   = params[13]
    header["kx0"]        = params[14]
    header["ky0"]        = params[15]
    header["nb"]         = params[16]
    header["nbands"]     = params[17]

    # If hdr.numparams is bigger than 20, there is a field 'additional'
    # containing an vector of hdr.numparams-20 elements.
    if header["numparams"] > 20:
        header["additional"] = []
        numAddParams = header["numparams"] - 20
        for i in range(numAddParams):
            header["additional"].append(struct.unpack("i", fileStream.read(4)))

    return header


def read_dense_data(fileStream, dense_shape, numNeurons):
    #Read timestep
    timestamp = fileStream.read(8)
    if len(timestamp) != 8:
        #EOF
        return (-1, None)
    try:
        idx = struct.unpack("d", timestamp)
        outmat = np.fromfile(fileStream, np.float32, numNeurons).reshape(dense_shape)
        return (idx, outmat)
    except:
        return (-1, None)


def read_sparse_data(fileStream,dense_shape):
    #Function assumes fileStream pointer is in the correct place - so that it can be run in a loop

    timeStamp = fileStream.read(8) # Should be a float64, if not then EOF

    if len(timeStamp) != 8: # EOF
        return (-1,None)

    timeStamp = struct.unpack("d", timeStamp)
    try:
        numActive = np.fromfile(fileStream,np.int32,1) 

        if numActive > 0:
            lin_idx   = np.zeros(numActive)
            vals      = np.zeros(numActive)

            #TODO: Speed up by reading in the full string initially
            # File alternates between int32 (index of active cell) and float32 (activity of cell)
            for i in range(numActive):
                lin_idx[i] = np.fromfile(fileStream,np.int32,1)
                vals[i]    = np.fromfile(fileStream,np.float32,1)
                #(idf[i],idx[i],idy[i]) = np.unravel_index(lin_idx[i],shape) # Linear indexing to subscripts

                # Compressed Sparse Column Matrix has efficient column slicing
                ij_mat = (np.zeros(numActive),lin_idx)
                sparseMat = sparse.csc_matrix((vals,ij_mat),shape=(1,dense_shape)) # 1 row, nf*ny*nx columns

        else:
            sparseMat = sparse.csc_matrix(np.zeros(1,dense_shape))
            return (timeStamp, sparseMat)

    except:
        return (timeStamp, None) 


def get_frame_info(hdr,fileStream):
    fileSize = os.path.getsize(fileStream.name)
    if hdr["filetype"] == 1:
        frameSize = hdr["recordsize"]*hdr["numrecords"]
        numFrames = (fileSize - hdr["headersize"]) / frameSize
    elif hdr["filetype"] == 2:
        frameSize = -1 # frameSize is a variable
        numFrames = hdr["nbands"]
    elif hdr["filetype"] == 3:
        frameSize = hdr["recordsize"] * hdr["numrecords"] + hdr["headersize"]
        numFrames = fileSize / frameSize
    elif hdr["filetype"] == 4:
        nxprocs   = hdr["nxGlobal"] / hdr["nx"]
        nyprocs   = hdr["nyGlobal"] / hdr["ny"]
        frameSize = hdr["recordsize"] * hdr["datasize"] * nxprocs * nyprocs+8
        numFrames = hdr["nbands"]
    elif hdr["filetype"] == 5:
        frameSize = hdr["recordsize"] * hdr["nbands"] + hdr["headersize"]
        numFrames = fileSize / frameSize
    elif hdr["filetype"] == 6:
        frameSize = -1 # frameSize is a variable
        numFrames = hdr["nbands"]
    return (frameSize,numFrames)


def get_pvp_data(fileStream,progressPeriod=0,lastFrame=-1,startFrame=0,skipFrames=1):
    # Usage: (data,hdr) = get_pvp_data(filename,progressperiod,lastFrame,startFrame,skipFrames)
    #
    # ===INPUTS===
    #  filename is a pvp file (any type)
    #
    #  progressPeriod makes a message print every progressPeriod frames.
    #
    #  lastFrame is the index of the last frame to read.  Default is all frames.
    #
    #  startFrame is the starting frame.
    #
    #  skipFrames specifies how many frames should be skipped between each iteration
    #
    # ===OUTPUTS===
    #  outStruct is a structure containing the data and time stamps
    #    outStruct["values"] returns a list containing dense numpy matrices
    #      In general, data has one matrix for each time step written.
    #      For activities, values is an nf-by-nx-by-ny array.
    #      Weights are not supported yet, but they will contain nxp-by-nyp-by-nfp arrays
    #    outStruct["time"] return a list of the timeStamps
    #
    #  hdr is a struct containing the information in the file's header
   
    hdr = read_header_file(fileStream)

    (frameSize, numFrames) = get_frame_info(hdr,fileStream)

    if lastFrame != -1:
       numFrames = lastFrame
   
    loopLen   = len(range(startFrame,numFrames,skipFrames))
 
    data       = loopLen * [None] # This pre-allocates the list
    timeStamps = loopLen * [None]
    if hdr["filetype"] == 3: #PVP_WGT_FILE_TYPE
        return (None, None)
       
    elif hdr["filetype"] == 4: #PVP_NONSPIKING_ACT_FILE
        if hdr["datatype"] == 3: #PV_FLOAT_TYPE
            shape = (hdr["nf"],hdr["nx"],hdr["ny"])
            numNeurons = shape[0]*shape[1]*shape[2]
            for i in range(startFrame,numFrames,skipFrames):
                if progressPeriod != 0:
                    if i%progressPeriod == 0:
                        sys.stdout.write(" Progress: %d/%d%s"%(i,loopLen,"\r"))
                        sys.stdout.flush();
                (timeStamps[i],data[i]) = read_dense_data(fileStream, shape, numNeurons)
                assert timeStamps[i] != -1
                #TODO: FutureWarning: comparison to `None` will result in an elementwise object comparison in the future.
                assert data[i] != None 
    elif hdr["filetype"] == 5: #PVP_KERNEL_FILE_TYPE
        # [For every frame]
        #   Header[(1, 80 bytes)]
        #   Extra Header[(3, int32), (2, float32), (1, uint32)]
        #   [For every arbor]
        #       [For every proc]
        #           [For every patch]
        #               shrunkenPatchNxNyOffset[(2, uint16), (1, uint32)]
        #               data[(nxp * nyp * nfp, dataType)]

        hdr["nxp"]        = hdr["additional"][0][0]
        hdr["nyp"]        = hdr["additional"][1][0]
        hdr["nfp"]        = hdr["additional"][2][0]
        hdr["wMin"]       = hdr["additional"][3][0]
        hdr["wMax"]       = hdr["additional"][4][0]
        hdr["numPatches"] = hdr["additional"][5][0]

        if hdr["datatype"] == 3: #PV_FLOAT_TYPE (precision is float32)
            for i in range(startFrame,numFrames,skipFrames):
                fileStream.seek(hdr["headersize"]) #Each frame has its own header, we expect the data to repeat
                for arbor in range(hdr["nbands"]):
                    # TODO: Convert arbor index into subscript?
                    for patch in range(hdr["numPatches"]):
                        # TODO: Handle shrunken patch info
                        fileStream.seek(16,1) # Move forward 16 bytes, for shrunkenPatch info
                        bytes_to_read = 8*hdr["nxp"]*hdr["nyp"]*hdr["nfp"]
                        tmp_dat = fileStream.read(bytes_to_read)
                        # TODO: Now what..?
                        pdb.set_trace()

    elif hdr["filetype"] == 6: #PVP_ACT_SPARSEVALUES_FILE_TYPE
       if hdr["datatype"] == 4: #PV_SPARSEVALUES_TYPE
           for i in range(startFrame,numFrames,skipFrames):
               if progressPeriod != 0:
                   if i%progressPeriod == 0:
                       sys.stdout.write(" Progress: %d/%d%s"%(i,loopLen,"\r"))
                       sys.stdout.flush();
               (timeStamps[i],sparseMat) = read_sparse_data(fileStream,hdr["nf"]*hdr["nx"]*hdr["ny"])
 
               assert timeStamps[i] != -1
               assert sparseMat != None 
 
               data[i] = np.ravel(sparseMat.todense()).reshape(hdr["ny"],hdr["nx"],hdr["nf"])
 
    outStruct = {}
    outStruct["time"]   = timeStamps
    outStruct["values"] = data
 
    sys.stdout.write(" Progress: %d/%d%s"%(numFrames,loopLen,"\r"))
    sys.stdout.write("%s Done.%s"%("\n","\n"))
    sys.stdout.flush();
 
    return (outStruct,hdr)
