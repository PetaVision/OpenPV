import numpy as np
import pdb
import os, sys

lib_path = os.path.abspath("/home/sheng/workspace/OpenPV/pv-core/python/")
sys.path.append(lib_path)
from pvtools import *


#inmat is a 3 dimentional numpy array that is (Y, X, Z), where Z is depth info
#Write header file
def writeHeaderFile(filestream, shape, numFrames):
   (ysize, xsize, zsize) = shape
   params = np.int32([
      80, #headersize
      20, #numparams
      4,  #filetype, non-spiking activity layer
      xsize, #nx
      ysize, #ny
      zsize,  #nf
      1,  #numrecords always 1
      xsize*ysize*zsize,#recordsize
      4,  #datasize
      3,  #datatype, float
      1,  #nxprocs
      1,  #nyprocs
      xsize, #nxGlobal
      ysize, #nyGlobal
      0,  #kx0 not used
      0,  #ky0 not used
      0,  #nb not used in reading pvp files
      numFrames   #nbands
   ])
   for param in params:
      filestream.write(param)
   filestream.write(np.float64(0)) #Header timestamp, not used?

#Inmat must be y by x by z
def writeData(filestream, inmat, idx):
   filestream.write(np.float64(idx)) #Write timestep
   #Writes in c format, where it's last dimention first, and goes backwards
   inmat.astype('float32').tofile(filestream)


#Usage example
if __name__ == "__main__":
    #Set output filename
    outputFileName = "test.pvp"

    outX = 4
    outY = 5
    outF = 6

    #How many frames we will end up writing out
    numFrames = 1

    #Python ordering goes from slowest to fastest
    #So here, we match dimensions to match what PV is expecting
    #Make random matrix
    outmat = np.random.rand(outY, outX, outF)

    #Open output file for writing in binary
    outMatFile = open(outputFileName, 'wb')

    #Write header out
    writeHeaderFile(outMatFile, (outY, outX, outF), numFrames)

    #Write data out
    #Subsequent calls to writeData will append outMatFile with further frames
    frameIdx = 0
    writeData(outMatFile, outmat, frameIdx)

    outMatFile.close()



#def writePvpFile(inmatlist, outfilename):
#   #Write binary
#   matfile = open(outfilename, 'wb')
#   #Write header
#   writeHeaderFile(inmatlist, matfile)
#   for time, inmat in enumerate(inmatlist):
#      print time, "out of", len(inmatlist)
#      writeData(inmat, time, matfile)
#   matfile.close()

