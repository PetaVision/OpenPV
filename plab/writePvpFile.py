import numpy as np

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
    #try:
    filestream.write(np.float64(idx)) #Write timestep
    #Writes in c format, where it's last dimention first, and goes backwards
    inmat.astype('float32').tofile(filestream)
    #except ValueError:
    #    time.sleep(5)
    #    if (attempts < 10):
    #        print "File IO failed. Attempt"
    #        writeData(filestream, inmat, idx, True, attempts+1)
    #    else:
    #        exit(1)
    #    #file write error, try again

#def writePvpFile(inmatlist, outfilename):
#   #Write binary
#   matfile = open(outfilename, 'wb')
#   #Write header
#   writeHeaderFile(inmatlist, matfile)
#   for time, inmat in enumerate(inmatlist):
#      print time, "out of", len(inmatlist)
#      writeData(inmat, time, matfile)
#   matfile.close()

