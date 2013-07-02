import numpy as np
outfile = "multiframe.pvp"
params = np.int32([
   80, #headersize
   20, #numparams
   4,  #filetype
   10, #nx
   10, #ny
   1,  #nf
   1,  #numrecords
   100,#recordsize
   4,  #datasize
   3,  #datatype, float, check this
   1,  #nxprocs
   1,  #nyprocs
   10, #nxGlobal
   10, #nyGlobal
   0,  #kx0
   0,  #ky0
   0,  #nb, doesn't matter
   3   #nbands
])

time = params[17] #nbands
recordsize = params[7]

f = open(outfile, 'wb')
for param in params:
   f.write(param)

f.write(np.float64(3)) #Header timestamp

for itime in range(time):
   f.write(np.float64(itime))
   for iR in range(recordsize):
      f.write(np.float32(itime))

f.close()
