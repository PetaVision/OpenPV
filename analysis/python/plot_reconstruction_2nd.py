"""
Plot a reconstruction of the retina image from the l1 activity and patches
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadSparse as rs
import PVConversions as conv
import PVReadWeights as rw
import math

def format_coord(x, y):
   col = int(x+0.5)
   row = int(y+0.5)
   if coord == 3:
      check = ((x - 0.5) % 16)
      if check < 4:
         x2 = ((x - 0.5) % 16) - 7 + (x / 16.0)
         y2 = ((y - 0.5) % 16) - 7 + (y / 16.0) 
      elif check < 10:
         x2 = ((x - 0.5) % 16) - 7.5 + (x / 16.0)
         y2 = ((y - 0.5) % 16) - 7.5 + (y / 16.0) 
      else:
         x2 = ((x - 0.5) % 16) - 8 + (x / 16.0)
         y2 = ((y - 0.5) % 16) - 8 + (y / 16.0) 
      x = (x / 16.0)
      y = (y / 16.0)
      

      if col>=0 and col<numcols and row>=0 and row<numrows:
         z = P[row,col]
         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
      else:
         return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))      

   if coord == 1:
      x2 = (x / 20.0)
      y2 = (y / 20.0)
      x = (x / 5.0)
      y = (y / 5.0)
      if col>=0 and col<numcols and row>=0 and row<numrows:
         z = P[row,col]
         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
      else:
         return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))


if len(sys.argv) < 6:
   print "usage: plot_avg_activity activity-filename [end_time step_time begin_time] w4-filename w5-filename"
   sys.exit()

extended = False
a1 = rs.PVReadSparse(sys.argv[1], extended)
end = int(sys.argv[2])
step = int(sys.argv[3])
begin = int(sys.argv[4])
w = rw.PVReadWeights(sys.argv[5])
wOff = rw.PVReadWeights(sys.argv[6])
atest = a1

coord = 1
endtest = end
steptest = step
begintest = begin
nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
space = 1
slPre = -math.log(4.0,2)
slPost = -math.log(1.0,2)

nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.

pa = []
grayp = []

for i in range(16):
   grayp.append(0.5)


for endtest in range(begintest+steptest, endtest, steptest):
   Atest = atest.avg_activity(begintest, endtest)
   lenofo = len(Atest)
   for i in range(lenofo):
      for j in range(lenofo):
         pa = np.append(pa, Atest[i,j])  
amax = np.max(pa)




for end in range(begin+step, end, step):
   A1 = a1.avg_activity(begin, end)
   AF = A1
   lenofo = len(A1)
   lenofb = lenofo * lenofo
   for j in range(lenofo):
      for i in range(lenofo):
         ix = conv.zPatchHead(j, nxp, slPre, slPost)
         jy = conv.zPatchHead(i, nxp, slPre, slPost)
         #print ix
         #print jy
         #print
         p = w.next_patch()
         pOff = wOff.next_patch()

         thep = grayp + ((A1[ix, jy]/(amax * 2))*p) 
         thep = thep - ((A1[ix, jy]/(amax * 2))*pOff)

         thep = np.reshape(thep, (nxp, nyp))
         numrows, numcols = thep.shape
         k = j * nx + i
         x = space + (space + nxp) * (k % nx)
         y = space + (space + nyp) * (k / ny)

         im[y:y+nyp, x:x+nxp] = thep

   fig = plt.figure()
   ax = fig.add_subplot(111)

   ax.set_xlabel('Kx GLOBAL')
   ax.set_ylabel('Ky GLOBAL')
   ax.set_title('Weight Patches')
   ax.format_coord = format_coord

   ax.imshow(im, cmap=cm.binary, interpolation='nearest', vmin=0, vmax=1)

   plt.show()

#end fig loop
