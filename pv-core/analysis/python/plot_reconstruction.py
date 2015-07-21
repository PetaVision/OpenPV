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


if len(sys.argv) < 7:
   print "usage: plot_avg_activity activity-filename [end_time step_time begin_time] w4, w5, activity-test"
   sys.exit()

extended = False
a1 = rs.PVReadSparse(sys.argv[1], extended)
end = int(sys.argv[2])
step = int(sys.argv[3])
begin = int(sys.argv[4])
w = rw.PVReadWeights(sys.argv[5])
wOff = rw.PVReadWeights(sys.argv[6])
atest = rs.PVReadSparse(sys.argv[7], extended)


coord = 1
endtest = 2000
steptest = 1999
begintest = 0
anX = 32
anY = 32
nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
space = 1
slPre = -math.log(4.0,2)
slPost = -math.log(1.0,2)


pa = []
grayp = []
graypo = []
grau = []
for i in range((32*32)):
   grayp.append(0.5)
   graypo.append(0.5)
   grau.append(0.5)
 

grayp = np.reshape(grayp, (32, 32))
grau =  np.reshape(grau, (32, 32))
graypo = np.reshape(graypo, (32, 32))



for endtest in range(begintest+steptest, endtest, steptest):
   Atest = atest.avg_activity(begintest, endtest)
   lenofo = len(Atest)
   for i in range(lenofo):
      for j in range(lenofo):
         pa = np.append(pa, Atest[i,j])  

amax = np.max(pa)
nxp2 = 1.0
count = 0

for end in range(begin+step, end, step):
   A1 = a1.avg_activity(begin, end)
   lenofo = len(A1)
   lenofb = lenofo * lenofo
   #print "a1 = ", np.shape(A1)
   #print "a2 = ", np.shape(A2)
   for j in range(lenofo):
      for i in range(lenofo):
         ix = conv.zPatchHead(i, nxp2, slPre, slPost)
         jy = conv.zPatchHead(j, nxp2, slPre, slPost)
         p = w.next_patch()
         pOff = wOff.next_patch()

         grayp[ix, jy] = grayp[ix, jy] + (np.sum(((A1[i, j]/amax)*p)) / (16*2))
         graypo[ix, jy] = graypo[ix, jy] - (np.sum(((A1[i, j]/amax)*pOff)) / (16*2))


   fig = plt.figure()
   ax = fig.add_subplot(211)

   ax.set_xlabel('Kx')
   ax.set_ylabel('Ky')
   ax.set_title('On Reconstruction')

   ax.imshow(grayp, cmap=cm.binary, interpolation='nearest', vmin=0, vmax=1)

   ax = fig.add_subplot(212)

   ax.set_xlabel('Kx')
   ax.set_ylabel('Ky')
   ax.set_title('Off Reconstruction')

   ax.imshow(graypo, cmap=cm.binary, interpolation='nearest', vmin=0, vmax=1)


   plt.show()

#end fig loop
