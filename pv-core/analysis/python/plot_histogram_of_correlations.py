"""
Plots the Histogram
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw
import PVConversions as conv
import scipy.cluster.vq as sp
import math

if len(sys.argv) < 2:
   print "usage: time_stability filename"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])


space = 1

d = np.zeros((4,4))

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = 10
marginstart = margin
marginend = nx - margin
acount = 0
patchposition = []
supereasytest = 1


# create feature list for comparing weights from on and off cells
f = np.zeros(w.patchSize)
f2 = np.zeros(w.patchSize)
fe1 = []
fe2 = []
fe3 = []
fe4 = []
fe5 = []
fe6 = []
fe7 = []
fe8 = []
fcomp = []

f = w.normalize(f)
f2 = w.normalize(f2)


# vertical lines from right side
f = np.zeros([w.nxp, w.nyp]) # first line
f[:,0] = 1
fe1.append(f)

f = np.zeros([w.nxp, w.nyp]) # second line
f[:,1] = 1
fe2.append(f)

f2 = np.zeros([w.nxp, w.nyp]) # third line
f2[:,2] = 1
fe3.append(f2)

f = np.zeros([w.nxp, w.nyp])
f[:,3] = 1
fe4.append(f)

#horizontal lines from the top
f = np.zeros([w.nxp, w.nyp])
f[0,:] = 1
fe5.append(f)

f = np.zeros([w.nxp, w.nyp])
f[1,:] = 1
fe6.append(f)

f = np.zeros([w.nxp, w.nyp])
f[2,:] = 1
fe7.append(f)

f = np.zeros([w.nxp, w.nyp])
f[3,:] = 1
fe8.append(f)

#print "f8", fe8
#print "f7", fe7
#print "f6", fe6
#print "f5", fe5
#print "f4", fe4
#print "f3", fe3
#print "f2", fe2
#print "f1", fe1









def whatFeature(k):
   result = []
   fcomp = []
   k = np.reshape(k,(nxp,nyp))

   f1 = k * fe1
   f1 = np.sum(f1)
   fcomp.append(f1)
   #print f1

   f2 = k * fe2
   f2 = np.sum(f2)
   #print f2
   fcomp.append(f2)

   f3 = k * fe3
   f3 = np.sum(f3)
   #print f3
   fcomp.append(f3)

   f4 = k * fe4
   f4 = np.sum(f4)
   #print f4
   fcomp.append(f4)

   f5 = k * fe5
   f5 = np.sum(f5)
   #print f5
   fcomp.append(f5)

   f6 = k * fe6
   f6 = np.sum(f6)
   #print f6
   fcomp.append(f6)

   f7 = k * fe7
   f7 = np.sum(f7)
   #print f7
   fcomp.append(f7)

   f8 = k * fe8
   f8 = np.sum(f8)
   #print f8
   fcomp.append(f8)

   fcomp = np.array(fcomp)
   t = fcomp.argmax()
   check = fcomp.max() / 4
   if check > 0.7:
      return t, t
   else:
      return -1, t


   #if maxp == f1:
      #print "f1"
   #   result.append(1)
   #if maxp == f2:
      #print "f2"
   #   result.append(2)
   #if maxp == f3:
      #print "f3"
   #   result.append(3)
   #if maxp == f4:
      #print "f4"
   #   result.append(4)
   #if maxp == f5:
      #print "f5"
   #   result.append(5)
   #if maxp == f6:
      #print "f6"
   #   result.append(6)
   #if maxp == f7:
      #print "f7"
   #   result.append(7)
   #if maxp == f8:
      #print "f8"
   #  result.append(8)

   #return result


space = 1

w = rw.PVReadWeights(sys.argv[1])
coord = sys.argv[2]
coord = int(coord)

nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = 10
marginstart = margin
marginend = nx - margin

nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.

h = np.zeros(9, dtype=int)
g = np.zeros(9, dtype=int)
for k in range(numpat):
   kxOn = conv.kxPos(k, nx, ny, nf)
   kyOn = conv.kyPos(k, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         d = whatFeature(p)
         e = d[0]
         f = d[1]
         e += 1
         f += 1
         h[e] += 1
         g[f] += 1 

fig = plt.figure()
ax = fig.add_subplot(211, axisbg='darkslategray')

ax.plot(np.arange(len(h)), h, 'o', color='y')

ax.set_xlabel('Features > 0.7')
ax.set_ylabel('COUNT')
ax.set_title('Weight Histogram')
ax.grid(True)

ax = fig.add_subplot(212, axisbg='darkslategray')

ax.plot(np.arange(len(g)), g, 'o', color='y')

ax.set_xlabel('All Features')
ax.set_ylabel('COUNT')
ax.grid(True)




plt.show()
