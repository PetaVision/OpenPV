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
from pylab import save
import math
import random

if len(sys.argv) < 3:
   print "usage: hamming filename value"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])



space = 1

d = np.zeros((4,4))

wmax = w.max
nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = 0
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
fe9 = []
fe10 = []
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

f = np.zeros([w.nxp, w.nyp])
f[:,4] = 1
fe5.append(f)

#horizontal lines from the top
f = np.zeros([w.nxp, w.nyp])
f[0,:] = 1
fe6.append(f)

f = np.zeros([w.nxp, w.nyp])
f[1,:] = 1
fe7.append(f)

f = np.zeros([w.nxp, w.nyp])
f[2,:] = 1
fe8.append(f)

f = np.zeros([w.nxp, w.nyp])
f[3,:] = 1
fe9.append(f)

f = np.zeros([w.nxp, w.nyp])
f[4,:] = 1
fe10.append(f)




#print "f1", fe1
#print "f2", fe2
#print "f3", fe3
#print "f4", fe4
#print "f5", fe5
#print "f6", fe6
#print "f7", fe7
#print "f8", fe8
#print "f9", fe9
#print "f10", fe10

fe1 = np.reshape(fe1, (25))
fe2 = np.reshape(fe2, (25))
fe3 = np.reshape(fe3, (25))
fe4 = np.reshape(fe4, (25))
fe5 = np.reshape(fe5, (25))
fe6 = np.reshape(fe6, (25))
fe7 = np.reshape(fe7, (25))
fe8 = np.reshape(fe8, (25))
fe9 = np.reshape(fe9, (25))
fe10 = np.reshape(fe10, (25))





def whatFeature(k):
   result = []
   fcomp = []

   for i in range(len(k)):
      if k[i] > (0.5 * wmax):
         k[i] = 1
      else:
         k[i] = 0

   diff1 = 0
   diff2 = 0
   diff3 = 0
   diff4 = 0
   diff5 = 0
   diff6 = 0
   diff7 = 0
   diff8 = 0
   diff9 = 0
   diff10 = 0

   for a, b in zip(k, fe1):
      if a!=b:
         diff1+=1
   for a, b in zip(k, fe2):
      if a!=b:
         diff2+=1
   for a, b in zip(k, fe3):
      if a!=b:
         diff3+=1
   for a, b in zip(k, fe4):
      if a!=b:
         diff4+=1
   for a, b in zip(k, fe5):
      if a!=b:
         diff5+=1
   for a, b in zip(k, fe6):
      if a!=b:
         diff6+=1
   for a, b in zip(k, fe7):
      if a!=b:
         diff7+=1
   for a, b in zip(k, fe8):
      if a!=b:
         diff8+=1
   for a, b in zip(k, fe9):
      if a!=b:
         diff9+=1
   for a, b in zip(k, fe10):
      if a!=b:
         diff10+=1

   dres = [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8, diff9, diff10]

   result = np.min(dres)

   return result


space = 1

w = rw.PVReadWeights(sys.argv[1])
coord = 1
coord = int(coord)

nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = 0

start = margin
marginend = nx - margin



nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.

where = []
zep = 0

for k in range(numpat):
   kx = conv.kxPos(k, nx, ny, nf)
   ky = conv.kyPos(k, nx, ny, nf)
   p = w.next_patch()
   if len(p) != nxp * nyp:
      continue
   acount+=1
   a = whatFeature(p)
   zep += a  

im = np.array([zep / float(acount)])

print zep
print acount
print im


np.savetxt('hamming-%s.txt' %(sys.argv[2]), im, fmt="%10.5f")
