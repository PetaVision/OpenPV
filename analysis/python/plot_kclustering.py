"""
Plots the k-means clustering
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw
import PVConversions as conv
import scipy.cluster.vq as sp
import math


def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = P[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)

if len(sys.argv) < 2:
   print "usage: kclustering filename"
   print len(sys.argv)
   sys.exit()

k = 8
space = 1

d = np.zeros((4,4))
w = rw.PVReadWeights(sys.argv[1])

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
x = w.numPatches
nf = w.nf
margin = 10
marginstart = margin
marginend = nx - margin


d = w.next_patch()
for ko in np.arange(x-1):
   p = w.next_patch()
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         d = np.vstack((d,p))

wd = sp.whiten(d)
result = sp.kmeans(wd, k)

nx_im = 2 * (nxp + space) + space
ny_im = 4 * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.
print
print 
for i in np.arange(k):
   a = result[0]
   a = a[i].reshape(nxp, nyp)
   numrows, numcols = a.shape
   print a

   x = space + (space + nxp) * (i % 4)
   y = space + (space + nyp) * (i / 4)

   im[y:y+nyp, x:x+nxp] = a

print
print
print
print "distortion : float = ", result[1]
print 
print
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title('Weight Patches')
ax.format_coord = format_coord

ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

plt.show()
