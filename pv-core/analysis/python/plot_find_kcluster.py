"""
Plots the k-means clustering
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import matplotlib.figure as mfig
import PVReadWeights as rw
import PVConversions as conv
import scipy.cluster.vq as sp
import math
import decimal as dc


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

w = rw.PVReadWeights(sys.argv[1])

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
therange = w.numPatches
nf = w.nf
margin = 10
marginstart = margin
marginend = nx - margin
acount = 0
d = np.zeros((nxp,nyp))

if len(sys.argv) > 1:
   for ko in np.arange(therange):
      p = w.next_patch()
      kxOn = conv.kxPos(ko, nx, ny, nf)
      kyOn = conv.kyPos(ko, nx, ny, nf)
      if marginstart < kxOn < marginend:
         if marginstart < kyOn < marginend:
            acount = acount + 1
            if kxOn == margin + 1 and kyOn == margin + 1:
               d = p
            else:
               d = np.vstack((d,p))
   
   wd = sp.whiten(d)
   result = sp.kmeans2(wd, k)
   cluster = result[1]
   a = result[0]

   nx_im = nx * (nxp + space) + space
   ny_im = ny * (nyp + space) + space

   im = np.zeros((nx_im, ny_im))
   im[:,:] = (w.max - w.min) / 2.

   nx_im2 = 2 * (nxp + space) + space
   ny_im2 = 4 * (nyp + space) + space

   im2 = np.zeros((nx_im2, ny_im2))
   im2[:,:] = (w.max - w.min) / 2.


   for i in np.arange(k):
      at = a[i].reshape(nxp, nyp)
      numrows, numcols = at.shape

      x = space + (space + nxp) * (i % 4)
      y = space + (space + nyp) * (i / 4)

      im2[y:y+nyp, x:x+nxp] = at


   kcount1 = 0.0
   kcount2 = 0.0
   kcount3 = 0.0
   kcount4 = 0.0
   kcount5 = 0.0
   kcount6 = 0.0
   kcount7 = 0.0
   kcount8 = 0.0

   for i in range(acount):
      if cluster[i] == 0:
         kcount1 = kcount1 + 1
      if cluster[i] == 1:
         kcount2 = kcount2 + 1
      if cluster[i] == 2:
         kcount3 = kcount3 + 1
      if cluster[i] == 3:
         kcount4 = kcount4 + 1
      if cluster[i] == 4:
         kcount5 = kcount5 + 1
      if cluster[i] == 5:
         kcount6 = kcount6 + 1
      if cluster[i] == 6:
         kcount7 = kcount7 + 1
      if cluster[i] == 7:
         kcount8 = kcount8 + 1

   kcountper1 = kcount1 / acount 
   kcountper2 = kcount2 / acount 
   kcountper3 = kcount3 / acount 
   kcountper4 = kcount4 / acount 
   kcountper5 = kcount5 / acount 
   kcountper6 = kcount6 / acount 
   kcountper7 = kcount7 / acount 
   kcountper8 = kcount8 / acount 


   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.set_axis_off()
   ax.text(2, -1, "%.2f                %.2f             %.2f                %.2f" %(kcountper1,  kcountper2,  kcountper3, kcountper4), fontsize='large', rotation='horizontal')
   ax.text(2, 11.5, "%.2f                %.2f             %.2f                %.2f" %(kcountper5,  kcountper6,  kcountper7, kcountper8), fontsize='large', rotation='horizontal')

   ax.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
   plt.show()


   feature = input('Please which k-cluster to compare:')
   feature -= 1

   count = 0
   d = np.zeros((nxp,nyp))

   w.rewind()
   for ko in np.arange(therange):
      kxOn = conv.kxPos(ko, nx, ny, nf)
      kyOn = conv.kyPos(ko, nx, ny, nf)
      p = w.next_patch()
      if marginstart < kxOn < marginend:
         if marginstart < kyOn < marginend:
            if cluster[count] == feature:
               e = p
               e = e.reshape(nxp, nyp)
               numrows, numcols = e.shape
               count = count + 1
            else:
               e = d
               count = count + 1
         else:
            e = d
      else:
         e = d
      x = space + (space + nxp) * (ko % nx)
      y = space + (space + nyp) * (ko / nx)

      im[y:y+nyp, x:x+nxp] = e

   fig = plt.figure()
   ax = fig.add_subplot(111)

   ax.set_xlabel('Kx GLOBAL')
   ax.set_ylabel('Ky GLOBAL')
   ax.set_title('Weight Patches')
   ax.format_coord = format_coord

   ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

   plt.show()
