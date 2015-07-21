"""
Plots the time stability for a random individual patch
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
import random

if len(sys.argv) < 2:
   print "usage: time_stability post_filename"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])

d = os.path.getsize(sys.argv[1])

space = 1

d = np.zeros((4,4))

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
patlen = nxp * nyp
numpat = w.numPatches
nf = w.nf
margin = 10
marginstart = margin
marginend = nx - margin
acount = 0
patchposition = []


nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp


zed = 0
allpatpos = []

tellcount = 0
for ko in range(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
            patpos = w.file.tell()
            allpatpos = np.append(allpatpos, patpos)
            tellcount += 1
      else:
         allpatpos = np.append(allpatpos, zed)
   else:
      allpatpos = np.append(allpatpos, zed)

w.rewind()

print "length of allpatpos", len(allpatpos)

exp = []
exppn = []
exp2 = []
exppn2 = []

body = w.recSize + 4
hs = w.headerSize
filesize = os.path.getsize(sys.argv[1])
bint = filesize / body

print
print "Number of steps = ", bint
#forwardjump = input('How many steps forward:')
forwardjump = 0
bint = bint - forwardjump

sqnxynum = math.sqrt(bint)
nxynum = int(round(sqnxynum, 0))
nxynum += 1


nx2 = nxynum
ny2 = nxynum
nx_im = nx2 * (nxp + space) + space
ny_im = ny2 * (nyp + space) + space
im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.


if forwardjump == 0:
   print
else:
   leap = ((body * forwardjump) + (100 * forwardjump)) 
   w.file.seek(leap, os.SEEK_CUR)


count = 1
allpat = []
count4 = 0
addp = 0

for i in range(bint):
   count4 = 0


   for k in range(numpat):
      p = w.next_patch()
      where = w.file.tell()
      if where == (allpatpos[k] + (327784 * i)):
         if count4 == 0:
            addp = p
            count4 += 1
         else:
            addp = np.add(addp, p)
            count4 += 1

   sub = addp / count4
   #print sub
   jump = hs
   #print "total", body + hs
   #print "where", where
   #print "jump",jump
   
   w.file.seek(jump, os.SEEK_CUR)




   if i == 0:
      don = sub
      allpat = don
      P = np.reshape(sub, (nxp, nyp))
      numrows, numcols = P.shape
      x = space + (space + nxp) * (count % nx2)
      y = space + (space + nyp) * (count / nx2)
      im[y:y+nyp, x:x+nxp] = P
      p = w.normalize(don)
      pn = p
      pn = np.reshape(np.matrix(pn),(1,16))
      p = np.reshape(np.matrix(p),(16,1))
      pm = pn * p
      exppn2 = np.append(exppn2, pn)
      exp2 = np.append(exp2,pm)

      count += 1
   
   else:
      don = sub
      allpat = np.append(allpat, don)

      P = np.reshape(sub, (nxp, nyp))
      numrows, numcols = P.shape
      x = space + (space + nxp) * (count % nx2)
      y = space + (space + nyp) * (count / nx2)
      im[y:y+nyp, x:x+nxp] = P

      p = w.normalize(don)
      p = np.reshape(np.matrix(p),(16,1))
      j1 = 0
      j2 = 16
      pm = np.matrix(exppn2[j1:j2]) * p
      exp2 = np.append(exp2,pm)
      count += 1





allpat = np.split(allpat, (count - 1))
lenallpat = len(allpat)
print np.shape(allpat)
print "count", count
print

for i in range(lenallpat):
   if i == 0:
      other = -(i + 1)
      p = w.normalize(allpat[other])
      pn = p
      pn = np.reshape(np.matrix(pn),(1,patlen))
      p = np.reshape(np.matrix(p),(patlen,1))
      pm = pn * p
      exppn = np.append(exppn, pn)
      exp = np.append(exp,pm)
   else:
      other = -(i + 1)
      p = w.normalize(allpat[other])
      p = np.reshape(np.matrix(p),(patlen,1))
      j1 = 0
      j2 = patlen
      pm = np.matrix(exppn[j1:j2]) * p
      exp = np.append(exp,pm)




fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title('Weight Patches')

ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)


fig2 = plt.figure()
ax2 = fig2.add_subplot(111, axisbg='darkslategray')

ax2.plot(np.arange(len(exp)), exp, '-o', color='w')
ax2.plot(np.arange(len(exp2)), exp2, '-o', color='k')


ax2.set_xlabel('Time   WHITE = backward through time   BLACK = forward through time')
ax2.set_ylabel('Avg Correlation')
ax2.set_title('Through Time')
ax2.set_xlim(0, len(exp))
ax2.set_ylim(0, 1)
ax2.grid(True)




plt.show()


sys.exit()






