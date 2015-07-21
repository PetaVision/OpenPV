"""
Plots the time stability
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw
import PVConversions as conv
import PVReadSparse as rs
import scipy.cluster.vq as sp
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


space = 1

weights = rw.PVReadWeights(sys.argv[1])
coord = 3

nx  = weights.nx
ny  = weights.ny
nxp = weights.nxp
nyp = weights.nyp

nx_imp = nx * (nxp + space) + space
ny_imp = ny * (nyp + space) + space

imp = np.zeros((nx_imp, ny_imp))
imp[:,:] = (weights.max - weights.min) / 2.

for k in range(weights.numPatches):
   P = weights.next_patch()
   if len(P) != nxp * nyp:
      continue

   P = np.reshape(P, (nxp, nyp))
   numrows, numcols = P.shape

   x = space + (space + nxp) * (k % nx)
   y = space + (space + nyp) * (k / nx)

   imp[y:y+nyp, x:x+nxp] = P

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title('Weight Patches')
ax.format_coord = format_coord

ax.imshow(imp, cmap=cm.jet, interpolation='nearest', vmin=weights.min, vmax=weights.max)


kxPre = input("enter x: ")
kyPre = input("enter y: ")

plt.show()





################################
extended = False



weights.rewind()
w = rw.PVReadWeights(sys.argv[1])
wl = rw.PVReadWeights(sys.argv[2])
wa = rs.PVReadSparse(sys.argv[3], extended)

numpat = w.numPatches

wnx = w.nx
wny = w.ny
wnf = w.nf
wnxp = w.nxp
wnyp = w.nyp

print "RX"
rx = conv.receptiveField(w, wl, wa, kxPre)
print
print "RY"
ry = conv.receptiveField(w, wl, wa, kyPre)

print
print "rx = ", rx
print "ry = ", ry
l1headx = rx[0]
l1tailx = rx[1]
rheadx = rx[2]
rtailx = rx[3]
l1heady = ry[0]
l1taily = ry[1]
rheady = ry[2]
rtaily = ry[3]


nx_im = wnx * (wnxp + space) + space
ny_im = wny * (wnyp + space) + space
small = w.max / 2


im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.


count = 0
wheremp = []

for k in range(w.numPatches):
   kx = conv.kxPos(k, wnx, wny, wnf)
   ky = conv.kyPos(k, wnx, wny, wnf)
   P = w.next_patch()
   if len(P) != wnxp * wnyp:
      continue

   if kx == kxPre:
      if ky == kyPre:
         mp = P
         for i in range(wnxp*wnxp):
            if mp[i] <= small:
               mp[i] = 0.0
            else:
               wheremp = np.append(wheremp, count) 
            count += 1
         mp = np.reshape(mp, (wnxp, wnyp))

   P = np.reshape(P, (wnxp, wnyp))
   numrows, numcols = P.shape

   x = space + (space + wnxp) * (k % wnx)
   y = space + (space + wnyp) * (k / wnx)

   im[y:y+wnyp, x:x+wnxp] = P


wherex = []
wherey = []

#print wheremp
for i in range(len(wheremp)):
   x = wheremp[i] % wnxp
   y = int(wheremp[i] / wnxp)
   #print "x = ", x + kxPre
   #print "y = ", y + kyPre
   #print
   wherex = np.append(wherex, x)
   wherey = np.append(wherey, y)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Weight Patche')
ax.imshow(mp, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)


w2 = rw.PVReadWeights(sys.argv[2])

numpat = w2.numPatches

d = np.zeros((4,4))
nx  = w2.nx
ny  = w2.ny
nxp = w2.nxp
nyp = w2.nyp
nf = w2.nf

pref = 2
d2 = np.zeros((1, (nxp * nyp)))

if pref == 2:

   d = np.zeros((4,4))
   nx  = w2.nx
   ny  = w2.ny
   nxp = w2.nxp
   nyp = w2.nyp
   nf = w2.nf

   whichpatx = l1headx
   whichpaty = l1heady

   rangex = whichpatx + 15
   rangey = whichpaty + 15

   nx_im2 = nx * (nxp + space) + space
   ny_im2 = ny * (nyp + space) + space

   im2 = np.zeros((nx_im2, ny_im2))
   im2[:,:] = (w2.max - w2.min) / 2.0


   count = 0
   count2 = 0
   for i in range(numpat):
      kx = conv.kxPos(i, nx, ny, nf)
      ky = conv.kyPos(i, nx, ny, nf)
      p = w2.next_patch()
      if whichpatx <= kx < rangex:
         if whichpaty <= ky < rangey:

            for (l) in range(len(wherex)):
               t1 = p

               if kx == (wherex[l] + kxPre - 7) and ky == (wherey[l] + kyPre - 7):
                  count2 += 1
                  e = t1
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  break
               
      
               else:
                  e = d2
                  e = e.reshape(nxp, nyp)


            count += 1

         else:
            e = d
            count += 1
      else:
         e = d
         count += 1

      x = space + (space + nxp) * (i % nx)
      y = space + (space + nyp) * (i / nx)

      im2[y:y+nyp, x:x+nxp] = e
   #print count2
   print

   coord = 1
   fig2 = plt.figure()
   ax2 = fig2.add_subplot(111)
   ax2.format_coord = format_coord
   ax2.set_title('Chosen Patches')
   ax2.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w2.min, vmax=w2.max)
   plt.show()




