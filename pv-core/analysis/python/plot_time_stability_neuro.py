"""
Plots the time stability
"""
import os
import sys
import numpy as np

import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw
import PVConversions as conv
import math

if len(sys.argv) < 6:
   print "usage: time_stability filename on, filename off, filename-on post, filename-off post"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])
wOff = rw.PVReadWeights(sys.argv[2])

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

tl = nxp * nyp * 2



def format_coord(x, y):
   col = int(x+0.5)
   row = int(y+0.5)
   x2 = (x / 16.0)
   y2 = (y / 16.0) 
   x = (x / 4.0)
   y = (y / 4.0)
   if col>=0 and col<numcols and row>=0 and row<numrows:
      z = P[row,col]
      return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
   else:
      return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))



k = 16


nx_im = 2 * (nxp + space) + space
ny_im = k * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.

nx_im2 = nx * (nxp)
ny_im2 = ny * (nyp)

im2 = np.zeros((nx_im2, ny_im2))
im2[:,:] = (w.max - w.min) / 2.

nx_im3 = nx * (nxp)
ny_im3 = ny * (nyp)

im3 = np.zeros((nx_im3, ny_im3))
im3[:,:] = (w.max - w.min) / 2.

  






total = []
logtotal = []

def k_stability_analysis(forwardjump):
   w = rw.PVReadWeights(sys.argv[1])

   count = 0
   d = np.zeros((nxp,nyp))

   w.rewind()

   ##########
   # Find Valuse of K-cluster[x] Patches
   ##########

   w = rw.PVReadWeights(sys.argv[3])
   wOff = rw.PVReadWeights(sys.argv[4])
   w.rewind()
   wOff.rewind()

   number = w.numPatches 
   count = 0

   exp = []
   expOff = []
   exppn = []
   exppnOff = []

   body = w.recSize + 4
   hs = w.headerSize
   filesize = os.path.getsize(sys.argv[3])
   bint = filesize / body

   bint = bint - forwardjump - 1
   print "bint = ", bint
   if forwardjump == 0:
      4
   else:
      leap = ((body * forwardjump) + (100 * forwardjump))
      w.file.seek(leap, os.SEEK_CUR)




   countso = 0

   for i in range(bint):
      #print "bint = ", i
      countso += 1
      cso = countso%10
      if cso == 0:
         print countso
      if i == 0:
         for j in range(numpat):
            if j == 0:
#               go = patpla[0] - hs - 29
               p = w.next_patch()
               pOff = wOff.next_patch()
               if len(p) == 0:
                  print"STOPPEP SUPER  EARLY"
                  sys.exit()
               don = p
               doff = pOff
               allpat = 0

               d = np.append(don, doff)
               p = w.normalize(d)
               pn = p
               pn = np.reshape(np.matrix(pn),(1,tl))
               p = np.reshape(np.matrix(p),(tl,1))
               pm = pn * p
               exppn = np.append(exppn, pn)
               exp = np.append(exp,pm)
               
            else:
#               pospost = patpla[j - 1]
#               poscur = patpla[j]
#               jump = poscur - pospost - 29
#               w.file.seek(jump, os.SEEK_CUR)
#               wOff.file.seek(jump, os.SEEK_CUR)
               p = w.next_patch()
               pOff = wOff.next_patch()
               if len(pOff) == 0:
                  print"STOPPED EARLY"
                  sys.exit()
               don = p
               doff = pOff
               d = np.append(don, doff)
               p = w.normalize(d)
               pn = p
               pn = np.reshape(np.matrix(pn),(1,tl))
               p = np.reshape(np.matrix(p),(tl,1))
               pm = pn * p
               exppn = np.append(exppn, pn)
               exp = np.append(exp,pm)
               #print "Ch-Ch-Changes", exppn
      else:
         count = 0
         prejump = hs #+ body #- patpla[lenpat-1]
         w.file.seek(prejump, os.SEEK_CUR)
         wOff.file.seek(prejump, os.SEEK_CUR)
         for j in range(numpat):
            if j == 0:
#               go = patpla[0] - 4 - 29
#               w.file.seek(go, os.SEEK_CUR)
#               wOff.file.seek(go, os.SEEK_CUR)
               p = w.next_patch()
               pOff = wOff.next_patch()
               test = p
               if len(test) == 0:
                  print "stop"
                  input('Press Enter to Continue')
                  sys.exit()
               don = p
               doff = pOff
               d = np.append(don, doff)
               p = w.normalize(d)
               p = np.reshape(np.matrix(p),(tl,1))
               j1 = 0
               j2 = tl
               pm = np.matrix(exppn[j1:j2]) * p
               expn = pm[0]
               count += 1
            else:
 #              pospost = patpla[j - 1]
 #              poscur = patpla[j]
 #              jump = poscur - pospost - 29
 #              w.file.seek(jump, os.SEEK_CUR)
 #              wOff.file.seek(jump, os.SEEK_CUR)
               p = w.next_patch()
               pOff = wOff.next_patch()
               test = pOff
               if len(test) == 0:
                  print "stop"
                  input('Press Enter to Continue')
                  sys.exit()
               don = p
               doff = pOff
               d = np.append(don, doff)
               p = w.normalize(d)
               p = np.reshape(np.matrix(p),(tl,1))
               j1 = tl * j
               j2 = tl * (j +1)
               pm = np.matrix(exppn[j1:j2]) * p
               pm = np.array(pm)
               #print "expn = ", np.shape(expn)
               #print "pm = ", np.shape(pm)
               expn = np.add(expn,pm[0])
               count += 1

      if i == 0:
         idra = np.average(exp)
      else:
         idra = np.append(idra, (expn/numpat))   

   ##########
   # Find Average of K-cluster[x] Weights
   ##########

   return idra




w = rw.PVReadWeights(sys.argv[3])

body = w.recSize + 4
hs = w.headerSize
filesize = os.path.getsize(sys.argv[3])
bint = filesize / body


print
print "Number of steps = ", bint

forwardjump = 0


idra = k_stability_analysis(forwardjump)




print "final idra = ", idra


a = sys.argv[5]

np.savetxt("time-stability-%s.txt" %(a), idra, fmt='%f', delimiter = ';')        


