"""
auto correlation
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadSparse as rs
import PVReadWeights as rw
import PVConversions as conv
import scipy.cluster.vq as sp
import math
import os

if len(sys.argv) < 3:
   print "usage: time_stability filename on, filename-on post,"
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

tl = nxp * nyp * 2
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

   w = rw.PVReadWeights(sys.argv[2])
   w.rewind()

   number = w.numPatches 
   count = 0

   exp = []
   expOff = []
   exppn = []
   exppnOff = []

   body = w.recSize + 4
   body = 475140
   hs = w.headerSize
   filesize = os.path.getsize(sys.argv[2])
   bint = filesize / body

   bint = bint - forwardjump - 1

   if forwardjump == 0:
      4
   else:
      leap = ((body * forwardjump) + (100 * forwardjump))
      w.file.seek(leap, os.SEEK_CUR)




   countso = 0

   for i in range(50): #bint):
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
               if len(p) == 0:
                  print"STOPPEP SUPER  EARLY"
                  sys.exit()
               exp = np.append(exp,p)
            else:
#               pospost = patpla[j - 1]
#               poscur = patpla[j]
#               jump = poscur - pospost - 29
#               w.file.seek(jump, os.SEEK_CUR)
#               wOff.file.seek(jump, os.SEEK_CUR)
               p = w.next_patch()
               if len(p) == 0:
                  print"STOPPED EARLY"
                  sys.exit()
               exp = np.append(exp,p)
               #print "Ch-Ch-Changes", exppn
      else:
         count = 0
         prejump = hs #+ body #- patpla[lenpat-1]
         w.file.seek(prejump, os.SEEK_CUR)
         exp2 = []
         for j in range(numpat):
            if j == 0:
#               go = patpla[0] - 4 - 29
#               w.file.seek(go, os.SEEK_CUR)
#               wOff.file.seek(go, os.SEEK_CUR)
               p = w.next_patch()
               test = p
               if len(test) == 0:
                  print "stop"
                  input('Press Enter to Continue')
                  sys.exit()
               exp2 = np.append(exp2, p)
               count += 1
            else:
 #              pospost = patpla[j - 1]
 #              poscur = patpla[j]
 #              jump = poscur - pospost - 29
 #              w.file.seek(jump, os.SEEK_CUR)
 #              wOff.file.seek(jump, os.SEEK_CUR)
               p = w.next_patch()
               test = p
               if len(test) == 0:
                  print "stop"
                  input('Press Enter to Continue')
                  sys.exit()
               exp2 = np.append(exp2, p)
               count += 1
         exp = np.vstack((exp, exp2))

   return exp

w = rw.PVReadWeights(sys.argv[1])

body = w.recSize + 4
hs = w.headerSize
filesize = os.path.getsize(sys.argv[2])
bint = filesize / body


print
print "Number of steps = ", bint

forwardjump = 0

idra = k_stability_analysis(forwardjump)

le = np.shape(idra)

print le

eg = le[1]
eg = eg / 2
eg = 1000
print "eg = ", eg

for i in range(eg): #numpat
   incon = idra[0:, i]
   #print incon[(len(le) / 2.):]
   #print type(incon)
   boxer = plt.acorr(incon, usevlines=False, normed = True, maxlags=50)
   boxer = boxer[1]
   #print "boxer = ", boxer
   #print "type boxer = ", type(boxer)
   the = boxer[(len(boxer) / 2):]
   if i == 0:
      huk = np.zeros((len(the)))
   huk = np.add(huk, boxer[len(boxer)/2:])

print "pre huk = ", huk

huk = huk / eg

print "huk = ", huk

a = sys.argv[3]

np.savetxt("time-stability-%s.txt" %(a), huk, fmt='%f', delimiter = ';')       

print "FIN"
