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

space = 1

d = np.zeros((4,4))

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
coor = nxp * nyp
print coor

patlen = nxp * nyp
numpat = w.numPatches
nf = w.nf
check = sys.argv[1]
a = check.find("w")
b = check[a].strip("w")
print b

margin = 10


marginstart = margin
marginend = nx - margin
acount = 0
patchposition = []


nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp



test = 0

#print testnumber
#testnumber = (69 + (62 * 128))

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
forwardjump = bint - 2
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






if len(sys.argv) == 2:

   h = w.histogram()

   low = 0
   high = 0

   for i in range(len(h)):
      if i < 126 and h[i] > 200: low += h[i]
      if i > 126 and h[i] > 200: high += h[i]

   print "low==", low, "high==", high, "total==", np.add.reduce(h)


   w_split_val = 255/2.
   if len(sys.argv) >= 3:
      w_split_val = float(sys.argv[2])

   ch = w.clique_histogram(w_split_val)
   print 'total =', sum(h)


   fig = plt.figure()
   ax = fig.add_subplot(211)

   ax.plot(np.arange(len(h)), h, '-o', color='b', linewidth=5.0)

   ax.set_xlabel('WEIGHT BINS')
   ax.set_ylabel('COUNT')
   ax.set_title('Weight Histogram')
   ax.set_xlim(0, 256)
   ax.grid(True)


   ax = fig.add_subplot(212)

   ax.plot(np.arange(len(ch)), ch, '-o', color='b',  linewidth=5.0)

   ax.set_xlabel('CLIQUE BINS')
   ax.set_ylabel('COUNT')
   ax.set_title('Clique Histogram')
   ax.set_xlim(0, 1+w.patchSize)
   ax.grid(True)

   plt.show()



else:
   w = rw.PVReadWeights(sys.argv[1])
   wOff = rw.PVReadWeights(sys.argv[2])

   h = w.histogram()
   hOff = wOff.histogram()

   low = 0
   high = 0
   lowOff = 0
   highOff = 0

   for i in range(len(h)):
      if i < 126 and h[i] > 200: low += h[i]
      if i > 126 and h[i] > 200: high += h[i]
   for i in range(len(hOff)):
      if i < 126 and hOff[i] > 200: lowOff += hOff[i]
      if i > 126 and hOff[i] > 200: highOff += hOff[i]

   print "On Weights: low==", low, "high==", high, "total==", np.add.reduce(h)
   print "Off Weights: low ==", lowOff, "high==", highOff, "total==", np.add.reduce(hOff)


   w_split_val = 255/2.
   if len(sys.argv) >= 4:
      w_split_val = float(sys.argv[3])


   chOn = w.clique_histogram(w_split_val)
   chOff = wOff.clique_histogram(w_split_val)
   print 'Clique On total =', sum(chOn)
   print 'Clique Off total =', sum(chOff)




   fig = plt.figure()


   ax = fig.add_subplot(221)
   ax.plot(np.arange(len(h)), h, '-o', color='b', linewidth=5.0)
   #ax.set_xlabel('WEIGHT BINS')
   ax.set_ylabel('ON WEIGHTS', fontsize = 'large')
   ax.set_title('Weight Histograms')
   ax.set_xlim(0, 256)


   ax = fig.add_subplot(223)
   ax.plot(np.arange(len(hOff)), hOff, '-o', color='b', linewidth=5.0)
   ax.set_xlabel('WEIGHT BINS')
   ax.set_ylabel('OFF WEIGHTS', fontsize = 'large')
   #ax.set_title('Weight Histogram')
   ax.set_xlim(0, 256)


   ax = fig.add_subplot(222)

   ax.plot(np.arange(len(chOn)), chOn, '-o', color='b', linewidth=5.0)

   #ax.set_xlabel('CLIQUE BINS')
   ax.set_ylabel('COUNT')
   ax.set_title('Clique Histograms')
   ax.set_xlim(0, 1+w.patchSize)
   ax.grid(True)



   ax = fig.add_subplot(224)

   ax.plot(np.arange(len(chOff)), chOff, '-o', color='b', linewidth=5.0)

   ax.set_xlabel('CLIQUE BINS')
   ax.set_ylabel('COUNT')
   #ax.set_title('Off Clique Histogram')
   ax.set_xlim(0, 1+w.patchSize)
   ax.grid(True)


   plt.show()
