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





nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp

nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.

for k in range(w.numPatches):
   P = w.next_patch()
   if len(P) != nxp * nyp:
      continue

   #print "p = ", P
   #if k == 500:
   #   sys.exit()
   P = np.reshape(P, (nxp, nyp))
   numrows, numcols = P.shape

   x = space + (space + nxp) * (k % nx)
   y = space + (space + nyp) * (k / nx)

   im[y:y+nyp, x:x+nxp] = P

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title('Weight Patches')

ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

plt.show()
sys.exit()












count = 1
allpat = []

for i in range(bint):

   if i == 0:
      go = patpos - hs - donepat
      w.file.seek(go, os.SEEK_CUR)
      p = w.next_patch()
      #print count
      #print w.file.tell()
      if len(p) == 0:
         print"STOPPEP SUPER  EARLY"
         sys.exit()
      don = p
      allpat = don

      P = np.reshape(p, (nxp, nyp))
      numrows, numcols = P.shape
      x = space + (space + nxp) * (count % nx2)
      y = space + (space + nyp) * (count / nx2)
      im[y:y+nyp, x:x+nxp] = P
      p = w.normalize(don)
      pn = p
      pn = np.reshape(np.matrix(pn),(1,coor))
      p = np.reshape(np.matrix(p),(coor,1))
      pm = pn * p
      exppn2 = np.append(exppn2, pn)
      exp2 = np.append(exp2,pm)


      count += 1
   else:
      prejump = body - patpos + hs
      w.file.seek(prejump, os.SEEK_CUR)
      go = patpos - 4 - donepat
      w.file.seek(go, os.SEEK_CUR)
      p = w.next_patch()
      #print w.file.tell()
      test = p
      #print count
      if len(test) == 0:
         print "stop"
         input('Press Enter to Continue')
         sys.exit()
      don = p
      allpat = np.append(allpat, don)

      P = np.reshape(p, (nxp, nyp))
      numrows, numcols = P.shape
      x = space + (space + nxp) * (count % nx2)
      y = space + (space + nyp) * (count / nx2)
      im[y:y+nyp, x:x+nxp] = P

      p = w.normalize(don)
      p = np.reshape(np.matrix(p),(coor,1))
      j1 = 0
      j2 = coor
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



#ax2 = fig.add_subplot(212, axisbg='darkslategray')

#ax2.plot(np.arange(len(exp)), exp, '-o', color='w')
#ax2.plot(np.arange(len(exp2)), exp2, '-o', color='k')


#ax2.set_xlabel('Time   WHITE = backward through time   BLACK = forward through time')
#ax2.set_ylabel('Avg Correlation')
#ax2.set_title('Through Time')
#ax2.set_xlim(0, len(exp))
#ax2.set_ylim(0, 1)
#ax2.grid(True)




plt.show()


sys.exit()









#################################
#Only nx=50 and ny=50


w = rw.PVReadWeights(sys.argv[1])
space = 1

d = np.zeros((4,4))

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = 30
marginstart = margin
marginend = nx - margin
acount = 0
patchposition = []






test = 0





body = w.recSize + 4
hs = w.headerSize
filesize = os.path.getsize(sys.argv[1])
bint = filesize / body

print
print "Number of steps = ", bint
forwardjump = input('How many steps forward:')

if forwardjump == 0:
   print
else:
   leap = (body * forwardjump)
   w.file.seek(leap, os.SEEK_CUR)

bint = bint - forwardjump


nx  = w.nx
ny  = w.ny
nx2 = 10
ny2 = 10
nxp = w.nxp
nyp = w.nyp

nx_im = nx2 * (nxp + space) + space
ny_im = ny2 * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.


count = 0
count2 = 0
info = []
for b in range(bint - 1):

   for ko in range(numpat):
      kxOn = conv.kxPos(ko, nx, ny, nf)
      kyOn = conv.kyPos(ko, nx, ny, nf)
      p = w.next_patch()
      if kxOn == 50:
         if kyOn == 50:
            dapat = p
            count2 += 1
            tell = w.file.tell()
            info = np.append(info, tell)
            P = np.reshape(dapat, (nxp, nyp))
            numrows, numcols = P.shape

            x = space + (space + nxp) * (count2 % nx2)
            y = space + (space + nyp) * (count2 / nx2)

            im[y:y+nyp, x:x+nxp] = P

   w.file.seek(hs, os.SEEK_CUR)
   count += 1
   #print count



info = np.vstack(info)

f = open('info.txt', 'w')
info = str(info)
f.write(info)




fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title('Weight Patches')

ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)



plt.show()
sys.exit()
