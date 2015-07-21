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

def format_coord(x, y):
   col = int(x+0.5)
   row = int(y+0.5)
   x2 = (x / (np.shape(a)[0]/nxp))
   y2 = (y / (np.shape(a)[1]/nyp))
   x = (x / 6.0)
   y = (y / 6.0)
   #if col>=0 and col<numcols and row>=0 and row<numrows:
   #   z = P[row,col]
   #   return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
   #else:
   return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))

global x
global y
global numrows
global numcols



if len(sys.argv) < 2:
   print "usage: time_stability post_filename"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])


if 1 == 1:
   space = 1
   nx = w.nx
   ny = w.ny
   nxp = w.nxp
   nyp = w.nyp

   nx2 = 17 #nxynum 
   ny2 = 17 #nxynum
   nx_im = nx2 * (nxp + space) + space
   ny_im = ny2 * (nyp + space) + space
   im = np.zeros((nx_im, ny_im))
   im[:,:] = (w.max - w.min) / 2.



def patchover(x, huk, testnumber):

   w = rw.PVReadWeights(sys.argv[x])


   space = 1

   d = np.zeros((4,4))

   nx = w.nx
   ny = w.ny
   nxp = w.nxp
   nyp = w.nyp
   coor = nxp * nyp
   #print coor

   patlen = nxp * nyp
   numpat = w.numPatches
   nf = w.nf
   check = sys.argv[1]
   a = check.find("w")
   b = check[a].strip("w")
   #print b

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


   if testnumber == 0:
      for i in range(1):
         while test == 0:
            testnumber = random.randint(0, numpat)
            for i in range(128):
               for j in range(128):
                  if marginstart < i < marginend:
                     if marginstart < j < marginend:
                        if testnumber == (j + (i*128)):
                           test = 1


   #testnumber = (69 + (62 * 128))
   postpat = []   

   for ko in range(numpat):
      kxOn = conv.kxPos(ko, nx, ny, nf)
      kyOn = conv.kyPos(ko, nx, ny, nf)
      prepat = w.file.tell()
      p = w.next_patch()
      postpat = w.file.tell()     
      #print postpat
      if marginstart < kxOn < marginend:
         if marginstart < kyOn < marginend:
            if testnumber == (kxOn + (kyOn*128)):
               patpos = w.file.tell()
               #print
               #print "kx", kxOn
               #print "ky", kyOn


   donepat = postpat - prepat
   #print donepat


   w.rewind()


   exp = []
   exppn = []
   exp2 = []
   exppn2 = []

   body = w.recSize + 4
   #print "body = ", body 
   body = 475140
   hs = w.headerSize
   #print "hs = ", hs
   filesize = os.path.getsize(sys.argv[x])
   bint = filesize / body
   #print "bint = ", bint
   #sys.exit()

   #print
   #print "Number of steps = ", bint
   #forwardjump = input('How many steps forward:')
   forwardjump = 0
   bint = bint - forwardjump

   sqnxynum = math.sqrt(bint)
   nxynum = int(round(sqnxynum, 0))
   nxynum += 1


   if forwardjump == 0:
      1
   else:
      leap = ((body * forwardjump) + (100 * forwardjump)) 
      w.file.seek(leap, os.SEEK_CUR)



   global count
   if huk == 0:
      count = 0
   allpat = []
   count2 = 1

   for i in range(bint):
      if i == 0:
         #print "count2 = ", count2
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

         count2 += 1
         count += 1
      else:
         #print "count2 = ", count2
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
         hh = bint / 16
         yy = i%hh
         #print "yy = ", yy


         if yy == 0:
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
            count2 += 1


   #allpat = np.split(allpat, (count - 1))
   #lenallpat = len(allpat)
   #print np.shape(allpat)
   #print "count", count
   #print

   #for i in range(lenallpat):
   #   if i == 0:
   #      other = -(i + 1)
   #      p = w.normalize(allpat[other])
   #      pn = p
   #      pn = np.reshape(np.matrix(pn),(1,patlen))
   #      p = np.reshape(np.matrix(p),(patlen,1))
   #      pm = pn * p
   #      exppn = np.append(exppn, pn)
   #      exp = np.append(exp,pm)
   #   else:
   #      other = -(i + 1)
   #      p = w.normalize(allpat[other])
   #      p = np.reshape(np.matrix(p),(patlen,1))
   #      j1 = 0
   #      j2 = patlen
   #      pm = np.matrix(exppn[j1:j2]) * p
   #      exp = np.append(exp,pm)

   wir = [[im], [testnumber]]
   return wir 

wpat = []

for i in range(2):
   i += 1
   if i == 1:
      for j in range(8):
         if i == 1 and j == 0:
            huk = 0
         else:
            huk = 1
         a = patchover(i, huk, 0)
         wpat = np.append(wpat, a[1])
         a = a[0]
   if i == 2:
      count +=17
      for j in range(8):
         if i == 1 and j == 0:
            huk = 0
         else:
            huk = 1
         a = patchover(i, huk, wpat[j])
         a = a[0]


   
   print "a = ", a


a = a[0]

w = rw.PVReadWeights(sys.argv[1])

som = 2000000 / 8



ind = []
where = []

for i in range(8):
   ind = np.append(ind, i*12)
   where = np.append(where, i * som)

ind = np.add(ind, 3)
where = np.divide(where, 1000000)



wherey = [(len(a)/ 5.0), ((len(a)/5.0) * 4)]


fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Time (x20000ms)')
#ax.set_ylabel('No Inhib Patches                                       Inhib Patches')
ax.set_title('Weight Patches')
#ax.format_coord = format_coord
plt.xticks(ind, where )
plt. yticks(wherey, ('Inhib Patches', 'No Inhib Patches'), rotation='horizontal')

ax.imshow(a, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)




plt.show()


sys.exit()
