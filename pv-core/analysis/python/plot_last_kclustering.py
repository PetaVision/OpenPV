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
print
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

if len(sys.argv) == 3:
   wOff = rw.PVReadWeights(sys.argv[2])
   if forwardjump == 0:
      print
   else:
      leap = ((body * forwardjump) + (100 * forwardjump)) 
      wOff.file.seek(leap, os.SEEK_CUR)

###########################################################################

space = 1

d = np.zeros((5,5))

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = int(nx / 2.5)
print "margin = ", margin
marginstart = margin
marginend = nx - margin
acount = 0

print "nx = ", nx
print "ny = ", ny
print "nxp = ", nxp
print "nyp = ", nyp



if len(sys.argv) == 2:
   def format_coord(x, y):
      col = int(x+0.5)
      row = int(y+0.5)
      x2 = (x / 20.0)
      y2 = (y / 20.0) 
      x = (x / 5.0)
      y = (y / 5.0)
      if col>=0 and col<numcols and row>=0 and row<numrows:
         z = P[row,col]
         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
      else:
         return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))


   k = 64  #24
   for ko in range(numpat):
      kxOn = conv.kxPos(ko, nx, ny, nf)
      kyOn = conv.kyPos(ko, nx, ny, nf)
      p = w.next_patch()
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

   k2 = k / 2

   nx_im = 2 * (nxp + space) + space
   ny_im = k2 * (nyp + space) + space

   im = np.zeros((nx_im, ny_im))
   im[:,:] = (w.max - w.min) / 2.


   nx_im2 = nx * (nxp + space) + space
   ny_im2 = ny * (nyp + space) + space

   im2 = np.zeros((nx_im2, ny_im2))
   im2[:,:] = (w.max - w.min) / 2.



   for i in np.arange(k):
      a = result[0]
      a = a[i].reshape(nxp, nyp)
      numrows, numcols = a.shape

      x = space + (space + nxp) * (i % k2)
      y = space + (space + nyp) * (i / k2)

      im[y:y+nyp, x:x+nxp] = a


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


   #ax.text( 2, 0, "%.2f                %.2f             %.2f                %.2f" %(kcountper1,  kcountper2,  kcountper3, kcountper4), fontsize='large', rotation='horizontal')
   #ax.text( 2, 10, "%.2f                %.2f             %.2f                %.2f" %(kcountper5,  kcountper6,  kcountper7, kcountper8), fontsize='large', rotation='horizontal')

   ax.set_title('K-Clusters')
   ax.set_axis_off()
   ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

   plt.show()
   t = 1
   while t == 1:

      feature = input('Please which k-cluster to compare: ')
      feature -= 1

      count = 0
      d = np.zeros((nxp,nyp))

      w.rewind()
      for ko in np.arange(numpat):
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


         im2[y:y+nyp, x:x+nxp] = e



      fig = plt.figure()
      ax = fig.add_subplot(111)

      ax.set_xlabel('Kx GLOBAL')
      ax.set_ylabel('Ky GLOBAL')
      ax.set_title('ON Weight Patches')
      ax.format_coord = format_coord
      ax.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

      plt.show()
      t = input("1 to repeat, 0 to end: ")


   sys.exit()











if len(sys.argv) == 3:
   
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


   k = 64
   for ko in range(numpat):
      kxOn = conv.kxPos(ko, nx, ny, nf)
      kyOn = conv.kyPos(ko, nx, ny, nf)
      p = w.next_patch()
      poff = wOff.next_patch()
      if marginstart < kxOn < marginend:
         if marginstart < kyOn < marginend:
            acount = acount + 1
            if kxOn == margin + 1 and kyOn == margin + 1:
               don = p
               doff = poff
               d = np.append(don, doff)
            else:
               don = p
               doff = poff
               e = np.append(don, doff)
               d = np.vstack((d,e))


   wd = sp.whiten(d)
   result = sp.kmeans2(wd, k)
   
   cluster = result[1]
   
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

  
   b = result[0]
   c = np.hsplit(b, 2)
   con = c[0]
   coff = c[1]

   for i in range(k):      
      d = con[i].reshape(nxp, nyp)
      numrows, numcols = d.shape

      x = space + (space + nxp) * (i % k)
      y = space + (space + nyp) * (i / k)

      im[y:y+nyp, x:x+nxp] = d
   for i in range(k):
      e = coff[i].reshape(nxp, nyp)
      numrows, numcols = e.shape

      i = i + k

      x = space + (space + nxp) * (i % k)
      y = space + (space + nyp) * (i / k)

      im[y:y+nyp, x:x+nxp] = e



   kcount1 = 0.0
   kcount2 = 0.0
   kcount3 = 0.0
   kcount4 = 0.0
   kcount5 = 0.0
   kcount6 = 0.0
   kcount7 = 0.0
   kcount8 = 0.0
   kcount9 = 0.0
   kcount10 = 0.0
   kcount11 = 0.0
   kcount12 = 0.0
   kcount13 = 0.0
   kcount14= 0.0
   kcount15 = 0.0
   kcount16 = 0.0

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
      if cluster[i] == 8:
         kcount9 = kcount9 + 1
      if cluster[i] == 9:
         kcount10 = kcount10 + 1
      if cluster[i] == 10:
         kcount11 = kcount11 + 1
      if cluster[i] == 11:
         kcount12 = kcount12 + 1
      if cluster[i] == 12:
         kcount13 = kcount13 + 1
      if cluster[i] == 13:
         kcount14 = kcount14 + 1
      if cluster[i] == 14:
         kcount15 = kcount15 + 1
      if cluster[i] == 15:
         kcount16 = kcount16 + 1





   kcountper1 = kcount1 / acount 
   kcountper2 = kcount2 / acount 
   kcountper3 = kcount3 / acount 
   kcountper4 = kcount4 / acount 
   kcountper5 = kcount5 / acount 
   kcountper6 = kcount6 / acount 
   kcountper7 = kcount7 / acount 
   kcountper8 = kcount8 / acount 
   kcountper9 = kcount9 / acount 
   kcountper10 = kcount10 / acount 
   kcountper11 = kcount11 / acount 
   kcountper12 = kcount12 / acount 
   kcountper13 = kcount13 / acount 
   kcountper14 = kcount14 / acount 
   kcountper15 = kcount15 / acount 
   kcountper16 = kcount16 / acount 





   fig = plt.figure()
   ax = fig.add_subplot(111)
   
   textx = (-7/16.0) * k
   texty = (10/16.0) * k
   
   ax.set_title('On and Off K-means')
   ax.set_axis_off()
   ax.text(textx, texty,'ON\n\nOff', fontsize='xx-large', rotation='horizontal') 
   #ax.text( -5, 12, "Percent %.2f   %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f" %(kcountper1,  kcountper2,  kcountper3, kcountper4, kcountper5, kcountper6, kcountper7, kcountper8, kcountper9, kcountper10, kcountper11, kcountper12, kcountper13, kcountper14, kcountper15, kcountper16), fontsize='large', rotation='horizontal')
   #ax.text(-4, 14, "Patch   1      2       3       4       5       6       7       8       9      10      11     12     13     14     15     16", fontsize='x-large', rotation='horizontal')

   ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

   plt.show()

   t = 1

   while t == 1:

      feature = input('Please which k-cluster to compare: ')
      feature -= 1

      count = 0
      d = np.zeros((nxp,nyp))

      w.rewind()
      for ko in np.arange(numpat):
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
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)

         im2[y:y+nyp, x:x+nxp] = e


      count = 0

      wOff.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = wOff.next_patch()
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
         x = (nxp) * (ko % nx)
         y = (nyp) * (ko / nx)

         im3[y:y+nyp, x:x+nxp] = e




      fig = plt.figure()
      ax = fig.add_subplot(111)

      ax.set_xlabel('Kx GLOBAL')
      ax.set_ylabel('Ky GLOBAL')
      ax.set_title('ON Weight Patches')
      ax.format_coord = format_coord
      ax.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
   
   
      fig2 = plt.figure()
      ax2 = fig2.add_subplot(111)

      ax2.set_xlabel('Kx GLOBAL')
      ax2.set_ylabel('Ky GLOBAL')
      ax2.set_title('OFF Weight Patches')
      ax2.format_coord = format_coord
      ax2.imshow(im3, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)


      plt.show()

      t = input("1 to repeat, 0 to end: ")
