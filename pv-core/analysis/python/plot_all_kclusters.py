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

if len(sys.argv) < 2:
   print "usage: kclustering filename on, filename off, k"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])


if len(sys.argv) == 3:
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


   k = 24
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


   k = 16
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

###
   nx_im2 = nx * (nxp)
   ny_im2 = ny * (nyp)

   im2 = np.zeros((nx_im2, ny_im2))
   im2[:,:] = (w.max - w.min) / 2.
##
   nx_im3 = nx * (nxp)
   ny_im3 = ny * (nyp)

   im3 = np.zeros((nx_im3, ny_im3))
   im3[:,:] = (w.max - w.min) / 2.
###
   nx_im4 = nx * (nxp)
   ny_im4 = ny * (nyp)

   im4 = np.zeros((nx_im4, ny_im4))
   im4[:,:] = (w.max - w.min) / 2.
###
   nx_im5 = nx * (nxp)
   ny_im5 = ny * (nyp)

   im5 = np.zeros((nx_im5, ny_im5))
   im5[:,:] = (w.max - w.min) / 2.
###
   nx_im6 = nx * (nxp)
   ny_im6 = ny * (nyp)

   im6 = np.zeros((nx_im6, ny_im6))
   im6[:,:] = (w.max - w.min) / 2.
###
   nx_im7 = nx * (nxp)
   ny_im7 = ny * (nyp)

   im7 = np.zeros((nx_im7, ny_im7))
   im7[:,:] = (w.max - w.min) / 2.
###
   nx_im8 = nx * (nxp)
   ny_im8 = ny * (nyp)

   im8 = np.zeros((nx_im8, ny_im8))
   im8[:,:] = (w.max - w.min) / 2.
###
   nx_im9 = nx * (nxp)
   ny_im9 = ny * (nyp)

   im9 = np.zeros((nx_im9, ny_im9))
   im9[:,:] = (w.max - w.min) / 2.
###
   nx_im10 = nx * (nxp)
   ny_im10 = ny * (nyp)

   im10 = np.zeros((nx_im10, ny_im10))
   im10[:,:] = (w.max - w.min) / 2.
###
   nx_im11 = nx * (nxp)
   ny_im11 = ny * (nyp)

   im11 = np.zeros((nx_im11, ny_im11))
   im11[:,:] = (w.max - w.min) / 2.
###
   nx_im12 = nx * (nxp)
   ny_im12 = ny * (nyp)

   im12 = np.zeros((nx_im12, ny_im12))
   im12[:,:] = (w.max - w.min) / 2.
###
   nx_im13 = nx * (nxp)
   ny_im13 = ny * (nyp)

   im13 = np.zeros((nx_im13, ny_im13))
   im13[:,:] = (w.max - w.min) / 2.
###
   nx_im14 = nx * (nxp)
   ny_im14 = ny * (nyp)

   im14 = np.zeros((nx_im14, ny_im14))
   im14[:,:] = (w.max - w.min) / 2.
###
   nx_im15 = nx * (nxp)
   ny_im15 = ny * (nyp)

   im15 = np.zeros((nx_im15, ny_im15))
   im15[:,:] = (w.max - w.min) / 2.
###
   nx_im16 = nx * (nxp)
   ny_im16 = ny * (nyp)

   im16 = np.zeros((nx_im16, ny_im16))
   im16[:,:] = (w.max - w.min) / 2.
###
   nx_im17 = nx * (nxp)
   ny_im17 = ny * (nyp)

   im17 = np.zeros((nx_im17, ny_im17))
   im17[:,:] = (w.max - w.min) / 2.








  
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
   ax.text( -5, 12, "Percent %.2f   %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f" %(kcountper1,  kcountper2,  kcountper3, kcountper4, kcountper5, kcountper6, kcountper7, kcountper8, kcountper9, kcountper10, kcountper11, kcountper12, kcountper13, kcountper14, kcountper15, kcountper16), fontsize='large', rotation='horizontal')
   ax.text(-4, 14, "Patch   1      2       3       4       5       6       7       8       9      10      11     12     13     14     15     16", fontsize='x-large', rotation='horizontal')

   ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

   plt.show()

   t = 1

   feature = 0

#######
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
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################

#######
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
      im3[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im4[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im5[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im6[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im7[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im8[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im9[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im10[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im11[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im12[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im13[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im14[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im15[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im16[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################
#######
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
      im17[y:y+nyp, x:x+nxp] = e
   #count = 0
   #wOff.rewind()
   #for ko in np.arange(numpat):
   #   kxOn = conv.kxPos(ko, nx, ny, nf)
   #   kyOn = conv.kyPos(ko, nx, ny, nf)
   #   p = wOff.next_patch()
   #   if marginstart < kxOn < marginend:
   #      if marginstart < kyOn < marginend:
   #         if cluster[count] == feature:
   #            e = p
   #            e = e.reshape(nxp, nyp)
   #            numrows, numcols = e.shape
   #            count = count + 1
   #         else:
   #            e = d
   #            count = count + 1
   #      else:
   #         e = d
   #   else:
   #      e = d
   #   x = (nxp) * (ko % nx)
   #   y = (nyp) * (ko / nx)
   #   im3[y:y+nyp, x:x+nxp] = e
   #   feature += 1
##################################

   fig = plt.figure()
   ax = fig.add_subplot(111)

   ax.set_xlabel('Kx GLOBAL')
   ax.set_ylabel('Ky GLOBAL')
   ax.set_title('K-cluster 1')
   ax.format_coord = format_coord
   ax.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
   
###   
   fig2 = plt.figure()
   ax2 = fig2.add_subplot(111)
   ax2.set_xlabel('Kx GLOBAL')
   ax2.set_ylabel('Ky GLOBAL')
   ax2.set_title('K-cluster 2')
   ax2.format_coord = format_coord
   ax2.imshow(im3, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig3 = plt.figure()
   ax3 = fig3.add_subplot(111)
   ax3.set_xlabel('Kx GLOBAL')
   ax3.set_ylabel('Ky GLOBAL')
   ax3.set_title('K-cluster 3')
   ax3.format_coord = format_coord
   ax3.imshow(im4, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig4 = plt.figure()
   ax4 = fig4.add_subplot(111)
   ax4.set_xlabel('Kx GLOBAL')
   ax4.set_ylabel('Ky GLOBAL')
   ax4.set_title('k-cluster 4')
   ax4.format_coord = format_coord
   ax4.imshow(im5, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig5 = plt.figure()
   ax5 = fig5.add_subplot(111)
   ax5.set_xlabel('Kx GLOBAL')
   ax5.set_ylabel('Ky GLOBAL')
   ax5.set_title('k-cluster 5')
   ax5.format_coord = format_coord
   ax5.imshow(im6, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig6 = plt.figure()
   ax6 = fig6.add_subplot(111)
   ax6.set_xlabel('Kx GLOBAL')
   ax6.set_ylabel('Ky GLOBAL')
   ax6.set_title('k-cluster 6')
   ax6.format_coord = format_coord
   ax6.imshow(im7, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig7 = plt.figure()
   ax7 = fig7.add_subplot(111)
   ax7.set_xlabel('Kx GLOBAL')
   ax7.set_ylabel('Ky GLOBAL')
   ax7.set_title('k-cluster 7')
   ax7.format_coord = format_coord
   ax7.imshow(im8, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig8 = plt.figure()
   ax8 = fig8.add_subplot(111)
   ax8.set_xlabel('Kx GLOBAL')
   ax8.set_ylabel('Ky GLOBAL')
   ax8.set_title('k-cluster 8')
   ax8.format_coord = format_coord
   ax8.imshow(im9, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig9 = plt.figure()
   ax9 = fig9.add_subplot(111)
   ax9.set_xlabel('Kx GLOBAL')
   ax9.set_ylabel('Ky GLOBAL')
   ax9.set_title('k-cluster 9')
   ax9.format_coord = format_coord
   ax9.imshow(im10, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig10 = plt.figure()
   ax10 = fig10.add_subplot(111)
   ax10.set_xlabel('Kx GLOBAL')
   ax10.set_ylabel('Ky GLOBAL')
   ax10.set_title('k-cluster 10')
   ax10.format_coord = format_coord
   ax10.imshow(im11, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig11 = plt.figure()
   ax11 = fig11.add_subplot(111)
   ax11.set_xlabel('Kx GLOBAL')
   ax11.set_ylabel('Ky GLOBAL')
   ax11.set_title('k-cluster 11')
   ax11.format_coord = format_coord
   ax11.imshow(im12, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig12 = plt.figure()
   ax12 = fig12.add_subplot(111)
   ax12.set_xlabel('Kx GLOBAL')
   ax12.set_ylabel('Ky GLOBAL')
   ax12.set_title('k-cluster 12')
   ax12.format_coord = format_coord
   ax12.imshow(im13, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig13 = plt.figure()
   ax13 = fig13.add_subplot(111)
   ax13.set_xlabel('Kx GLOBAL')
   ax13.set_ylabel('Ky GLOBAL')
   ax13.set_title('k-cluster 13')
   ax13.format_coord = format_coord
   ax13.imshow(im14, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig14 = plt.figure()
   ax14 = fig14.add_subplot(111)
   ax14.set_xlabel('Kx GLOBAL')
   ax14.set_ylabel('Ky GLOBAL')
   ax14.set_title('k-cluster 14')
   ax14.format_coord = format_coord
   ax14.imshow(im15, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig15 = plt.figure()
   ax15 = fig15.add_subplot(111)
   ax15.set_xlabel('Kx GLOBAL')
   ax15.set_ylabel('Ky GLOBAL')
   ax15.set_title('k-cluster 15')
   ax15.format_coord = format_coord
   ax15.imshow(im16, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
###   
   fig16 = plt.figure()
   ax16 = fig16.add_subplot(111)
   ax16.set_xlabel('Kx GLOBAL')
   ax16.set_ylabel('Ky GLOBAL')
   ax16.set_title('k-cluster 16')
   ax16.format_coord = format_coord
   ax16.imshow(im17, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)




   plt.show()

