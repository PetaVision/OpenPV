"""
Plots the Histogram
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
   print "usage: time_stability filename"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])
wOff = rw.PVReadWeights(sys.argv[2])
w2 = rw.PVReadWeights(sys.argv[3])


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
supereasytest = 1


# create feature list for comparing weights from on and off cells
f = np.zeros(w.patchSize)
f2 = np.zeros(w.patchSize)
fe1 = []
fe2 = []
fe3 = []
fe4 = []
fe5 = []
fe6 = []
fe7 = []
fe8 = []
fcomp = []

f = w.normalize(f)
f2 = w.normalize(f2)


# vertical lines from right side
f = np.zeros([w.nxp, w.nyp]) # first line
f[:,0] = 1
fe1.append(f)

f = np.zeros([w.nxp, w.nyp]) # second line
f[:,1] = 1
fe2.append(f)

f2 = np.zeros([w.nxp, w.nyp]) # third line
f2[:,2] = 1
fe3.append(f2)

f = np.zeros([w.nxp, w.nyp])
f[:,3] = 1
fe4.append(f)

#horizontal lines from the top
f = np.zeros([w.nxp, w.nyp])
f[0,:] = 1
fe5.append(f)

f = np.zeros([w.nxp, w.nyp])
f[1,:] = 1
fe6.append(f)

f = np.zeros([w.nxp, w.nyp])
f[2,:] = 1
fe7.append(f)

f = np.zeros([w.nxp, w.nyp])
f[3,:] = 1
fe8.append(f)

#print "f8", fe8
#print "f7", fe7
#print "f6", fe6
#print "f5", fe5
#print "f4", fe4
#print "f3", fe3
#print "f2", fe2
#print "f1", fe1


def whatFeature(k):
   result = []
   fcomp = []
   k = np.reshape(k,(nxp,nyp))

   f1 = k * fe1
   f1 = np.sum(f1)
   fcomp.append(f1)
   #print f1

   f2 = k * fe2
   f2 = np.sum(f2)
   #print f2
   fcomp.append(f2)

   f3 = k * fe3
   f3 = np.sum(f3)
   #print f3
   fcomp.append(f3)

   f4 = k * fe4
   f4 = np.sum(f4)
   #print f4
   fcomp.append(f4)

   f5 = k * fe5
   f5 = np.sum(f5)
   #print f5
   fcomp.append(f5)

   f6 = k * fe6
   f6 = np.sum(f6)
   #print f6
   fcomp.append(f6)

   f7 = k * fe7
   f7 = np.sum(f7)
   #print f7
   fcomp.append(f7)

   f8 = k * fe8
   f8 = np.sum(f8)
   #print f8
   fcomp.append(f8)

   fcomp = np.array(fcomp)
   t = fcomp.argmax()
   check = fcomp.max() / 4  
   if check > 0.0:
      1
   else:
      result = [10]
      return result

   maxp = np.max(fcomp)

   if maxp == f1:
      #print "f1"
      result.append(1)
   if maxp == f2:
      #print "f2"
      result.append(2)
   if maxp == f3:
      #print "f3"
      result.append(3)
   if maxp == f4:
      #print "f4"
      result.append(4)
   if maxp == f5:
      #print "f5"
      result.append(5)
   if maxp == f6:
      #print "f6"
      result.append(6)
   if maxp == f7:
      #print "f7"
      result.append(7)
   if maxp == f8:
      #print "f8"
     result.append(8)
   if len(result) > 1:
      q = len(result)
      q = q-1
      ri = random.randint(1,q)
      op = result[ri]
      result = []
      result.append(op)

   return result





if 1 == 1:
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
   ax.text( -5, 12, "Percent %.2f   %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f" %(kcountper1,  kcountper2,  kcountper3, kcountper4, kcountper5, kcountper6, kcountper7, kcountper8, kcountper9, kcountper10, kcountper11, kcountper12, kcountper13, kcountper14, kcountper15, kcountper16), fontsize='large', rotation='horizontal')
   ax.text(-4, 14, "Patch   1      2       3       4       5       6       7       8       9      10      11     12     13     14     15     16", fontsize='x-large', rotation='horizontal')

   ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

   plt.show()

   t = 1

   nx = w.nx
   ny = w.ny
   nxp = w.nxp
   nyp = w.nyp
   numpat = w.numPatches
   nf = w.nf
   margin = 32
   marginstart = margin
   marginend = nx - margin
   acount = 0
   patchposition = []
   supereasytest = 1




   if 1 == 1:

      feature = 0

      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         print "qi = ", qi
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      print "prefinal = ", prefinal
      print "count2 = ", count2
      sys.exit()
      prefinal = np.average(prefinal, axis=0)
      postfinal = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal = np.append(postfinal, prefinal[b])
      feature += 1
#######################################################


      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal2 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal2 = np.append(postfinal2, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal3 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal3 = np.append(postfinal3, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal4 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal4 = np.append(postfinal4, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal5 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal5 = np.append(postfinal5, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal6 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal6 = np.append(postfinal6, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal7 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal7 = np.append(postfinal7, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal8 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal8 = np.append(postfinal8, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal9 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal9 = np.append(postfinal9, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal10 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal10 = np.append(postfinal10, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal11 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal11 = np.append(postfinal11, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal12 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal12 = np.append(postfinal12, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal13 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal13 = np.append(postfinal13, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal14 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal14 = np.append(postfinal14, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal15 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal15 = np.append(postfinal15, prefinal[b])
      feature += 1
#######################################################
      count = 0
      d = np.zeros((nxp,nyp))
      where = []
      zep = []
      where2 = []
      zep2 = []
      w = rw.PVReadWeights(sys.argv[1])
      w.rewind()
      for ko in np.arange(numpat):
         kxOn = conv.kxPos(ko, nx, ny, nf)
         kyOn = conv.kyPos(ko, nx, ny, nf)
         p = w.next_patch()
         afz = whatFeature(p)
         zep.append(afz)
         if marginstart < kxOn < marginend:
            if marginstart < kyOn < marginend:
               if cluster[count] == feature:
                  a = np.array(whatFeature(p))
                  a = np.array(a)
                  a = a[0]
                  where.append(a)  
                  e = p
                  e = e.reshape(nxp, nyp)
                  numrows, numcols = e.shape
                  count = count + 1
               else:
                  e = d
                  count = count + 1
                  where.append(10)
            else:
               e = d
               where.append(10)
         else:
            e = d
            where.append(10)
         x = (nxp) * (ko % nx)
         y = ( nyp) * (ko / nx)
         im2[y:y+nyp, x:x+nxp] = e
      count = 0
      wherebox = where
      wherebox = np.reshape(wherebox, (nx,ny))
      prefinal = []
      prefinal = np.array(prefinal)
      for o in range(8):
         o += 1
         thefeature = o
         count2 = 0
         qi = np.zeros((1,17))
         for k in range(numpat):
            kx = conv.kxPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if marginstart < kx < marginend:
               if marginstart < ky < marginend:
                  if where[k] == thefeature:
                     howmany = [1]
                     w = [0, 1]
                     for i in range(16):
                        i+=1
                        box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                        count = 0
                        for g in range(len(box)):
                           for h in range(len(box)):
                              if box[g,h] == thefeature:
                                 count+=1
                                 q = count
                        w = np.append(w, q)
                        q = q - w[-2]
                        q = q / float((i*8))
                        howmany = np.append(howmany, q)
                     count2 += 1.0
                     qi = np.add(qi, howmany)
         qi = qi / count2
         if o == 1:
            prefinal = qi
         else:
            prefinal = np.vstack((prefinal, qi))
      prefinal = np.average(prefinal, axis=0)
      postfinal16 = []
      for b in range(len(prefinal)):
         if b > 0:
            postfinal16 = np.append(postfinal16, prefinal[b])
      feature += 1




#######################################################




      fig = plt.figure()
      ax = fig.add_subplot(111, axisbg='darkslategray')
      ax.set_xlabel('Distance\n with-inhib=Yellow')
      ax.set_ylabel('Number of Shared Features')
      ax.set_title('proximity')
      ax.plot((np.arange(len(postfinal))+1), postfinal, "-o", color=cm.Paired(0.0))
      ax.plot((np.arange(len(postfinal2))+1), postfinal2, "-o", color=cm.Paired(0.06))
      ax.plot((np.arange(len(postfinal3))+1), postfinal3, "-o", color=cm.Paired(0.12))
      ax.plot((np.arange(len(postfinal4))+1), postfinal4, "-o", color=cm.Paired(0.18))
      ax.plot((np.arange(len(postfinal5))+1), postfinal5, "-o", color=cm.Paired(0.24))
      ax.plot((np.arange(len(postfinal6))+1), postfinal6, "-o", color=cm.Paired(0.3))
      ax.plot((np.arange(len(postfinal7))+1), postfinal7, "-o", color=cm.Paired(0.36))
      ax.plot((np.arange(len(postfinal8))+1), postfinal8, "-o", color=cm.Paired(0.42))
      ax.plot((np.arange(len(postfinal9))+1), postfinal9, "-o", color=cm.Paired(0.48))
      ax.plot((np.arange(len(postfinal10))+1), postfinal10, "-o", color=cm.Paired(0.54))
      ax.plot((np.arange(len(postfinal11))+1), postfinal11, "-o", color=cm.Paired(0.6))
      ax.plot((np.arange(len(postfinal12))+1), postfinal12, "-o", color=cm.Paired(0.66))
      ax.plot((np.arange(len(postfinal13))+1), postfinal13, "-o", color=cm.Paired(0.72))
      ax.plot((np.arange(len(postfinal14))+1), postfinal14, "-o", color=cm.Paired(0.78))
      ax.plot((np.arange(len(postfinal15))+1), postfinal15, "-o", color=cm.Paired(0.84))
      ax.plot((np.arange(len(postfinal16))+1), postfinal16, "-o", color=cm.Paired(0.9))
###







      plt.show()

#end fig loop



















#############################
      """
      fig = plt.figure()
      ax = fig.add_subplot(111)

      ax.set_xlabel('Kx GLOBAL')
      ax.set_ylabel('Ky GLOBAL')
      ax.set_title('ON Weight Patches')
      ax.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
   
   
      fig2 = plt.figure()
      ax2 = fig2.add_subplot(111)

      ax2.set_xlabel('Kx GLOBAL')
      ax2.set_ylabel('Ky GLOBAL')
      ax2.set_title('OFF Weight Patches')
      ax2.imshow(im3, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)


      plt.show()
      """
      t = input("1 to repeat, 0 to end: ")
   
