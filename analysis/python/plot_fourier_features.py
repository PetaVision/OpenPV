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
import radialProfile
import pylab as py
import random

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
      rl = len(result)
      ri = random.randint(0, rl)
      wn = result[ri-1]
      result = []
      result.append(wn)
   return result


space = 1

w = rw.PVReadWeights(sys.argv[1])
coord = 1
coord = int(coord)

nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = 32
marginstart = margin
marginend = nx - margin



nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.

where = []
zep = []


for k in range(numpat):
   kx = conv.kxPos(k, nx, ny, nf)
   ky = conv.kyPos(k, nx, ny, nf)
   p = w.next_patch()
   afz = whatFeature(p)
   zep.append(afz)
   if len(p) != nxp * nyp:
      continue
   if marginstart < kx < marginend:
      if marginstart < ky < marginend: 
         a = whatFeature(p)
         a = a[0]
         if a != 10:
            where.append(a)  



e = len(where)

d = math.sqrt(e)


wi = np.reshape(where, (d, d))


F1 = np.fft.fft2(wi)
F2 = np.fft.fftshift(F1)
psd2D = np.abs(F2)**2
psd1D = radialProfile.azimuthalAverage(psd2D)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.log10(wi), cmap=py.cm.Greys)



fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(np.log10(psd2D))

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.semilogy(psd1D)
ax3.set_xlabel('Spatial Frequency')
ax3.set_ylabel('Power Spectrum')

plt.show()


sys.exit()













def ratio(k):
   k = np.reshape(k,(nxp,nyp))

   v = float(np.max(np.sum(k, axis=0)))
   h = float(np.max(np.sum(k, axis=1)))
   vert = 2
   hori = 1
   fall = 0

   if (v/h) >= 1.45:
      result = vert
   elif (v/h) <= 0.45:
      result = hori
   else:
      result = fall
   return result


d = []
k = 16
for ko in range(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         acount += 1
         don = ratio(p)
         d = np.append(d, don)

vcount = 0
hcount = 0
fcount = 0
for i in range(len(d)):
   if d[i] == 2:
      vcount+=1
   if d[i] == 1:
      hcount+=1
   if d[i] == 0:
      fcount+=1

sys.exit()
#######################################

if 1 == 1:
   
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

count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
count9 = 0
count10 = 0
count11 = 0
count12 = 0
count13 = 0
count14 = 0
count15 = 0
count16 = 0

im1 = []
im2 = []
im3 = []
im4 = []
im5 = []
im6 = []
im7 = []
im8 = []
im9 = []
im10 = []
im11 = []
im12 = []
im13 = []
im14 = []
im15 = []
im16 = []

feature = 0
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count1+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im1 = np.append(im1, e)

feature += 1

#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count2+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im2 = np.append(im2, e)

feature += 1

#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count3+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im3 = np.append(im3, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count4+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im4 = np.append(im4, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count5+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im5 = np.append(im5, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count6+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im6 = np.append(im6, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count7+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im7 = np.append(im7, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count8+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im8 = np.append(im8, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count9+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im9 = np.append(im9, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count10+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im10 = np.append(im10, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count11+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im11 = np.append(im11, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count12+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im12 = np.append(im12, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count13+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im13 = np.append(im13, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count14+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im14 = np.append(im14, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count15+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im15 = np.append(im15, e)

feature += 1
#######
cf = np.matrix(con[feature])
count = 0
d = np.zeros((nxp,nyp))
d = 0
w.rewind()
for ko in np.arange(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         if cluster[count] == feature:
            e = np.matrix(p)
            e = e.reshape(16, 1)
            e = cf * e
            count = count + 1
            count16+=1
         else:
            e = d
            count = count + 1
      else:
         e = d
   else:
      e = d
   im16 = np.append(im16, e)

feature += 1

numpat = float(numpat)

im1 = (im1 * count1) / numpat
im2 = (im2 * count2) / numpat
im3 = (im3 * count3) / numpat
im4 = (im4 * count4) / numpat
im5 = (im5 * count5) / numpat
im6 = (im6 * count6) / numpat
im7 = (im7 * count7) / numpat
im8 = (im8 * count8) / numpat
im9 = (im9 * count9) / numpat
im10 = (im10 * count10) / numpat
im11 = (im11 * count11) / numpat
im12 = (im12 * count12) / numpat
im13 = (im13 * count13) / numpat
im14 = (im14 * count14) / numpat
im15 = (im15 * count15) / numpat
im16 = (im16 * count16) / numpat

totalim = im1
totalim = np.vstack((totalim, im2))
totalim = np.vstack((totalim, im3))
totalim = np.vstack((totalim, im4))
totalim = np.vstack((totalim, im5))
totalim = np.vstack((totalim, im6))
totalim = np.vstack((totalim, im7))
totalim = np.vstack((totalim, im8))
totalim = np.vstack((totalim, im9))
totalim = np.vstack((totalim, im10))
totalim = np.vstack((totalim, im11))
totalim = np.vstack((totalim, im12))
totalim = np.vstack((totalim, im13))
totalim = np.vstack((totalim, im14))
totalim = np.vstack((totalim, im15))
totalim = np.vstack((totalim, im16))



totalim = np.average(totalim, axis=0)
print "shape = ", np.shape(totalim)

totalim = np.reshape(totalim, (128, 128))

F1 = np.fft.fft2(totalim)
F2 = np.fft.fftshift(F1)
psd2D = np.abs(F2)**2
psd1D = radialProfile.azimuthalAverage(psd2D)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.log10(totalim), cmap=py.cm.Greys)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(np.log10(psd2D))

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.semilogy(psd1D)
ax3.set_xlabel('Spatial Frequency')
ax3.set_ylabel('Power Spectrum')

plt.show()

sys.exit()

