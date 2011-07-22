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


space = 1


nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = 15
marginstart = margin
marginend = nx - margin
acount = 0
patchposition = []
supereasytest = 1

d = np.zeros((nxp,nyp))


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
   if check > 0.7:
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
      rl = len(result)
      ri = random.randint(0, rl)
      wn = result[ri-1]
      result = []
      result.append(wn)
      #print "result = ", result
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

start = margin
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
         acount+=1
         a = whatFeature(p)
         a = a[0]
         if a != 10:
            where.append(a)
            #print a


count = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0



for i in range(len(where)):
   if where[i] == 1:
      count1 += 1
   if where[i] == 2:
      count2 += 1
   if where[i] == 3:
      count3 += 1
   if where[i] == 4:
      count4 += 1
   if where[i] == 5:
      count5 += 1
   if where[i] == 6:
      count6 += 1
   if where[i] == 7:
      count7 += 1
   if where[i] == 8:
      count8 += 1


h = [count1, count2, count3, count4, count5, count6, count7, count8]
h2 = [0, count1, count2, count3, count4, count5, count6, count7, count8]

hmax = np.max(h)
hmin = np.min(h)
hratio = float(hmax)/hmin

ptotal = len(where) / float(acount)

print "hmax = ", hmax
print "hmin = ", hmin
print "hratio = ", hratio
print "% of total = ", ptotal

fig = plt.figure()
ax = fig.add_subplot(111)
loc = np.array(range(len(h)))+0.5
width = 1.0
ax.set_title('Feature Histogram')
ax.set_xlabel('Total Number of Features = %1.0i \n ratio = %f \n percent of total = %f' %(len(where), hratio, ptotal)) 
ax.bar(loc, h, width=width, bottom=0, color='b')
#ax.plot(np.arange(len(h2)), h2, ls = '-', marker = 'o', color='b', linewidth = 4.0)

plt.show()
