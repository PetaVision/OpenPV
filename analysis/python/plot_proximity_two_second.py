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
w2 = rw.PVReadWeights(sys.argv[2])


space = 1

d = np.zeros((5,5))

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = 40
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
fe9 = []
fe10 = []
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

f = np.zeros([w.nxp, w.nyp])
f[:,4] = 1
fe5.append(f)


#horizontal lines from the top
f = np.zeros([w.nxp, w.nyp])
f[0,:] = 1
fe6.append(f)

f = np.zeros([w.nxp, w.nyp])
f[1,:] = 1
fe7.append(f)

f = np.zeros([w.nxp, w.nyp])
f[2,:] = 1
fe8.append(f)

f = np.zeros([w.nxp, w.nyp])
f[3,:] = 1
fe9.append(f)

f = np.zeros([w.nxp, w.nyp])
f[4,:] = 1
fe10.append(f)

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

   f9 = k * fe9
   f9 = np.sum(f9)
   #print f8
   fcomp.append(f9)

   f10 = k * fe10
   f10 = np.sum(f10)
   #print f8
   fcomp.append(f10)


   fcomp = np.array(fcomp)
   t = fcomp.argmax()
   check = fcomp.max() / 5  
   if check > 0.0:
      1
   else:
      result = [20]
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
   if maxp == f9:
      #print "f8"
     result.append(9)
   if maxp == f10:
      #print "f8"
     result.append(10)

   if len(result) > 1:
      q = len(result)
      q = q-1
      ri = random.randint(1,q)
      op = result[ri]
      result = []
      result.append(op)

   return result


coord = 1
coord = int(coord)

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
         a = np.array(whatFeature(p))
         a = np.array(a)
         a = a[0]
         where.append(a)  
      else:
         a = np.array(whatFeature(p))
         a = np.array(a)
         a = a[0]
         
         where.append(a)
   else:
      a = np.array(whatFeature(p))
      a = np.array(a)
      a = a[0]

      where.append(a)

where2 = []
zep2 = []


for k in range(numpat):
   kx = conv.kxPos(k, nx, ny, nf)
   ky = conv.kyPos(k, nx, ny, nf)
   p = w2.next_patch()
   afz = whatFeature(p)
   zep2.append(afz)
   if len(p) != nxp * nyp:
      continue
   if marginstart < kx < marginend:
      if marginstart < ky < marginend: 
         a = np.array(whatFeature(p))
         a = np.array(a)
         a = a[0]
         where2.append(a)  
      else:
         a = np.array(whatFeature(p))
         a = np.array(a)
         a = a[0]

         where2.append(a)
   else:
      a = np.array(whatFeature(p))
      a = np.array(a)
      a = a[0]

      where2.append(a)











wherebox = where
wherebox = np.reshape(wherebox, (nx,ny))


wherebox2 = where2
wherebox2 = np.reshape(wherebox2, (nx,ny))

##
##
#test = 1422%nx

#   (k-i)%ny
#   (k+1+i)%ny  
#   (k-1)/nx
#   (k+1+i)/nx


#a = wherebox[2:4, 2:4]
#print a

#sys.exit()
prefinal = []
prefinal = np.array(prefinal)

prefinal2 = []
prefinal2 = np.array(prefinal2)


for o in range(10):
   o += 1
   thefeature = o

   count2 = 0
   qi = np.zeros((1,26))
   for k in range(numpat):
      kx = conv.kxPos(k, nx, ny, nf)
      ky = conv.kyPos(k, nx, ny, nf)

      if marginstart < kx < marginend:
         if marginstart < ky < marginend:
            if where[k] == thefeature:
               howmany = [1]
               w = [0, 1]
               for i in range(25):
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
                  q = q / float((i))
                  howmany = np.append(howmany, q)
               count2 += 1.0
               qi = np.add(qi, howmany)

   qi = qi / count2

   if o == 1:
      prefinal = qi
   else:
      prefinal = np.vstack((prefinal, qi))

prefinal = np.average(prefinal, axis=0)



for o in range(10):
   o += 1
   thefeature = o

   count2 = 0
   qi = np.zeros((1,26))
   for k in range(numpat):
      kx = conv.kxPos(k, nx, ny, nf)
      ky = conv.kyPos(k, nx, ny, nf)

      if marginstart < kx < marginend:
         if marginstart < ky < marginend:
            if where2[k] == thefeature:
               howmany = [1]
               w = [0, 1]
               for i in range(25):
                  i+=1
                  box = wherebox2[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
                  count = 0
                  for g in range(len(box)):
                     for h in range(len(box)):
                        if box[g,h] == thefeature:
                           count+=1
                           q = count
                  w = np.append(w, q)
                  q = q - w[-2]
                  q = q / float((i))
                  howmany = np.append(howmany, q)
               count2 += 1.0
               qi = np.add(qi, howmany)

   qi = qi / count2

   if o == 1:
      prefinal2 = qi
   else:
      prefinal2 = np.vstack((prefinal2, qi))

#print "1stprefinal", prefinal
#print "1stprefinal2", prefinal2


prefinal2 = np.average(prefinal2, axis=0)
#print "prefinal", prefinal
#print "prefinal2", prefinal2

postfinal = []
postfinal2 = []

for b in range(len(prefinal)):
   if b > 0:
      postfinal = np.append(postfinal, prefinal[b])
for b in range(len(prefinal2)):
   if b > 0:
      postfinal2 = np.append(postfinal2, prefinal2[b])



fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Distance')
ax.set_ylabel('Number of Shared Features')
ax.set_title('Proximity\n With Inhibition')
ax.set_ylim(0.105, 0.145)
ax.plot((np.arange(len(postfinal))+1), postfinal, "-o", color='b', linewidth=5.0)
#ax.plot(np.arange(len(prefinal[2])), prefinal[2], "-o", color=cm.spectral(0.4))
#ax.plot(np.arange(len(prefinal[3])), prefinal[3], "-o", color=cm.spectral(0.5))
#ax.plot(np.arange(len(prefinal[4])), prefinal[4], "-o", color=cm.spectral(0.6))
#ax.plot(np.arange(len(prefinal[5])), prefinal[5], "-o", color=cm.spectral(0.7))
#ax.plot(np.arange(len(prefinal[6])), prefinal[6], "-o", color=cm.spectral(0.8))
#ax.plot(np.arange(len(prefinal[7])), prefinal[7], "-o", color=cm.spectral(0.9))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

ax2.plot((np.arange(len(postfinal2))+1), postfinal2, "-o", color='b', linewidth=5.0)
ax2.set_xlabel('Distance\n')
ax2.set_ylabel('Number of Shared Features')
ax2.set_title('Proximity\n Without Inhibition')
ax2.set_ylim(0.105, 0.145)

plt.show()

#end fig loop
