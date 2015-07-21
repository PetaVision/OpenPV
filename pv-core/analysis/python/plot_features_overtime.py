"""
Plots the time stability
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

if len(sys.argv) < 2:
   print "usage: time_stability filename"
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

   maxp = np.max(fcomp) / 4.0
   return maxp

   #if maxp < 0.7:
   #   return 10
   #else:
   #   return maxp

   #if maxp == f1:
      #print "f1"
   #   result.append(1)
   #if maxp == f2:
      #print "f2"
   #   result.append(2)
   #if maxp == f3:
      #print "f3"
   #   result.append(3)
   #if maxp == f4:
      #print "f4"
   #   result.append(4)
   #if maxp == f5:
      #print "f5"
   #   result.append(5)
   #if maxp == f6:
      #print "f6"
   #   result.append(6)
   #if maxp == f7:
      #print "f7"
   #   result.append(7)
   #if maxp == f8:
      #print "f8"
   #  result.append(8)

   #return result


w = rw.PVReadWeights(sys.argv[1])
w.rewind()

numpat = w.numPatches 
count = 0

exp = []
exppn = []

body = w.recSize + 4
hs = w.headerSize
filesize = os.path.getsize(sys.argv[1])
bint = filesize / body

#f = open(sys.argv[1], 'r')
#print "tell = ", f.tell()
#print "body = ", body
#print "hs = ", hs
#sys.exit()

print
print "Number of steps = ", bint
forwardjump = input('How many steps forward:')

bint = bint - forwardjump

if forwardjump == 0:
   4
else:
   leap = ((body * forwardjump) + (100 * forwardjump))
   w.file.seek(leap, os.SEEK_CUR)


totala = 0
count = 0

for i in range(bint):
   if i == 0:
      count2 = 0
      for j in range(numpat):
         p = w.next_patch()
         if len(p) == 0:
            print"STOPPED EARLY"
            sys.exit()
         a = whatFeature(p)
         totala += a
      vtotal = [totala / numpat]
      count += 1
   else:
      count2 = 0
      totala = 0
      prejump = hs
      w.file.seek(prejump, os.SEEK_CUR)
      for j in range(numpat):
         p = w.next_patch()
         test = p
         if len(test) == 0:
            print "stop"
            sys.exit()
         a = whatFeature(p)
         totala += a
      pret = totala / numpat
      vtotal = np.append(vtotal, pret)
      count += 1
   print "count = ", count

print vtotal
print np.shape(vtotal)



fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(np.arange(len(vtotal)), vtotal, '-o', color='y')

ax.set_xlabel('Time')
ax.set_ylabel('Feature Strength')
ax.set_title('Feature Strength Overtime')
ax.set_ylim(0, 1)
ax.set_xlim(0, len(vtotal))
ax.grid(True)

plt.show()
