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
margin = 50
marginstart = margin
marginend = nx - margin
acount = 0
patchposition = []
supereasytest = 1

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
   if len(p) != nxp * nyp:
      continue
   if marginstart < kx < marginend:
      if marginstart < ky < marginend: 
         where.append([p])  
      else:
         where.append([p])
   else:
      where.append([p])




wherebox = where
wherebox = np.reshape(wherebox, (nx,ny, 25))



print "shape = ", np.shape(wherebox)


prefinal = []
prefinal = np.array(prefinal)

prefinal2 = []
tprefinal2 = np.array(prefinal2)


count2 = 0
qi = np.zeros((1,26))
for k in range(numpat):
   kx = conv.kxPos(k, nx, ny, nf)
   ky = conv.kyPos(k, nx, ny, nf)

   if marginstart < kx < marginend:
      if marginstart < ky < marginend:

         howmany = [1]
         w = [0, 1]
         a = np.matrix(wherebox[kx, ky])
         if np.sum(a) > 4.0:

            for i in range(25):
               i+=1
               box = wherebox[((ky-i)):((ky+1+i)), ((kx-i)):((kx+1+i))]
               count = 0
               qq = []
               countw = 0
               bleng = len(box)-2

               for g in range(bleng):
                  b = np.matrix(box[g+1,0])
                  q = (a * np.transpose(b)) / (math.sqrt(a*np.transpose(a))*math.sqrt(b*np.transpose(b)))
                  if countw == 0:
                     qq = np.append(qq, q)
                  else:
                     qq = np.add(qq, q)

                  countw+=1
                  b = np.matrix(box[g+1, (len(box)-1)])
                  q = (a * np.transpose(b)) / (math.sqrt(a*np.transpose(a))*math.sqrt(b*np.transpose(b)))
                  qq = np.add(qq, q)
                  countw+=1

               for h in range(len(box)):
                  b = np.matrix(box[0, h])
                  q = (a * np.transpose(b)) / (math.sqrt(a*np.transpose(a))*math.sqrt(b*np.transpose(b)))

                  qq = np.add(qq, q)
                  countw+=1

                  b = np.matrix(box[(len(box)-1), h])
                  q = (a * np.transpose(b)) / (math.sqrt(a*np.transpose(a))*math.sqrt(b*np.transpose(b)))

                  qq = np.add(qq, q)
                  countw+=1
               qq = qq / countw
               howmany = np.append(howmany, qq)
            count2 += 1.0

            qi = np.add(qi, howmany)
print
print "pre qi = ", qi
qi = qi / count2
print "count2 = ", count2 

print
print
qi = np.reshape(qi,(np.shape(qi)[1], 1))

print "qi = ", qi
print "qi shape = ", np.shape(qi)


fig = plt.figure()
ax = fig.add_subplot(111, axisbg='darkslategray')

ax.set_xlabel('Distance\n with-inhib=Yellow without-inhib=Red')
ax.set_ylabel('Number of Shared Features')
ax.set_title('proximity')

ax.plot((np.arange(len(qi))+1), qi, "-o", color='y')
#ax.plot((np.arange(len(postfinal2))+1), postfinal2, "-o", color='r')
#ax.plot(np.arange(len(prefinal[2])), prefinal[2], "-o", color=cm.spectral(0.4))
#ax.plot(np.arange(len(prefinal[3])), prefinal[3], "-o", color=cm.spectral(0.5))
#ax.plot(np.arange(len(prefinal[4])), prefinal[4], "-o", color=cm.spectral(0.6))
#ax.plot(np.arange(len(prefinal[5])), prefinal[5], "-o", color=cm.spectral(0.7))
#ax.plot(np.arange(len(prefinal[6])), prefinal[6], "-o", color=cm.spectral(0.8))
#ax.plot(np.arange(len(prefinal[7])), prefinal[7], "-o", color=cm.spectral(0.9))

ax.set_ylim(0.0, 1.0)

plt.show()

#end fig loop

