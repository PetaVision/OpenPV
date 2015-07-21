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

w = rw.PVReadWeights(sys.argv[1])


space = 0

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

coord = 1

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
      return t
   else:
      return 10

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


def format_coord(x, y):
   col = int(x+0.5)
   row = int(y+0.5)
   if coord == 3:
      check = ((x - 0.5) % 16)
      if check < 4:
         x2 = ((x - 0.5) % 16) - 7 + (x / 16.0)
         y2 = ((y - 0.5) % 16) - 7 + (y / 16.0) 
      elif check < 10:
         x2 = ((x - 0.5) % 16) - 7.5 + (x / 16.0)
         y2 = ((y - 0.5) % 16) - 7.5 + (y / 16.0) 
      else:
         x2 = ((x - 0.5) % 16) - 8 + (x / 16.0)
         y2 = ((y - 0.5) % 16) - 8 + (y / 16.0) 
      x = (x / 16.0)
      y = (y / 16.0)
      

      if col>=0 and col<numcols and row>=0 and row<numrows:
         z = P[row,col]
         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
      else:
         return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))      

   if coord == 1:
      x2 = (x / 20.0)
      y2 = (y / 20.0)
      x = (x / 5.0)
      y = (y / 5.0)
      if col>=0 and col<numcols and row>=0 and row<numrows:
         z = P[row,col]
         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
      else:
         return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))

"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""

if len(sys.argv) < 2:
   print "usage: plot_weight_patches filename, 0 for regular or 1 for alternative coordanite system, 3 for l2 layer coordanite system"
   sys.exit()


nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.

d = np.zeros((4,4))
o = np.zeros((16,1))


ff1 = o
for i in range(16):
   ff1[i] = 0.15
ff1 = ff1.reshape((4,4))

ff2 =  np.zeros((16,1))
for i in range(16):
   ff2[i] = 0.3
ff2 = ff2.reshape((4,4))

ff3 =  np.zeros((16,1))
for i in range(16):
   ff3[i] = 0.45
ff3 = ff3.reshape((4,4))

ff4 =  np.zeros((16,1))
for i in range(16):
   ff4[i] = 0.6
ff4 = ff4.reshape((4,4))

ff5 =  np.zeros((16,1))
for i in range(16):
   ff5[i] = 0.7
ff5 = ff5.reshape((4,4))

ff6 =  np.zeros((16,1))
for i in range(16):
   ff6[i] = 0.8
ff6 = ff6.reshape((4,4))

ff7 =  np.zeros((16,1))
for i in range(16):
   ff7[i] = 0.9
ff7 = ff7.reshape((4,4))

ff8 = np.zeros((16,1))
for i in range(16):
   ff8[i] = 1.0
ff8 = ff8.reshape((4,4))





count = 0
count2 = 0
for k in range(w.numPatches):
   p = w.next_patch()
   if len(p) != nxp * nyp:
      continue
   
   a = whatFeature(p)
   if a == 0:
      p = ff1
   if a == 1:
      p = ff2
   if a == 2:
      p = ff3
   if a == 3:
      p = ff4
   if a == 4:
      p = ff5
   if a == 5:
      p = ff6
   if a == 6:
      p = ff7
   if a == 7:
      p = ff8
   if a == 10:
      p = d
   count += 1
   #print count2
   #print p
   numrows, numcols = p.shape
   x = space + (space + nxp) * (k % nx)
   y = space + (space + nyp) * (k / nx)

   im[y:y+nyp, x:x+nxp] = p


fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL\n number of patches = %i' %(count))
ax.set_ylabel('Ky GLOBAL')
ax.set_title('Weight Patches')
ax.format_coord = format_coord
ax.text(550.0, 0.0, fe1, backgroundcolor = cm.spectral(0.15))
ax.text(550.0, 75.0, fe2, backgroundcolor = cm.spectral(0.3))
ax.text(550.0, 150.0, fe3, backgroundcolor = cm.spectral(0.45))
ax.text(550.0, 225.0, fe4, backgroundcolor = cm.spectral(0.6))
ax.text(550.0, 300.0, fe5, backgroundcolor = cm.spectral(0.7))
ax.text(550.0, 375.0, fe6, backgroundcolor = cm.spectral(0.8))
ax.text(550.0, 450.0, fe7, backgroundcolor = cm.spectral(0.9))
ax.text(550.0, 525.0, fe8, backgroundcolor = cm.spectral(1.0))



ax.imshow(im, cmap=cm.spectral, interpolation='nearest', vmin=0, vmax=1)

plt.show()

#end fig loop
