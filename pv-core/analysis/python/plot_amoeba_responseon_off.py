"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import matplotlib.image as mpimg
import PVReadWeights as rw
import PVReadSparse as rs
import math

"""
mi=mpimg.imread(sys.argv[3])
imgplot = plt.imshow(mi, interpolation='Nearest')
imgplot.set_cmap('hot')
plt.show()
"""

def nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post):
   a = math.pow(2.0, (zScaleLog2Pre - zScaleLog2Post))
   ia = a

   if ia < 2:
      k0 = 0
   else:
      k0 = ia/2 - 1

   if a < 1.0 and kzPre < 0:
      k = kzPre - (1.0/a) + 1
   else:
      k = kzPre

   return k0 + (a * k)


def zPatchHead(kzPre, nzPatch, zScaleLog2Pre, zScaleLog2Post):
   a = math.pow(2.0, (zScaleLog2Pre - zScaleLog2Post))

   if a == 1:
      shift = -(0.5 * nzPatch)
      return shift + nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post)

   shift = 1 - (0.5 * nzPatch)

   if (nzPatch % 2) == 0 and a < 1:
      kpos = (kzPre < 0)

      if kzPre < 0:
         kpos = -(1+kzPre)
      else:
         kpos = kzPre

      l = (2*a*kpos) % 2
      if kzPre < 0:
         shift -= l == 1
      else:
         shift -= l == 0
   elif (nzPatch % 2) == 1 and a < 1:
      shift = -(0.5 * nzPatch)

   neighbor = nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post)

   if nzPatch == 1:
      return neighbor

   return shift + neighbor
"""
a = zPatchHead(int(sys.argv[1]), 5, -math.log(4, 2), -math.log(1, 2))
print a
print int(a)
sys.exit()
"""



vmax = 100.0 # Hz
space = 1
extended = False


w = rw.PVReadWeights(sys.argv[1])
wOff = rw.PVReadWeights(sys.argv[2])

sw = rw.PVReadWeights(sys.argv[3])
swOff = rw.PVReadWeights(sys.argv[4])

nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp


nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space

predub = np.zeros(((nx*nx),(nxp * nxp)))
predubOff = np.zeros(((nx*nx),(nxp * nxp)))
spredub = np.zeros(((nx*nx),(nxp * nxp)))
spredubOff = np.zeros(((nx*nx),(nxp * nxp)))


numpat = w.numPatches
print "numpat = ", numpat

for k in range(numpat):
   p = w.next_patch()
   pOff = wOff.next_patch()
   sp = sw.next_patch()
   spOff = swOff.next_patch()

   predub[k] = p
   predubOff[k] = pOff
   spredub[k] = sp
   spredubOff[k] = spOff

print "weights done"

#print "p = ", P
#if k == 500:
#   sys.exit()


#end fig loop




activ = rs.PVReadSparse(sys.argv[5], extended)
sactiv = rs.PVReadSparse(sys.argv[6], extended)


end = int(sys.argv[7])
step = int(sys.argv[8])
begin = int(sys.argv[9])

count = 0
for end in range(begin+step, end, step):
   A = activ.avg_activity(begin, end)
   sA = sactiv.avg_activity(begin, end)


   this = 10 + count
   count += 1
   print "this = ", this
   print "file = ", sys.argv[this]
   print
   numrows, numcols = A.shape

   min = np.min(A)
   max = np.max(A)

   s = np.zeros(numcols)
   for col in range(numcols):
       s[col] = np.sum(A[:,col])
   s = s/numrows

   b = np.reshape(A, (len(A)* len(A)))

   c = np.shape(b)[0]
   
   mi=mpimg.imread(sys.argv[this])


   print "a w start"
   rr = nx / 64
   im = np.zeros((64, 64))
   ims = np.zeros((64, 64))


   for yi in range(len(A)):
      for xi in range(len(A)):
         x = int(zPatchHead(int(xi), 5, -math.log(rr, 2), -math.log(1, 2)))
         y = int(zPatchHead(int(yi), 5, -math.log(rr, 2), -math.log(1, 2)))
         if 58 > x >= 0 and 58 > y >= 0:
            if A[yi, xi] > 0:
               patch = predub[yi * (nx) + xi]
               patchOff = predubOff[yi * (nx) + xi]
               spatch = spredub[yi * (nx) + xi]
               spatchOff = spredubOff[yi * (nx) + xi]

               patch = np.reshape(patch, (nxp, nxp))
               patchOff = np.reshape(patchOff, (nxp, nxp))
               spatch = np.reshape(spatch, (nxp, nxp))
               spatchOff = np.reshape(spatchOff, (nxp, nxp))
               for yy in range(nyp):
                  for xx in range(nxp):
                     im[y + yy, x + xx] += patch[yy, xx] * A[yi, xi]
                     im[y + yy, x + xx] -= patchOff[yy, xx] * A[yi, xi]

                     ims[y + yy, x + xx] += spatch[yy, xx] * sA[yi, xi]
                     ims[y + yy, x + xx] -= spatchOff[yy, xx] * sA[yi, xi]



   fig = plt.figure()
   ax = fig.add_subplot(3,1,1)

   ax.imshow(mi, interpolation='Nearest', cmap='gray')

   ax = fig.add_subplot(3,1,2)
   #ax.imshow(mi, interpolation='Nearest', cmap='gray', origin="lower")
   ax.set_xlabel('regular')
   ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin = 0.0, vmax = np.max(im))
   

   ax = fig.add_subplot(313)
   ax.set_xlabel('scrambled')
   ax.imshow(ims, cmap=cm.jet, interpolation='nearest', vmin = 0.0, vmax = np.max(ims))






   plt.show()

#end fig loop
