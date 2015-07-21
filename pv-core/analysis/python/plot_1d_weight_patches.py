"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw

"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""

if len(sys.argv) < 2:
   print "usage: plot_weight_patches filename, 0 for regular or 1 for alternative coordanite system, 3 for l2 layer coordanite system"
   sys.exit()

space = 2

weights = rw.PVReadWeights(sys.argv[1])

nx  = weights.nx
ny  = weights.ny
nxp = weights.nxp
nyp = weights.nyp
nyp = nxp
numpat = weights.numPatches

nx2 = numpat
ny2 = numpat 
nx1 = 1.

nx_im = nx1 * (nxp + space) + space
ny_im = ny2 * (nyp + space) + space


im = np.zeros((nx_im, ny_im))
im[:,:] = (weights.max - weights.min) / 2.


print numpat

for k in range(numpat):
   P = weights.next_patch()
   a = []
   for i in range(nxp):
      a = np.append(a,P)
   a = np.reshape(a, (nxp,nxp))

   P = a
   numrows, numcols = P.shape

   x = space + (space + nxp) * (k % nx2)
   y = space + (space) * (k / nx2)
   im[y:y+nyp, x:x+nxp] = P


fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title('Weight Patches')

ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=weights.min, vmax=weights.max)

plt.show()

#end fig loop
