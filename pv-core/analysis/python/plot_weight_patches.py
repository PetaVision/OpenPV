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
      x = (x / 6.0)
      y = (y / 6.0)
      if col>=0 and col<numcols and row>=0 and row<numrows:
         z = P[row,col]
         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
      else:
         return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))

"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""

if len(sys.argv) < 3:
   print "usage: plot_weight_patches filename, 0 for regular or 1 for alternative coordanite system, 3 for l2 layer coordanite system"
   sys.exit()

space = 1

weights = rw.PVReadWeights(sys.argv[1])
coord = sys.argv[2]
coord = int(coord)

nx  = weights.nx
ny  = weights.ny
nxp = weights.nxp
nyp = weights.nyp

nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (weights.max - weights.min) / 2.

for k in range(weights.numPatches):
   P = weights.next_patch()
   if len(P) != nxp * nyp:
      continue

   #print "p = ", P
   #if k == 500:
   #   sys.exit()
   P = np.reshape(P, (nxp, nyp))
   numrows, numcols = P.shape

   x = space + (space + nxp) * (k % nx)
   y = space + (space + nyp) * (k / nx)

   im[y:y+nyp, x:x+nxp] = P

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title('Weight Patches')
ax.format_coord = format_coord

ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=weights.min, vmax=weights.max)

plt.show()

#end fig loop
