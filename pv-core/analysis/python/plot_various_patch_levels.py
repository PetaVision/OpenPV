"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVConversions as conv
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

if len(sys.argv) < 3:
   print "usage: plot_weight_patches filename, 0 for regular or 1 for alternative coordanite system, 3 for l2 layer coordanite system"
   sys.exit()
coord = 3
space = 1

w = rw.PVReadWeights(sys.argv[1])


nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
nf = w.nf

patsize = nxp * nyp

nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space
small = w.max / 2


im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.



for k in range(w.numPatches):
   P = w.next_patch()
   if len(P) != nxp * nyp:
      continue
   #for k in range(patsize):

      #if P[k] < small:
      #   P[k] = 0.0

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

ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)



w2 = rw.PVReadWeights(sys.argv[2])

numpat = w2.numPatches
d = np.zeros((4,4))
nx  = w2.nx
ny  = w2.ny
nxp = w2.nxp
nyp = w2.nyp
nf = w2.nf

pref = input('enter 1 for single patches or 2 for patch set: ')


if pref == 1:

   con = 1
   while con == 1:
      w2.rewind()
      whichpatx = input("What is the x axis of the patch? ")
      whichpaty = input("What is the y axis of the patch? ")
      con = 0






      count = 0
      coord = 1


      nx_im2 = nx * (nxp + space) + space
      ny_im2 = ny * (nyp + space) + space

      im2 = np.zeros((nx_im2, ny_im2))
      im[:,:] = (w2.max - w2.min) / 2.



      for i in range(numpat):
         kx = conv.kxPos(i, nx, ny, nf)
         ky = conv.kyPos(i, nx, ny, nf)
         p = w2.next_patch()
         if kx == whichpatx:
            if ky == whichpaty:
               e = p
               e = e.reshape(nxp, nyp)
               numrows, numcols = e.shape
               count += 1
               im3 = e
            else:
               e = d
               count += 1
         else:
            e = d
            count += 1

         x = space + (space + nxp) * (i % nx)
         y = space + (space + nyp) * (i / nx)

         im2[y:y+nyp, x:x+nxp] = e


      #fig2 = plt.figure()
      #ax2 = fig2.add_subplot(111)

      #ax2.set_xlabel('Kx GLOBAL')
      #ax2.set_ylabel('Ky GLOBAL')
      #ax2.set_title('Weight Patches')
      #ax2.format_coord = format_coord

      #ax2.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w2.min, vmax=w2.max)

      fig3 = plt.figure()
      ax3 = fig3.add_subplot(111)
      ax3.set_title('Chosen Patch')
      ax3.imshow(im3, cmap=cm.jet, interpolation='nearest', vmin=w2.min, vmax=w2.max)
   
      con = input("1 to continue, 0 to stop: ")
   
   plt.show()
   con = 0

if pref == 2:
   numpat = w2.numPatches
   d = np.zeros((4,4))
   nx  = w2.nx
   ny  = w2.ny
   nxp = w2.nxp
   nyp = w2.nyp
   nf = w2.nf


   print
   print "enter x,y axes of the upper left patch in the l2 patch you want to look at"
   whichpatx = input("What is the x axis of the patch?")
   whichpaty = input("What is the y axis of the patch?")


   rangex = whichpatx + 15
   rangey = whichpaty + 15

   nx_im2 = nx * (nxp + space) + space
   ny_im2 = ny * (nyp + space) + space

   im2 = np.zeros((nx_im2, ny_im2))
   im2[:,:] = (w2.max - w2.min) / 2.0

   count = 0
   for i in range(numpat):
      kx = conv.kxPos(i, nx, ny, nf)
      ky = conv.kyPos(i, nx, ny, nf)
      p = w2.next_patch()
      if whichpatx <= kx < rangex:
         if whichpaty <= ky < rangey:
            e = p
            e = e.reshape(nxp, nyp)
            numrows, numcols = e.shape
            count += 1

         else:
            e = d
            count += 1
      else:
         e = d
         count += 1
      x = space + (space + nxp) * (i % nx)
      y = space + (space + nyp) * (i / nx)
      #print count
      im2[y:y+nyp, x:x+nxp] = e
   coord = 1
   fig2 = plt.figure()
   ax2 = fig2.add_subplot(111)
   ax2.format_coord = format_coord
   ax2.set_title('Chosen Patches')
   ax2.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w2.min, vmax=w2.max)
   plt.show()
