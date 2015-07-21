"""
Plot a histogram of weight clique size
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import PVReadWeights as rw
import PVConversions as conv

if len(sys.argv) < 2:
   print "usage: plot_clique_histogram filename [w_split_val]"
   exit()

count = 4
w_split_val = 255/2.

if len(sys.argv) >= 3:
   count = int(sys.argv[2])

if len(sys.argv) >= 4:
   w_split_val = int(sys.argv[3])

w = rw.PVReadWeights(sys.argv[1])

for k in range(w.numPatches):
   b = w.next_patch_bytes()
   csize = w.clique_size(b, count)
   if csize == count:
      nxg = w.nxGlobal
      nyg = w.nyGlobal
      nxb = w.nxprocs
      nyb = w.nyprocs
      kx = conv.kxBlockedPos(k, nxg, nyg, w.nf, nxb, nyb)
      ky = conv.kyBlockedPos(k, nxg, nyg, w.nf, nxb, nyb)

l = w.clique_locations(count, w_split_val)

print "number of neurons is ", len(l[0,:])
if len(l[0,:]) < 100: print l

fig = plt.figure()
ax = fig.add_subplot(111, axisbg='darkslategray')

ax.plot(l[0,:], l[1,:], 'o', color='y')

ax.set_xlabel('KX GLOBAL')
ax.set_ylabel('KY GLOBAL')
ax.set_title('Location of cliques of size ' + str(count))
ax.set_xlim(0, w.nxGlobal)
ax.set_ylim(0, w.nyGlobal)
ax.grid(True)

plt.show()
