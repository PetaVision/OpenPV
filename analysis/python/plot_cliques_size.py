"""
Plot a histogram of weight clique size
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import PVReadWeights as rw

if len(sys.argv) < 2:
   print "usage: plot_clique_histogram filename [w_split_val]"
   exit()

count = 4
w_split_val = 255/2.

if len(sys.argv) == 3:
   count = int(sys.argv[2])

w = rw.PVReadWeights(sys.argv[1])

fig = plt.figure()
ax = fig.add_subplot(111, axisbg='darkslategray')

#a = np.array([ (1,2,3,4), (1,2,3,4) ])

#x = (1,2,3,4)
#y1 = (2,2,2,2)
#y2 = (1,1,1,1)

#ax.plot(x, y1, 'o', color='y')
#ax.plot(x, y2, 'o', color='r')

#l = w.clique_locations(count, w_split_val)
#ax.plot(l[0,:], l[1,:], 'o', color='b')

#l = w.clique_locations(count+1, w_split_val)
#ax.plot(l[0,:], l[1,:], 'o', color='g')

#l = w.clique_locations(count+2, w_split_val)
#ax.plot(l[0,:], l[1,:], 'o', color='y')
l = w.clique_locations(count, w_split_val)
ax.plot(l[0,:], l[1,:], 'o', color='y')

#l = w.clique_locations(count+3, w_split_val)
#ax.plot(l[0,:], l[1,:], 'o', color='r')

ax.set_xlabel('KX GLOBAL')
ax.set_ylabel('KY GLOBAL')
ax.set_title('Location of cliques of size ' + str(count))
ax.set_xlim(0, w.nxGlobal)
ax.set_ylim(0, w.nyGlobal)
ax.grid(True)

plt.show()
