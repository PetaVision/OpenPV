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

w_split_val = 255/2.
if len(sys.argv) >= 3:
   w_split_val = float(sys.argv[2])

w = rw.PVReadWeights(sys.argv[1])
h = w.clique_histogram(w_split_val)
print 'total =', sum(h)

fig = plt.figure()
ax = fig.add_subplot(111, axisbg='darkslategray')

ax.plot(np.arange(len(h)), h, 'o', color='y')

ax.set_xlabel('CLIQUE BINS')
ax.set_ylabel('COUNT')
ax.set_title('Clique Histogram')
ax.set_xlim(0, 1+w.patchSize)
ax.grid(True)

plt.show()
