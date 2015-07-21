
"""
Plot the highest activity of four different bar positionings
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadSparse as rs
import PVReadWeights as rw
import PVConversions as conv
import scipy.cluster.vq as sp
import math


"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""
extended = False
vmax = 100.0 # Hz

if len(sys.argv) < 1:
   print "usage: plot_avg_activity b-vs-w histo"
   sys.exit()

last = np.genfromtxt(sys.argv[1], unpack=True, delimiter=";")


lasth = np.histogram(last, range=(0, np.max(last)), bins=100)
ca = lasth[0]
cb = lasth[1]
ca = np.insert(ca, len(ca), 0)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(cb, ca, '-o', color=cm.Blues(1.), linewidth=5.0)
ax.set_ylabel("number of patches")
ax.set_xlabel("(fbest-fworst)/(fbest+fworst) value")

plt.show()


sys.exit()
