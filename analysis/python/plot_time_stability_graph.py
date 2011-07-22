
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


ts1 = np.loadtxt(sys.argv[1])
ts2 = np.loadtxt(sys.argv[2])
ts3 = np.loadtxt(sys.argv[3])
ts4 = np.loadtxt(sys.argv[4])
ts5 = np.loadtxt(sys.argv[5])
ts6 = np.loadtxt(sys.argv[6])
#ts7 = np.loadtxt(sys.argv[7])
#ts8 = np.loadtxt(sys.argv[8])
#ts9 = np.loadtxt(sys.argv[9])
#ts10 = np.loadtxt(sys.argv[10])
#ts11 = np.loadtxt(sys.argv[11])
#ts12 = np.loadtxt(sys.argv[12])

"""
no1 = np.loadtxt(sys.argv[13])
no2 = np.loadtxt(sys.argv[14])
no3 = np.loadtxt(sys.argv[15])
no4 = np.loadtxt(sys.argv[16])
no5 = np.loadtxt(sys.argv[17])
no6 = np.loadtxt(sys.argv[18])
no7 = np.loadtxt(sys.argv[19])
no8 = np.loadtxt(sys.argv[20])
no9 = np.loadtxt(sys.argv[21])
no10 = np.loadtxt(sys.argv[22])
no11 = np.loadtxt(sys.argv[23])
no12 = np.loadtxt(sys.argv[24])
"""


where = []
for i in range(len(ts1)):
   where = np.append(where, i)
where2 = []
#for i in range(len(ts7)):
#   where2 = np.append(where2, i)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(where, ts1, '-o', color=cm.Greens(0.15), linewidth=5.0) #0.12
ax.plot(where, ts2, '-o', color=cm.Greens(0.3), linewidth=5.0) #0.2
ax.plot(where, ts3, '-o', color=cm.Greens(0.45), linewidth=5.0) #0.28
ax.plot(where, ts4, '-o', color=cm.Greens(0.6), linewidth=5.0) #0.36
ax.plot(where, ts5, '-o', color=cm.Greens(0.75), linewidth=5.0) #0.44
ax.plot(where, ts6, '-o', color=cm.Greens(0.9), linewidth=5.0)
"""
ax.plot(where2, ts7, '-o', color=cm.Blues(0.15), linewidth=5.0)
ax.plot(where2, ts8, '-o', color=cm.Blues(0.3), linewidth=5.0)
ax.plot(where2, ts9, '-o', color=cm.Blues(0.45), linewidth=5.0)
ax.plot(where2, ts10, '-o', color=cm.Blues(0.6), linewidth=5.0)
ax.plot(where2, ts11, '-o', color=cm.Blues(0.75), linewidth=5.0)
ax.plot(where2, ts12, '-o', color=cm.Blues(0.9), linewidth=5.0)
"""

"""
ax.plot(where, no1, '-o', color=cm.Reds(0.12), linewidth=5.0)
ax.plot(where, no2, '-o', color=cm.Reds(0.20), linewidth=5.0)
ax.plot(where, no3, '-o', color=cm.Reds(0.28), linewidth=5.0)
ax.plot(where, no4, '-o', color=cm.Reds(0.36), linewidth=5.0)
ax.plot(where, no5, '-o', color=cm.Reds(0.44), linewidth=5.0)
ax.plot(where, no6, '-o', color=cm.Reds(0.52), linewidth=5.0)
ax.plot(where, no7, '-o', color=cm.Reds(0.60), linewidth=5.0)
ax.plot(where, no8, '-o', color=cm.Reds(0.68), linewidth=5.0)
ax.plot(where, no9, '-o', color=cm.Reds(0.76), linewidth=5.0)
ax.plot(where, no10, '-o', color=cm.Reds(0.84), linewidth=5.0)
ax.plot(where, no11, '-o', color=cm.Reds(0.92), linewidth=5.0)
ax.plot(where, no12, '-o', color=cm.Reds(1.), linewidth=5.0)
"""


ax.set_ylabel("")
ax.set_xlabel("time steps\n Green=Regular Blue=Expanded Space")
ax.set_title("")

plt.show()



