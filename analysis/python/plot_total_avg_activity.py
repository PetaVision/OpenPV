"""
Plot average activity as a function of time over an entire layer
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadSparse as rs

extended = True
vmax = 100.0 # Hz

if len(sys.argv) < 2:
   print "usage: plot_total_avg_activity filename"
   sys.exit()

activ = rs.PVReadSparse(sys.argv[1], extended)

numNeurons = activ.nxg_ex * activ.nyg_ex

time = []
activity = []

try:
    while True:
        r = activ.next_record()
        a = 1000 * len(r) / (numNeurons * activ.dt)
        activity.append(a)
        time.append(activ.time)
except MemoryError:
    print 'Finished reading', activ.timestep, ' record'

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('TIME')
ax.set_ylabel('AVERAGE ACTIVITY (Hz)')
ax.plot(time, activity, 'o')

plt.show()
