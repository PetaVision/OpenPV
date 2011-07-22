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


twhenf = []
for i in range(40):
   t = i * 25
   twhenf = np.append(twhenf, t)
print twhenf
activity = []
for i in range(500):
   if i%25 == 0:
      w = i
      e = w + 7.5
   if i >= w and i <= e:
      activity = np.append(activity, 1)
   else:
      activity = np.append(activity, 0)


fig = plt.figure()
ax = fig.add_subplot(111)
   
ax.set_title('On and Off K-means')

ax.set_autoscale_on(False)
ax.set_ylim(0,2)
ax.set_xlim(0, len(activity))
ax.plot(np.arange(len(activity)), activity, color='y', ls = '-')


plt.show()




