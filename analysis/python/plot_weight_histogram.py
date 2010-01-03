"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import PVReadWeights as rw

sys.path.append('/Users/manghel/Documents/workspace/PetaVision/analysis/python/')

if len(sys.argv) < 2:
   print "usage: plot_weight_histogram filename"
   exit()

w = rw.PVReadWeights(sys.argv[1])
h = w.histogram()

low = 0
high = 0

for i in range(len(h)):
  if i < 126 and h[i] > 200: low += h[i]
  if i > 126 and h[i] > 200: high += h[i]

print "low==", low, "high==", high, "total==", np.add.reduce(h)

fig = plt.figure()
ax = fig.add_subplot(111, axisbg='darkslategray')

ax.plot(np.arange(len(h)), h, 'o', color='y')

ax.set_xlabel('WEIGHT BINS')
ax.set_ylabel('COUNT')
ax.set_title('Weight Histogram')
ax.set_xlim(0, 256)
ax.grid(True)

plt.show()
