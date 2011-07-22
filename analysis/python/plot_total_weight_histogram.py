"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import PVReadWeights as rw

if len(sys.argv) < 3:
   print "usage: plot_weight_histogram filename1 filename2"
   exit()

w = rw.PVReadWeights(sys.argv[1])
wOff = rw.PVReadWeights(sys.argv[2])
h = w.histogram()
hOff = wOff.histogram()

low = 0
high = 0
lowOff = 0
highOff = 0

for i in range(len(h)):
  if i < 126 and h[i] > 200: low += h[i]
  if i > 126 and h[i] > 200: high += h[i]
for i in range(len(hOff)):
   if i < 126 and hOff[i] > 200: lowOff += hOff[i]
   if i > 126 and hOff[i] > 200: highOff += hOff[i]

print "On Weights: low==", low, "high==", high, "total==", np.add.reduce(h)
print "Off Weights: low ==", lowOff, "high==", highOff, "total==", np.add.reduce(hOff)


w_split_val = 255/2.
#if len(sys.argv) >= 3:
#   w_split_val = float(sys.argv[2])


chOn = w.clique_histogram(w_split_val)
chOff = wOff.clique_histogram(w_split_val)
print 'Clique On total =', sum(chOn)
print 'Clique Off total =', sum(chOff)




fig = plt.figure()


ax = fig.add_subplot(221, axisbg='darkslategray')
ax.plot(np.arange(len(h)), h, 'o', color='y')
ax.set_xlabel('WEIGHT BINS')
ax.set_ylabel('COUNT')
ax.set_title('On Weight Histogram')
ax.set_xlim(0, 256)
ax.grid(True)


ax = fig.add_subplot(222, axisbg='darkslategray')
ax.plot(np.arange(len(hOff)), hOff, 'o', color='y')
ax.set_xlabel('WEIGHT BINS')
ax.set_ylabel('COUNT')
ax.set_title('Off Weight Histogram')
ax.set_xlim(0, 256)
ax.grid(True)


ax = fig.add_subplot(223, axisbg='darkslategray')

ax.plot(np.arange(len(chOn)), chOn, 'o', color='y')

ax.set_xlabel('CLIQUE BINS')
ax.set_ylabel('COUNT')
ax.set_title('On Clique Histogram')
ax.set_xlim(0, 1+w.patchSize)
ax.grid(True)



ax = fig.add_subplot(224, axisbg='darkslategray')

ax.plot(np.arange(len(chOff)), chOff, 'o', color='y')

ax.set_xlabel('CLIQUE BINS')
ax.set_ylabel('COUNT')
ax.set_title('Off Clique Histogram')
ax.set_xlim(0, 1+w.patchSize)
ax.grid(True)


plt.show()
