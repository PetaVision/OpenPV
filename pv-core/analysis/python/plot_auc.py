"""
Plot AUC
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





auc1 = np.genfromtxt(sys.argv[1], unpack=True)
auc2 = np.genfromtxt(sys.argv[2], unpack=True)
auc3 = np.genfromtxt(sys.argv[3], unpack=True)
auc4 = np.genfromtxt(sys.argv[4], unpack=True)
auc5 = np.genfromtxt(sys.argv[5], unpack=True)
auc6 = np.genfromtxt(sys.argv[6], unpack=True)

"""
a = np.histogram(auc11, range=(0,1), bins=100)
print 
b = a[0]
c = a[1]
b = np.insert(b, [0], 0)

print np.shape(b)
print
print np.shape(c)
"""


x = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
x2 = [0.3, 0.45, 0.6, 0.75, 0.9]


fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(x2, auc1, color='b', linewidth=1)
ax.plot(x2, auc2, color='r', linewidth=1)
ax.plot(x, auc3, color='y', linewidth=1)
ax.plot(x, auc4, color='k', linewidth=1)
ax.plot(x, auc5, color='m', linewidth=1)
ax.plot(x, auc6, color='g', linewidth=1)

ax.axis([np.min(x), np.max(x), 0, 1])

ax.set_ylabel('auc value')
ax.set_xlabel('blue=reg 2.5mil, red=reg 3.5mil, magenta=reg 3.5mil \n yellow=expanded 0.5mil, black=expanded 1.0mil, green = expanded 1.5mil')


plt.show()

sys.exit()
