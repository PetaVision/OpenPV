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





roc1 = np.genfromtxt(sys.argv[1], unpack=True, delimiter=";")
#auc21 = np.genfromtxt(sys.argv[2], unpack=True)
#auc31 = np.genfromtxt(sys.argv[3], unpack=True)
#auc41 = np.genfromtxt(sys.argv[4], unpack=True)
#auc51 = np.genfromtxt(sys.argv[5], unpack=True)
print "unpacked"
"""
one = 0
zero = 0
a = roc1[0,:]
print a
print np.shape(a)
for i in range(len(a)):
   if a[i] == 1:
      one+=1
   if a[i] == 0:
      zero+=1
print "one = ", one
print "zero = ", zero
sys.exit()
"""
for i in range(2):
   roc1 = np.delete(roc1, 0, axis=0)

print np.shape(roc1)
a = np.split(roc1, 2, axis=1)[0]
b = np.split(roc1, 2, axis=1)[1]

print "a = ", a
print "b = ", b

c = np.histogram(a, range=(0, np.max(a)), bins=np.max(a))
d = np.histogram(b, range=(0, np.max(b)), bins=np.max(b))



ca = c[0]
cb = c[1]
da = d[0]
db = d[1]

ca = np.insert(ca, len(ca), 0)
da = np.insert(da, len(da), 0)



fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(cb, ca, '-o', color=cm.Blues(1.), linewidth=5.0)
ax.plot(db, da, '-o', color = 'r', linewidth=5.0)
ax.set_ylabel("")
ax.set_xlabel("Firing Rate")

plt.show()

sys.exit()
