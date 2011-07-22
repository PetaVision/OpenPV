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


roc1 = np.genfromtxt(sys.argv[1], unpack=True)
#roc2 = np.genfromtxt(sys.argv[2], unpack=True)
"""
roc3 = np.genfromtxt(sys.argv[3], unpack=True)
roc4 = np.genfromtxt(sys.argv[4], unpack=True)
roc5 = np.genfromtxt(sys.argv[5], unpack=True)
roc6 = np.genfromtxt(sys.argv[6], unpack=True)
roc7 = np.genfromtxt(sys.argv[7], unpack=True)
roc8 = np.genfromtxt(sys.argv[8], unpack=True)
roc9 = np.genfromtxt(sys.argv[9], unpack=True)
roc10 = np.genfromtxt(sys.argv[10], unpack=True)
roc11 = np.genfromtxt(sys.argv[11], unpack=True)
roc12 = np.genfromtxt(sys.argv[12], unpack=True)
roc13 = np.genfromtxt(sys.argv[13], unpack=True)
roc14 = np.genfromtxt(sys.argv[14], unpack=True)
roc15 = np.genfromtxt(sys.argv[15], unpack=True)
roc16 = np.genfromtxt(sys.argv[16], unpack=True)
"""


print roc1
sys.exit()


seed1 = 0
seed2 = 0

for i in range(len(roc1)):
   if roc1[i] > 0.75:
      seed1+=1
#for i in range(len(roc2)):
#   if roc2[i] > 0.75:
#      seed2+=1

#res = float(seed1)/seed2
#print res
print seed1 * 10000 / 1000, "seconds"



roc1 = np.insert(roc1, 0, 1)
#roc2 = np.insert(roc2, 0, 1)

nm1 = []
nm2 = []

for i in range(99):
   a = roc1[i] * (99./(99-i+1))
   nm1 = np.append(nm1, a)
#for i in range(99):
#   a = roc2[i] * (99./(99-i+1))
#   nm2 = np.append(nm2, a)

#ac = plt.acorr(roc1, lw = 7, color='r', usevlines=False, linestyle='-',marker='.',
#maxlags=98)#,detrend=mlab.detrend)
#b = ac[0]
#c = ac[1]
#print "ROC1"
#print b[98:]
#print c[98:]
#bc = plt.acorr(roc2, lw = 7, color='r', usevlines=False, linestyle='-',marker='.',
#maxlags=98)#,detrend=mlab.detrend)
#d = bc[0]
#e = bc[1]
#print "ROC2"
#print d[98:]
#print e[98:]




#print nm1
#print nm2

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(np.arange(len(roc1)), roc1, '-o', color='k')
#ax.plot(np.arange(len(roc2)), roc2, '-o', color='r')


ax.set_xlabel('CLIQUE BINS')
ax.set_ylabel('COUNT')
ax.set_title('Clique Histogram')
#ax.set_xlim(0, 1+(np.max(res)/sh[1]))
ax.grid(True)

plt.show()
sys.exit()

ax2 = fig.add_subplot(212)
ax2.grid(True)
ax2.plot(np.arange(len(nm1)), nm1, '-o', color='k')
#ax2.plot(np.arange(len(nm2)), nm2, '-o', color='r')



"""
ax2 = fig.add_subplot(212)
ax2.grid(True)
ax2.acorr(roc1, lw = 7, color='k', usevlines=False, linestyle='-',marker='.', maxlags=98)#,detrend=mlab.detrend)
ax2.acorr(roc2, lw = 7, color='r', usevlines=False, linestyle='-',marker='.',
maxlags=98)#,detrend=mlab.detrend)
"""


plt.show()


