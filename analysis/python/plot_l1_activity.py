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

extended = False

a1 = rs.PVReadSparse(sys.argv[1], extended)
end = int(sys.argv[2])
numofsteps = int(sys.argv[3])

nx = a1.nx
ny = a1.ny

numneur = nx * ny

activity = []
count = 0


counta=0
for k in range(end):
   A=a1.next_activity()
   #print "sum = ", np.sum(A)
   d = k / numofsteps
   #act = np.append(activity, np.sum(A))
   act = np.sum(A)

   if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
      if k == (numofsteps * d):
         A1p = act
         #print "k at first = ", k
      else:
         A1p = np.vstack((A1p,act))
   if k == (numofsteps-1):
      A1q = 0 #A1p.sum(axis=0) 
      #print A1q
   if k == ((numofsteps*d) + (numofsteps-1)): #and k != (numofsteps-1):
      A1q = np.vstack((A1q, A1p.sum(axis=0)))
      #print A1q


t1 = A1q / float(numneur)
#print t1
t1 = t1 / (numofsteps / 2000.0)
#print t1


fig = plt.figure()
ax = fig.add_subplot(111)
   
ax.plot(np.arange(np.shape(t1)[0]), t1, color='y', ls = '-')

plt.show() 

sys.exit()
